"""
NutriLens — Food segmentation model (food vs background), Colab/Kaggle script.
Spec: 05-portion-estimation.md §6 (segmentation) and §7 (aux-model training).

WHAT THIS IS
------------
Trains the Stage-shared segmenter: a **DeepLabV3-MobileNetV3-Large** that outputs a
binary **food / not-food** mask. Recognition already identifies the dish, so we do
NOT need per-food-class masks — collapsing every food class to one "food" label
gives a cleaner, higher-IoU target and a trivial argmax at serve time.

DATA — FoodSeg103 (already pixel-labelled), no manual work
-----------------------------------------------------------
Loads FoodSeg103 via HuggingFace `datasets` and collapses its 104 class ids to
binary (any food id > 0 -> food). If the dataset id below ever changes, edit
`DATASET_ID` / the column autodetection.

USAGE (Colab / Kaggle, T4)
---------------------------
  pip install -q datasets
  python segmenter_train.py --train --out-dir ./artifacts     # ~20-40 min on a T4
  python segmenter_train.py --export --out-dir ./artifacts    # TorchScript

OUTPUTS (in --out-dir)
-----------------------
  food_segmenter.pth   TorchScript (returns logits; loads via torch.jit.load)
  metrics.json         {food_iou, pixel_acc, ...}

HOW IT PLUGS INTO THE BACKEND
------------------------------
Drop `food_segmenter.pth` into `backend/models/`. The lifespan loads it and
`app/ml/portion/segmenter.py` runs it — replacing the classical GrabCut fallback.
Serving resizes to 512x512, argmax over 2 channels, mask resized back to original.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SEED = 42
IMG_SIZE = 512
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
# FoodSeg103 on HuggingFace. If unavailable, swap for another food-seg dataset
# exposing an image + an integer-mask column (e.g. a UEC-FoodPix mirror).
DATASET_ID = "EduardoPacheco/FoodSeg103"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ------------------------------ dataset --------------------------------------
def _pick_column(features, candidates: list[str]) -> str:
    for c in candidates:
        if c in features:
            return c
    raise KeyError(f"none of {candidates} in dataset columns {list(features)}")


class FoodSegDataset(Dataset):
    """Wraps a HF split; yields (image_tensor, binary_mask_long) at IMG_SIZE."""

    def __init__(self, hf_split, image_col: str, mask_col: str, train: bool):
        self.ds = hf_split
        self.image_col = image_col
        self.mask_col = mask_col
        self.train = train

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        row = self.ds[i]
        img = row[self.image_col].convert("RGB")
        mask = row[self.mask_col]  # PIL with integer class ids (0 = background)

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        if self.train and random.random() < 0.5:  # aligned horizontal flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        x = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        m = np.asarray(mask)
        binary = torch.from_numpy((m > 0).astype(np.int64))  # any food id -> 1
        return x, binary


def build_loaders():
    from datasets import load_dataset

    print(f"Loading {DATASET_ID} via HuggingFace datasets ...")
    ds = load_dataset(DATASET_ID)
    train_key = "train"
    val_key = "validation" if "validation" in ds else "test"
    cols = ds[train_key].features
    image_col = _pick_column(cols, ["image", "img", "pixel_values"])
    mask_col = _pick_column(cols, ["label", "annotation", "mask", "segmentation", "labels"])
    print(f"image column='{image_col}' mask column='{mask_col}' | splits: {list(ds)}")

    train_ds = FoodSegDataset(ds[train_key], image_col, mask_col, train=True)
    val_ds = FoodSegDataset(ds[val_key], image_col, mask_col, train=False)
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


# ------------------------------- model ---------------------------------------
def build_model() -> nn.Module:
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    # ImageNet-pretrained backbone + a fresh 2-class head (background, food).
    return deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone="IMAGENET1K_V1", num_classes=2, aux_loss=False
    )


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Return (food_IoU, pixel_accuracy) on the validation split."""
    model.eval()
    inter = union = correct = total = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = model(imgs)["out"].argmax(1)
        food_p, food_g = pred == 1, masks == 1
        inter += (food_p & food_g).sum().item()
        union += (food_p | food_g).sum().item()
        correct += (pred == masks).sum().item()
        total += masks.numel()
    iou = inter / max(union, 1)
    return iou, correct / max(total, 1)


def train(out_dir: str) -> None:
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader = build_loaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.0
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs)["out"], masks)
            loss.backward()
            opt.step()
        sched.step()
        iou, acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch + 1}/{EPOCHS}] food_IoU={iou:.4f} pixel_acc={acc:.4f}")
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), outp / "segmenter_state.pth")
    (outp / "metrics.json").write_text(
        json.dumps({"food_iou": best_iou, "img_size": IMG_SIZE, "arch": "deeplabv3_mobilenet_v3_large"}, indent=2)
    )
    print(f"Saved best (food_IoU={best_iou:.4f}). Run --export next.")


# ------------------------------- export --------------------------------------
class _SegExport(nn.Module):
    """Wrap DeepLabV3 so the traced module returns the logits tensor, not a dict."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def export(out_dir: str) -> None:
    from pathlib import Path

    outp = Path(out_dir)
    state = outp / "segmenter_state.pth"
    if not state.exists():
        raise SystemExit("No segmenter_state.pth — run --train first.")
    model = build_model()
    model.load_state_dict(torch.load(state, map_location="cpu"))
    wrapper = _SegExport(model).eval()
    traced = torch.jit.trace(wrapper, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    traced.save(str(outp / "food_segmenter.pth"))
    print(f"Exported {outp}/food_segmenter.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--out-dir", default="./artifacts")
    args = ap.parse_args()
    seed_everything()
    if not (args.train or args.export):
        ap.error("nothing to do — pass --train and/or --export")
    if args.train:
        train(args.out_dir)
    if args.export:
        export(args.out_dir)


if __name__ == "__main__":
    main()
