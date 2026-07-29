"""
NutriLens — Container classifier (bowl vs plate), single-file Colab/Kaggle script.
Spec: 05-portion-estimation.md §3 (Stage 1) and §7 (aux-model training).

WHAT THIS IS
------------
Stage 1 of the portion pipeline needs to know *what the food sits in* to get a
coarse scale prior. This trains the smallest useful version of that: a 2-class
classifier — **bowl** vs **plate** — because the deadline is tight. It mirrors the
food-101 training recipe (transfer learning: warm up the head, then fine-tune),
but on a tiny MobileNetV3-Small backbone so it trains in minutes, not hours.

WHERE THE DATA COMES FROM (no manual labelling)
------------------------------------------------
`--collect` pulls images that already carry `Plate` / `Bowl` bounding boxes from
**Open Images V7** using FiftyOne, crops each box to a patch, and lays them out as
an ImageFolder:  data/containers/{train,val}/{bowl,plate}/*.jpg .
That is the whole "collect the data" step — automated, already labelled, free.

USAGE (Colab / Kaggle, T4 is plenty)
-------------------------------------
  pip install -q fiftyone
  python container_train.py --collect --per-class 1500     # ~a few minutes
  python container_train.py --train                         # ~5-10 min on a T4
  python container_train.py --export                        # TorchScript + labels
Or all at once:
  python container_train.py --collect --train --export --per-class 1500

OUTPUTS (in --out-dir, default ./artifacts)
--------------------------------------------
  container_classifier.pth   TorchScript (load with torch.jit.load — same as 04)
  labels_container.json      {"0": "bowl", "1": "plate"}
  metrics.json               {val_acc, per_class_acc, ...}

HOW IT PLUGS INTO THE BACKEND
------------------------------
The backend's `ContainerModel` protocol (app/ml/portion/__init__.py) needs
`classify(image) -> (class, confidence)`; this model provides exactly that. Map
its "plate"/"bowl" output to a container prior in `container_priors.py`
(plate -> dinner_plate, bowl -> bowl_large, say). The *rim pixel area* the scale
maths needs is a separate CV step (ellipse/contour fit), not this classifier.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

# ------------------------------- config --------------------------------------
SEED = 42
IMG_SIZE = 224  # MobileNetV3 native-ish; small = fast
# Open Images V7 detection label names -> our folder (lowercase) names.
OPEN_IMAGES_CLASSES = {"Bowl": "bowl", "Plate": "plate"}
CLASSES = sorted(OPEN_IMAGES_CLASSES.values())  # ["bowl", "plate"]
WARMUP_EPOCHS = 2  # head only, backbone frozen
FINETUNE_EPOCHS = 8  # unfreeze everything
WARMUP_LR = 1e-3
FINETUNE_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
VAL_FRACTION = 0.15
MIN_BOX_PX = 48  # skip tiny crops (unrecognisable, hurt training)
MIN_BOX_AREA_FRAC = 0.015  # and boxes < 1.5% of the image


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ---------------------- data collection (Open Images) ------------------------
def collect(data_root: str, per_class: int) -> None:
    """Download Plate/Bowl from Open Images V7, crop the boxes into an ImageFolder.

    Over-fetches images (each may hold several boxes) then caps to `per_class`
    crops per class, split deterministically into train/val by image id.
    """
    import fiftyone.zoo as foz  # heavy import; only needed for collection

    out = Path(data_root) / "containers"
    for split in ("train", "val"):
        for cls in CLASSES:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"Downloading Open Images V7 samples for {list(OPEN_IMAGES_CLASSES)} ...")
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=list(OPEN_IMAGES_CLASSES),
        max_samples=per_class * 4,  # over-fetch: many boxes get filtered/other-class
        shuffle=True,
        seed=SEED,
        dataset_name=f"oi_containers_{per_class}",
    )

    counts = {c: 0 for c in CLASSES}
    rng = random.Random(SEED)
    for sample in ds:
        detections = getattr(sample.ground_truth, "detections", None) or []
        try:
            img = Image.open(sample.filepath).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        split = "val" if rng.random() < VAL_FRACTION else "train"
        for det in detections:
            cls = OPEN_IMAGES_CLASSES.get(det.label)
            if cls is None or counts[cls] >= per_class:
                continue
            # FiftyOne bbox = [x, y, width, height] relative, top-left origin.
            bx, by, bw, bh = det.bounding_box
            if bw * bh < MIN_BOX_AREA_FRAC:
                continue
            x0, y0 = int(bx * w), int(by * h)
            x1, y1 = int((bx + bw) * w), int((by + bh) * h)
            if (x1 - x0) < MIN_BOX_PX or (y1 - y0) < MIN_BOX_PX:
                continue
            crop = img.crop((x0, y0, x1, y1))
            crop.save(out / split / cls / f"{sample.id}_{counts[cls]}.jpg", quality=90)
            counts[cls] += 1
        if all(counts[c] >= per_class for c in CLASSES):
            break

    print(f"Collected crops: {counts}  ->  {out}")
    if min(counts.values()) < 100:
        print("WARNING: very few crops for a class — raise --per-class or check network.")


# ------------------------------- datasets ------------------------------------
def build_loaders(data_root: str, batch_size: int, workers: int):
    root = Path(data_root) / "containers"
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.2),
        ]
    )
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(), norm])

    train_ds = ImageFolder(root / "train", transform=train_tf)
    val_ds = ImageFolder(root / "val", transform=eval_tf)
    # Enforce a stable class order matching CLASSES / labels json.
    assert train_ds.classes == CLASSES, f"unexpected classes: {train_ds.classes}"

    # Class-balanced sampling so a class imbalance in Open Images doesn't bias it.
    targets = np.array(train_ds.targets)
    class_count = np.bincount(targets, minlength=len(CLASSES))
    weights = (1.0 / np.clip(class_count, 1, None))[targets]
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(targets), replacement=True)

    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f"train={len(train_ds)} val={len(val_ds)} class_count={class_count.tolist()}")
    return train_loader, val_loader


# ------------------------------- model ---------------------------------------
def build_model() -> nn.Module:
    model = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = (not frozen) or name.startswith("classifier")


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = np.zeros(len(CLASSES))
    total = np.zeros(len(CLASSES))
    for imgs, labels in loader:
        pred = model(imgs.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), pred):
            total[t] += 1
            correct[t] += int(t == p)
    acc = correct.sum() / max(total.sum(), 1)
    per_class = (correct / np.clip(total, 1, None)).tolist()
    return acc, per_class


def train(data_root: str, out_dir: Path, batch_size: int, workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader = build_loaders(data_root, batch_size, workers)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    def run_phase(epochs, lr, freeze, tag):
        set_backbone_frozen(model, freeze)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best = 0.0
        for epoch in range(epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                opt.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                opt.step()
            sched.step()
            acc, per_class = evaluate(model, val_loader, device)
            best = max(best, acc)
            pc = {c: round(a, 3) for c, a in zip(CLASSES, per_class)}
            print(f"[{tag} {epoch + 1}/{epochs}] val_acc={acc:.4f} per_class={pc}")
        return best

    run_phase(WARMUP_EPOCHS, WARMUP_LR, True, "warmup")
    run_phase(FINETUNE_EPOCHS, FINETUNE_LR, False, "finetune")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "container_state.pth")
    acc, per_class = evaluate(model, val_loader, device)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "val_acc": acc,
                "per_class_acc": dict(zip(CLASSES, per_class)),
                "classes": CLASSES,
                "img_size": IMG_SIZE,
                "backbone": "mobilenet_v3_small",
            },
            indent=2,
        )
    )
    print(f"Saved raw weights + metrics.json (val_acc={acc:.4f}). Run --export next.")


# ------------------------------- export --------------------------------------
def export(out_dir: Path) -> None:
    """Trace to TorchScript so the backend loads it exactly like the food model."""
    state_path = out_dir / "container_state.pth"
    if not state_path.exists():
        raise SystemExit("No container_state.pth — run --train first.")
    model = build_model()
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    traced.save(str(out_dir / "container_classifier.pth"))
    (out_dir / "labels_container.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(CLASSES)}, indent=2)
    )
    print(f"Exported {out_dir}/container_classifier.pth + labels_container.json")


# --------------------------------- main --------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="download + crop the dataset")
    ap.add_argument("--train", action="store_true", help="train the classifier")
    ap.add_argument("--export", action="store_true", help="export TorchScript + labels")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-dir", default="./artifacts")
    ap.add_argument("--per-class", type=int, default=1500, help="crops per class to collect")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    seed_everything()
    if not (args.collect or args.train or args.export):
        ap.error("nothing to do — pass at least one of --collect / --train / --export")
    if args.collect:
        collect(args.data_root, args.per_class)
    if args.train:
        train(args.data_root, Path(args.out_dir), args.batch_size, args.workers)
    if args.export:
        export(Path(args.out_dir))


if __name__ == "__main__":
    main()
