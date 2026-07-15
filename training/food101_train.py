"""
NutriLens — EfficientNet-B4 on Food-101, single-file Kaggle/Colab training script.
Spec: 04-ml-model.md. Target: 88%+ top-1 on the official test split.

Recipe (beyond the base spec, to push past ~86%):
  RandAugment + Random Erasing, MixUp/CutMix with soft-target CE, EMA weights,
  LR warmup -> cosine decay, longer fine-tune, horizontal-flip TTA at final eval.

Free-tier friendly: checkpoints every epoch and auto-resumes after disconnects.
Run the SAME command repeatedly until training completes.

Kaggle (recommended — attach the "food41" dataset as input to skip the 5 GB download):
  !python food101_train.py --data-root /kaggle/tmp/data --ckpt-dir /kaggle/working/ckpt
Colab:
  !python food101_train.py --data-root /content/data --ckpt-dir /content/drive/MyDrive/nutrilens/ckpt
Afterwards (test-set eval + TorchScript export):
  !python food101_train.py ... --export-only
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ----------------------------- configuration ---------------------------------
SEED = 42
IMG_SIZE = 380                # EfficientNet-B4 native resolution
NUM_CLASSES = 101
WARMUP_EPOCHS = 3             # phase 1: head only, backbone frozen
WARMUP_LR = 1e-3
FINETUNE_LR = 3e-4
LR_WARMUP_EPOCHS = 2          # linear LR warmup inside the fine-tune phase
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
PATIENCE = 8                  # early stop (generous: mixup curves are noisy)
VAL_PER_CLASS = 100           # 750 train/class -> 650 train / 100 val
EMA_DECAY = 0.999
MIX_PROB = 0.7                # fraction of batches that get MixUp or CutMix
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ------------------------------- dataset -------------------------------------
class Food101Split(Dataset):
    """Official Food-101 split. `items` = list of (relative_path, class_idx)."""

    def __init__(self, images_dir: Path, items: list, transform):
        self.images_dir = images_dir
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        img = Image.open(self.images_dir / f"{rel}.jpg").convert("RGB")
        return self.transform(img), label


def locate_food101(data_root: str) -> tuple[Path, Path]:
    """Return (images_dir, meta_dir). Prefers a Kaggle-attached dataset; else downloads."""
    candidates = [
        Path("/kaggle/input/food41"),                # kmader/food41 (meta under meta/meta)
        Path("/kaggle/input/food-101/food-101"),     # other Kaggle mirrors
        Path(data_root) / "food-101",                # torchvision download layout
    ]
    for base in candidates:
        if (base / "images").is_dir():
            for meta in (base / "meta" / "meta", base / "meta"):
                if (meta / "train.txt").is_file():
                    print(f"Found Food-101 at {base}")
                    return base / "images", meta
    print("Food-101 not found locally — downloading via torchvision (~5 GB, ~10 min)...")
    from torchvision.datasets import Food101
    Food101(data_root, split="train", download=True)
    base = Path(data_root) / "food-101"
    return base / "images", base / "meta"


def build_datasets(data_root: str, ckpt_dir: Path):
    images_dir, meta_dir = locate_food101(data_root)
    classes = sorted((meta_dir / "classes.txt").read_text().split())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def read_split(name):
        lines = (meta_dir / f"{name}.txt").read_text().split()
        return [(rel, class_to_idx[rel.split("/")[0]]) for rel in lines]

    train_items, test_items = read_split("train"), read_split("test")

    # Fixed validation carve-out: 100 imgs/class from train, seed 42, persisted.
    split_file = ckpt_dir / "split_val.json"
    if split_file.exists():
        val_rels = set(json.loads(split_file.read_text())["val_rels"])
    else:
        rng = random.Random(SEED)
        per_class: dict[int, list[str]] = {}
        for rel, y in train_items:
            per_class.setdefault(y, []).append(rel)
        val_rels = set()
        for rels in per_class.values():
            rng.shuffle(rels)
            val_rels.update(rels[:VAL_PER_CLASS])
        split_file.write_text(json.dumps({"seed": SEED, "val_rels": sorted(val_rels)}))

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    eval_tf = T.Compose([T.Resize(400), T.CenterCrop(IMG_SIZE), T.ToTensor(), norm])

    train_ds = Food101Split(images_dir, [it for it in train_items if it[0] not in val_rels], train_tf)
    val_ds = Food101Split(images_dir, [it for it in train_items if it[0] in val_rels], eval_tf)
    test_ds = Food101Split(images_dir, test_items, eval_tf)
    return train_ds, val_ds, test_ds, classes


# ------------------------------- model / EMA ---------------------------------
def build_model() -> nn.Module:
    model = M.efficientnet_b4(weights=M.EfficientNet_B4_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True),
                                     nn.Linear(in_features, NUM_CLASSES))
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


class ModelEMA:
    """Exponential moving average of weights — evaluates/exports better than raw weights."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.ema = copy.deepcopy(unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        src = unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(src[k])


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, p in unwrap(model).named_parameters():
        p.requires_grad = (not frozen) or name.startswith("classifier")


def make_optimizer(model: nn.Module, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


# --------------------------- mixup / cutmix / loss ----------------------------
def smooth_one_hot(labels: torch.Tensor) -> torch.Tensor:
    t = torch.full((labels.size(0), NUM_CLASSES), LABEL_SMOOTHING / (NUM_CLASSES - 1),
                   device=labels.device)
    return t.scatter_(1, labels.unsqueeze(1), 1.0 - LABEL_SMOOTHING)


def apply_mix(imgs: torch.Tensor, targets: torch.Tensor):
    """Randomly apply MixUp or CutMix to the batch; targets are soft (N,101)."""
    if random.random() > MIX_PROB:
        return imgs, targets
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    if random.random() < 0.5:  # MixUp
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        imgs = lam * imgs + (1 - lam) * imgs[perm]
    else:                      # CutMix
        lam = float(np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
        h, w = imgs.shape[2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm][:, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    return imgs, lam * targets + (1 - lam) * targets[perm]


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# ------------------------------ train / eval ---------------------------------
@torch.inference_mode()
def evaluate(model, loader, device, tta: bool = False):
    model.eval()
    top1 = top5 = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        if tta:  # horizontal-flip test-time augmentation
            out = (out + model(torch.flip(imgs, dims=[3]))) / 2
        _, pred5 = out.topk(5, dim=1)
        correct = pred5.eq(labels.unsqueeze(1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += labels.size(0)
    return top1 / n, top5 / n


@torch.inference_mode()
def per_class_accuracy(model, loader, device):
    model.eval()
    correct, total = torch.zeros(NUM_CLASSES), torch.zeros(NUM_CLASSES)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for imgs, labels in loader:
        pred = model(imgs.to(device)).argmax(dim=1).cpu()
        for t, p in zip(labels, pred):
            total[t] += 1
            confusion[t, p] += 1
            correct[t] += int(t == p)
    return (correct / total.clamp(min=1)).tolist(), confusion


def train_one_epoch(model, loader, optimizer, scaler, device, autocast,
                    grad_accum, ema, use_mix, tag):
    model.train()
    t0, running = time.time(), 0.0
    optimizer.zero_grad()
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = smooth_one_hot(labels.to(device, non_blocking=True))
        if use_mix:
            imgs, targets = apply_mix(imgs, targets)
        with autocast():
            loss = soft_cross_entropy(model(imgs), targets) / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
        running += loss.item() * grad_accum
        if step % 200 == 0:
            print(f"  [{tag}] step {step}/{len(loader)} loss={running / (step + 1):.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)


# ---------------------------- checkpoint / export ----------------------------
def save_ckpt(path, model, ema, optimizer, scheduler, scaler, epoch, phase,
              best_val, best_source, bad_epochs, classes):
    torch.save({
        "epoch": epoch, "phase": phase, "best_val": best_val,
        "best_source": best_source, "bad_epochs": bad_epochs, "classes": classes,
        "model_state": unwrap(model).state_dict(),
        "ema_state": ema.ema.state_dict() if ema else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict(),
    }, path)


def export(ckpt_dir: Path, test_loader, device, classes):
    best = torch.load(ckpt_dir / "best.pth", map_location=device)
    source = best.get("best_source") or "model"
    state = best["ema_state"] if source == "ema" and best.get("ema_state") else best["model_state"]
    model = build_model().to(device)
    model.load_state_dict(state)
    print(f"Exporting from '{source}' weights (val top-1 at save time: {best['best_val']:.4f})")

    print("Final evaluation on the official test split (25,250 images)...")
    t1, t5 = evaluate(model, test_loader, device)
    t1_tta, t5_tta = evaluate(model, test_loader, device, tta=True)
    print(f"TEST      top-1={t1:.4f} top-5={t5:.4f}")
    print(f"TEST+TTA  top-1={t1_tta:.4f} top-5={t5_tta:.4f}   (target >= 0.80, aiming 0.88+)")
    per_class, confusion = per_class_accuracy(model, test_loader, device)

    art = ckpt_dir / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "metrics.json").write_text(json.dumps({
        "test_top1": t1, "test_top5": t5,
        "test_top1_tta": t1_tta, "test_top5_tta": t5_tta,
        "weights_source": source, "image_size": IMG_SIZE, "seed": SEED,
    }, indent=2))
    with open(art / "per_class_acc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for c, a in zip(classes, per_class):
            w.writerow([c, f"{a:.4f}"])
    torch.save(confusion, art / "confusion_matrix.pt")

    model.eval().to("cpu")
    traced = torch.jit.trace(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    traced.save(str(ckpt_dir / "food101_efficientnet_b4.pth"))
    (ckpt_dir / "labels_food101.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(classes)}, indent=2))
    print(f"Done. Artifacts in {ckpt_dir}: food101_efficientnet_b4.pth, "
          f"labels_food101.json, artifacts/")


# ---------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--epochs-finetune", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto (16/GPU)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    seed_everything()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size or 16 * max(n_gpu, 1)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} x{n_gpu} | batch={batch_size} (accum x{args.grad_accum}) "
          f"| ckpt: {ckpt_dir}")

    train_ds, val_ds, test_ds, classes = build_datasets(args.data_root, ckpt_dir)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    loader_kw = dict(batch_size=batch_size, num_workers=args.workers,
                     pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    if args.export_only:
        export(ckpt_dir, test_loader, device, classes)
        return

    model = build_model().to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)  # T4 x2 on Kaggle -> ~2x throughput
    try:
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast("cuda")  # noqa: E731
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast

    # ---- resume ----
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best.pth"
    start_epoch, phase, best_val, best_source, bad = 0, "warmup", 0.0, "model", 0
    ckpt = None
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        phase, start_epoch = ckpt["phase"], ckpt["epoch"] + 1
        best_val, best_source, bad = ckpt["best_val"], ckpt["best_source"], ckpt["bad_epochs"]
        unwrap(model).load_state_dict(ckpt["model_state"])
        print(f"Resumed epoch {ckpt['epoch']} (phase={phase}, best={best_val:.4f})")

    ema = ModelEMA(model)
    if ckpt is not None and ckpt.get("ema_state"):
        ema.ema.load_state_dict(ckpt["ema_state"])

    # ---- phase 1: warm-up (head only, no mixup, no EMA benefit yet) ----
    if phase == "warmup":
        set_backbone_frozen(model, True)
        optimizer = make_optimizer(model, WARMUP_LR)
        if ckpt is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
        for epoch in range(start_epoch, WARMUP_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, scaler, device, autocast,
                            args.grad_accum, None, False, f"warmup {epoch + 1}/{WARMUP_EPOCHS}")
            v1, v5 = evaluate(model, val_loader, device)
            print(f"[warmup {epoch + 1}] val top-1={v1:.4f} top-5={v5:.4f}")
            save_ckpt(last_path, model, None, optimizer, None, scaler,
                      epoch, "warmup", best_val, best_source, 0, classes)
        phase, start_epoch, ckpt = "finetune", 0, None
        ema = ModelEMA(model)  # re-seed EMA from the warmed-up weights

    # ---- phase 2: fine-tune all layers ----
    set_backbone_frozen(model, False)
    optimizer = make_optimizer(model, FINETUNE_LR)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                           total_iters=LR_WARMUP_EPOCHS),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, T_max=args.epochs_finetune - LR_WARMUP_EPOCHS, eta_min=ETA_MIN)],
        milestones=[LR_WARMUP_EPOCHS])
    if ckpt is not None and ckpt["phase"] == "finetune":
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])

    for epoch in range(start_epoch, args.epochs_finetune):
        train_one_epoch(model, train_loader, optimizer, scaler, device, autocast,
                        args.grad_accum, ema, True,
                        f"finetune {epoch + 1}/{args.epochs_finetune}")
        scheduler.step()
        raw1, _ = evaluate(model, val_loader, device)
        ema1, ema5 = evaluate(ema.ema, val_loader, device)
        cand_val, cand_src = max((raw1, "model"), (ema1, "ema"))
        improved = cand_val > best_val
        if improved:
            best_val, best_source, bad = cand_val, cand_src, 0
            save_ckpt(best_path, model, ema, optimizer, scheduler, scaler,
                      epoch, "finetune", best_val, best_source, bad, classes)
        else:
            bad += 1
        print(f"[finetune {epoch + 1}] val raw={raw1:.4f} ema={ema1:.4f} "
              f"(top-5 {ema5:.4f}) best={best_val:.4f} [{best_source}] bad={bad}")
        save_ckpt(last_path, model, ema, optimizer, scheduler, scaler,
                  epoch, "finetune", best_val, best_source, bad, classes)
        if bad >= PATIENCE:
            print(f"Early stop: no improvement for {PATIENCE} epochs.")
            break

    export(ckpt_dir, test_loader, device, classes)


if __name__ == "__main__":
    main()
