"""
M0 sanity check (see 09-build-phases.md): downloads Food-101 (~5 GB, first run only),
loads the official split via torchvision, and renders one training batch to a PNG so a
human can eyeball that images/labels line up before any real training happens.

Usage:
  python verify_setup.py --data-root ./data --out batch_preview.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.utils import make_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out", default="batch_preview.png")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    print("Loading Food-101 official train split (downloads ~5 GB on first run)...")
    train_ds = Food101(args.data_root, split="train", download=True, transform=transform)
    print(f"Official train split loaded: {len(train_ds)} images, {len(train_ds.classes)} classes")
    assert len(train_ds.classes) == 101, "expected 101 Food-101 classes"

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    images, labels = next(iter(loader))

    grid = make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title(", ".join(train_ds.classes[i] for i in labels[:8]) + " ...")
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Batch of {len(images)} images rendered to {args.out}")


if __name__ == "__main__":
    main()
