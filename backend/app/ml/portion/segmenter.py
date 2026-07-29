"""
app/ml/portion/segmenter.py
==============================

WHY THIS FILE EXISTS
---------------------
Every portion estimate needs the food's pixel mask to measure its area (`05 §6`).
This file is the *real* `Segmenter` — a trained TorchScript food-vs-background
segmentation model (DeepLabV3-MobileNetV3, produced by `training/portion/`),
loaded exactly like the classifier in `04`. It replaces the crude classical
fallback in the estimator whenever the weights are present.

WHAT PROBLEM IT SOLVES
------------------------
The classical fallback (Otsu/GrabCut) tends to grab the *plate*, not the food, so
the area — and therefore the mass — is over-estimated. A network trained on food
masks learns "which pixels are food" directly, across bowls, mixed plates, and
colored crockery, which is what the scale→area→mass chain actually needs.

HOW IT WORKS INTERNALLY
--------------------------
- `segment(image)` resizes to the model's fixed input, runs the network, takes the
  argmax over the 2 output channels (0 = background, 1 = food), then **resizes the
  binary mask back to the original image size** (nearest-neighbour, so it stays a
  clean 0/1 mask) — the estimator counts these pixels for the area.
- The served TorchScript module is expected to return the logits tensor directly
  (the training export wraps DeepLabV3 to return its `"out"` tensor), but we still
  defensively unwrap a dict if one comes through.
- `load_food_segmenter(...)` is the `lru_cache`d startup factory (mirrors
  `load_classifier` / `load_container_model`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Binary food/not-food (chosen) over per-food-class masks: recognition already
  identified the dish, so all we need is its silhouette; collapsing the classes
  gives a cleaner, higher-IoU target and a trivial argmax at serve time.
- Nearest-neighbour resize of the *mask* back to full resolution rather than
  measuring area at the model's low resolution: keeps the area in the original
  pixel space, consistent with how the reference/container scale is measured.

COMMON STUDENT MISTAKES
--------------------------
- Bilinear-resizing the mask (introduces fractional 0..1 values and fuzzy edges) —
  use nearest so it stays binary.
- Measuring area on the resized-down mask and forgetting to scale it back to the
  original resolution the scale (cm/px) was computed in.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

# Fixed square input the model is trained and served at (keep train/serve identical).
SEG_INPUT_SIZE = 512


class FoodSegmenter:
    """TorchScript food-vs-background segmenter (`Segmenter` protocol, `05 §6`)."""

    def __init__(self, module: Any, transform: Any, device: Any, input_size: int = SEG_INPUT_SIZE):
        self.module = module
        self.transform = transform
        self.device = device
        self.input_size = input_size

    def segment(self, image: Image.Image) -> np.ndarray:
        """Return a boolean food mask the same H×W as `image`."""
        import torch

        rgb = image.convert("RGB")
        w, h = rgb.size
        with torch.inference_mode():
            x = self.transform(rgb).unsqueeze(0).to(self.device)
            out = self.module(x)
            if isinstance(out, dict):  # defensively unwrap DeepLabV3's {"out": ...}
                out = out["out"]
            pred = out.argmax(dim=1)[0].detach().to("cpu").numpy().astype(np.uint8)
        # Resize the binary mask back to the original resolution (nearest = stays 0/1).
        mask_img = Image.fromarray(pred * 255).resize((w, h), Image.NEAREST)
        return np.asarray(mask_img) > 127


def _build_seg_transform(input_size: int = SEG_INPUT_SIZE) -> Any:
    """Eval transform matching `training/portion` segmenter training."""
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@lru_cache(maxsize=1)
def load_food_segmenter(weights_path: str, device: str) -> FoodSegmenter:
    """Load the TorchScript food segmenter once at startup (mirrors `load_classifier`)."""
    import torch

    dev = torch.device(device)
    module = torch.jit.load(weights_path, map_location=dev).eval()
    return FoodSegmenter(module, _build_seg_transform(), dev)
