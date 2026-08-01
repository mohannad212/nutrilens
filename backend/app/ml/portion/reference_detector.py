"""
app/ml/portion/reference_detector.py
========================================

WHY THIS FILE EXISTS
---------------------
Stage 2 (`05 §4`) is the strongest scale cue: a known-size object in the frame
pins cm-per-pixel independently of the plate. This file is the *real*
`ReferenceDetector` implementation — it wraps an **off-the-shelf COCO object
detector** (no training needed) to find cutlery and report each one's pixel span,
which `PortionEstimator._reference_scale` turns into a precise scale.

SCOPE (deliberately narrow)
---------------------------
Restricted to the three utensils COCO already knows and that reliably appear with
a meal: **fork, spoon, knife**. (COCO also has "cell phone", but we skip it — a
phone's size varies by model and it is rarely in a food photo.) Everything else
COCO detects is ignored.

HOW IT WORKS INTERNALLY
--------------------------
- `CocoReferenceDetector.detect(image)` calls an injected `detect_fn` that returns
  raw `(coco_label, score, box)` triples, keeps only fork/spoon/knife above a
  score threshold, and reports each as `(reference_class, pixel_length)`.
- `pixel_length` is the bounding-box **diagonal**, `√(w² + h²)`, not a single
  side. Cutlery is thin and elongated, so at an angle its box goes nearly square
  and a side under-reads its true length by up to ~30%; the diagonal ≈ the real
  length at any orientation.
- `load_coco_reference_detector(...)` builds a pretrained torchvision detector
  (Faster R-CNN MobileNetV3, small + fast) and returns a detector wired to it.
  It is `lru_cache`d and only constructed if enabled (it downloads weights once),
  mirroring how the classifier loads.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Off-the-shelf COCO detector over a custom-trained one: fork/spoon/knife are
  already COCO classes, so we get Stage 2 with **zero training** — the fastest path
  to real reference-object scaling (`05 §7` explicitly suggests this shortcut).
- Injected `detect_fn` (not a hard torchvision dependency inside the class): lets
  the mapping + geometry be unit-tested with a stub, and keeps the heavy model out
  of import time.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "A fork's size varies (dinner vs salad fork) — isn't the scale wrong?"
A: Bounded: cutlery varies far less than crockery, forks are ranked below exact
   references (credit card/coin) in `reference_sizes.py`, and the ±band + user
   slider absorb the rest. Still tighter than the container fallback.

Q: "Why the box diagonal for pixel length?"
A: Orientation robustness. A thin rod at 45° makes an almost-square box; the
   diagonal stays ≈ the rod's true length, a single side does not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from PIL import Image

# COCO label -> our `reference_sizes.py` key. Only the three utensils we trust.
DEFAULT_COCO_TO_REFERENCE: dict[str, str] = {
    "fork": "fork",
    "spoon": "spoon",
    "knife": "knife",
}

# detect_fn contract: image -> list of (coco_label, score, (x0, y0, x1, y1)).
DetectFn = Callable[[Image.Image], list[tuple[str, float, tuple[float, float, float, float]]]]


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    """Bounding-box diagonal `√(w² + h²)` — orientation-robust length of a thin object."""
    x0, y0, x1, y1 = box
    return math.hypot(x1 - x0, y1 - y0)


class CocoReferenceDetector:
    """`ReferenceDetector` protocol impl over an off-the-shelf COCO detector."""

    def __init__(
        self,
        detect_fn: DetectFn,
        coco_to_reference: dict[str, str] | None = None,
        score_threshold: float = 0.5,
    ):
        self.detect_fn = detect_fn
        self.coco_to_reference = coco_to_reference or DEFAULT_COCO_TO_REFERENCE
        self.score_threshold = score_threshold

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        """Return `[(reference_class, pixel_length), ...]` for fork/spoon/knife found.

        Ignores non-utensil detections and anything below the score threshold.
        Never raises — a detector failure yields an empty list so the estimator
        falls back to the container/shape-only stages.
        """
        try:
            raw = self.detect_fn(image)
        except Exception:
            return []
        out: list[tuple[str, float]] = []
        for label, score, box in raw:
            ref = self.coco_to_reference.get(label)
            if ref is None or score < self.score_threshold:
                continue
            length = _box_diagonal(box)
            if length > 0:
                out.append((ref, length))
        return out


def _make_torchvision_detect_fn(model: Any, categories: list[str], device: Any) -> DetectFn:
    """Wrap a torchvision detection model into the `detect_fn` contract."""
    import torch
    import torchvision.transforms.functional as F

    def detect_fn(
        image: Image.Image,
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        x = F.to_tensor(image.convert("RGB")).to(device)  # detector normalises internally
        with torch.inference_mode():
            pred = model([x])[0]
        results = []
        for label_idx, score, box in zip(
            pred["labels"], pred["scores"], pred["boxes"], strict=False
        ):
            name = categories[int(label_idx.item())]
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            results.append((name, float(score.item()), (x0, y0, x1, y1)))
        return results

    return detect_fn


@lru_cache(maxsize=1)
def load_coco_reference_detector(device: str, score_threshold: float) -> CocoReferenceDetector:
    """Build a pretrained COCO detector (downloads weights once) for Stage 2.

    Uses Faster R-CNN MobileNetV3-Large-320 — small and fast enough for CPU serving
    while still detecting cutlery well. Cached so startup loads it only once.
    """
    import torch
    import torchvision

    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval().to(torch.device(device))
    categories: list[str] = weights.meta["categories"]  # COCO names, index-aligned
    detect_fn = _make_torchvision_detect_fn(model, categories, torch.device(device))
    return CocoReferenceDetector(detect_fn, score_threshold=score_threshold)
