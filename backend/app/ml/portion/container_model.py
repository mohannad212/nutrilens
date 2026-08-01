"""
app/ml/portion/container_model.py
====================================

WHY THIS FILE EXISTS
---------------------
The `PortionEstimator` talks to Stage 1 through the `ContainerModel` protocol
(`classify(image)` + `rim_area_px(image)`). This file is the *real* implementation
of that protocol, wiring together the two independent halves it needs:

  1. **What** container it is — a trained TorchScript classifier
     (`container_classifier.pth`, produced by `training/portion/`), loaded exactly
     like the food model in `04`.
  2. **How big** it is in the image — a classical-CV rim-area detector
     (`RimAreaDetector`, pure OpenCV, no training), which measures the enclosed
     pixel area of the plate/bowl so the estimator can derive the area-based scale
     `cm_per_px = √(A_real / A_px)`.

WHAT PROBLEM IT SOLVES
------------------------
The classifier only says "plate" / "bowl"; on its own that gives no scale. Scale
also needs the container's pixel *area*, which is a geometry measurement, not a
classification — so it lives in a separate, testable OpenCV component here. This
file also bridges the naming gap: the 2-class model emits `plate`/`bowl`, but the
priors table (`container_priors.py`) keys on `dinner_plate`/`bowl_large`, so
`TrainedContainerModel` maps model labels onto prior keys.

HOW IT WORKS INTERNALLY
--------------------------
- `RimAreaDetector.detect(image)` thresholds the image (Otsu, both polarities so a
  bright plate on a dark table *and* a dark bowl on a light table both work), takes
  the largest sensible contour that isn't the whole frame, and returns its pixel
  area — or `None` if nothing plausible is found (estimator then degrades).
- `ContainerClassifier` is a thin TorchScript wrapper: eval transform → module →
  softmax → `(label, confidence)`.
- `TrainedContainerModel` implements the protocol: `classify` maps the model label
  to a prior key (`plate → dinner_plate`, `bowl → bowl_large`; anything else →
  `unknown`, which the estimator treats as "no scale"); `rim_area_px` delegates to
  the detector.
- `load_container_model(...)` is the `lru_cache`d factory the app lifespan calls
  once at startup (mirrors `load_classifier`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Splitting "class" (learned) from "rim area" (classical CV): each is testable in
  isolation — the detector against synthetic shapes with known areas, the mapping
  against a stub classifier — and the rim geometry needs no labelled data.
- Label→prior mapping in the adapter rather than retraining with prior-key labels:
  keeps the model's own label space simple and lets the mapping (and the size
  priors) be tuned without touching the weights.

COMMON STUDENT MISTAKES
--------------------------
- Returning the raw `plate` label to the estimator: it must be a `container_priors`
  key, or `prior_for` won't find a size and the scale silently vanishes.
- Measuring a single axis instead of the area: the estimator's scale is area-based
  (shape-agnostic), so this component returns an *area*, not a diameter.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# The 2-class model's labels -> `container_priors.py` keys. `plate` is treated as a
# dinner plate and `bowl` as a large bowl (the common cases); tune here freely.
DEFAULT_LABEL_TO_PRIOR: dict[str, str] = {
    "plate": "dinner_plate",
    "bowl": "bowl_large",
}


class RimAreaDetector:
    """Classical-CV enclosed pixel area of the container rim (`05 §3`)."""

    def __init__(self, min_area_frac: float = 0.05, max_area_frac: float = 0.95):
        # Ignore blobs smaller than this fraction of the frame (noise) or larger
        # than this (the background itself, not the container).
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac

    def detect(self, image: Image.Image) -> float | None:
        """Return the container rim's enclosed area in pixels, or None.

        Never raises — on any failure (or no plausible region) returns None so the
        estimator falls through to shape-only rather than erroring `/recognise`.
        """
        try:
            import cv2  # local import: keeps the module importable without cv2

            gray = np.asarray(image.convert("L"))
            h, w = gray.shape
            frame_area = float(h * w)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Try both polarities: the container may be brighter OR darker than the
            # surface it sits on.
            area = max(
                self._largest_region(cv2, binary, frame_area),
                self._largest_region(cv2, 255 - binary, frame_area),
            )
            if area < self.min_area_frac * frame_area:
                return None
            return area
        except Exception:
            return None

    def _largest_region(self, cv2: Any, binary: np.ndarray, frame_area: float) -> float:
        """Largest contour area below the full-frame cap (excludes the background)."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = 0.0
        cap = self.max_area_frac * frame_area
        for c in contours:
            a = float(cv2.contourArea(c))
            if a < cap:
                best = max(best, a)
        return best


class ContainerClassifier:
    """Thin TorchScript wrapper: image -> (label, confidence). Mirrors `04`."""

    def __init__(self, module: Any, idx_to_class: dict[str, str], transform: Any, device: Any):
        self.module = module
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.device = device

    def predict(self, image: Image.Image) -> tuple[str, float]:
        import torch

        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.module(x), dim=1)[0]
            conf, idx = probs.max(dim=0)
        return self.idx_to_class[str(int(idx.item()))], round(float(conf.item()), 4)


class TrainedContainerModel:
    """`ContainerModel` protocol impl: classifier + rim detector + label mapping."""

    def __init__(
        self,
        classifier: ContainerClassifier,
        rim_detector: RimAreaDetector | None = None,
        label_to_prior: dict[str, str] | None = None,
    ):
        self.classifier = classifier
        self.rim_detector = rim_detector or RimAreaDetector()
        self.label_to_prior = label_to_prior or DEFAULT_LABEL_TO_PRIOR

    def classify(self, image: Image.Image) -> tuple[str, float]:
        """Return `(container_prior_key, confidence)`.

        Maps the model's label to a `container_priors` key; an unmapped label
        becomes `unknown`, which the estimator reads as "no scale from Stage 1".
        """
        raw_label, conf = self.classifier.predict(image)
        return self.label_to_prior.get(raw_label, "unknown"), conf

    def rim_area_px(self, image: Image.Image) -> float | None:
        return self.rim_detector.detect(image)


def _build_container_transform() -> Any:
    """Eval transform matching `training/portion` (Resize 256 -> CenterCrop 224)."""
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@lru_cache(maxsize=1)
def load_container_model(weights_path: str, labels_path: str, device: str) -> TrainedContainerModel:
    """Load the TorchScript container classifier + build the full `ContainerModel`.

    Cached so the lifespan's startup load isn't repeated. Raises whatever
    `torch.jit.load` raises if the weights are missing/corrupt — the caller
    (lifespan) decides how to degrade.
    """
    import torch

    dev = torch.device(device)
    module = torch.jit.load(weights_path, map_location=dev).eval()
    idx_to_class: dict[str, str] = json.loads(Path(labels_path).read_text())
    classifier = ContainerClassifier(module, idx_to_class, _build_container_transform(), dev)
    return TrainedContainerModel(classifier)
