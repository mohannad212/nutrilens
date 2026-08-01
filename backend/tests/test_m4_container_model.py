"""
tests/test_m4_container_model.py
===================================

Covers the real `ContainerModel` implementation (`app/ml/portion/container_model.py`)
that Stage 1 uses once the trained classifier lands:

- `RimAreaDetector` measures the enclosed pixel area of a synthetic container
  shape (filled circle / square) close to its true area, both when the shape is
  brighter than the background and when it is darker, and returns None on a
  blank frame.
- `TrainedContainerModel` maps the 2-class model labels (`plate`/`bowl`) onto the
  `container_priors` keys (`dinner_plate`/`bowl_large`) and passes an unknown
  label through as `unknown` (which the estimator reads as "no scale").
- End-to-end: a `PortionEstimator` wired with a stub classifier + the real rim
  detector upgrades an estimate to `method="container_prior"` scaled from the
  detected rim area — no torch weights needed.

The trained TorchScript classifier itself is validated by the training pipeline's
own eval (`training/portion`), not here — these tests stub it.
"""

import cv2
import numpy as np
from PIL import Image

from app.ml.portion import PortionEstimator, describe_scale_source
from app.ml.portion.container_model import (
    DEFAULT_LABEL_TO_PRIOR,
    RimAreaDetector,
    TrainedContainerModel,
)

IMG_W, IMG_H = 640, 480


def _circle_image(radius: int, fg=255, bg=0) -> Image.Image:
    arr = np.full((IMG_H, IMG_W), bg, dtype=np.uint8)
    cv2.circle(arr, (IMG_W // 2, IMG_H // 2), radius, fg, thickness=-1)
    return Image.fromarray(arr, "L").convert("RGB")


def _square_image(side: int, fg=255, bg=0) -> Image.Image:
    arr = np.full((IMG_H, IMG_W), bg, dtype=np.uint8)
    cx, cy, h = IMG_W // 2, IMG_H // 2, side // 2
    arr[cy - h : cy + h, cx - h : cx + h] = fg
    return Image.fromarray(arr, "L").convert("RGB")


# ------------------------------ RimAreaDetector ------------------------------
def test_rim_area_matches_circle_area():
    det = RimAreaDetector()
    r = 120
    area = det.detect(_circle_image(r))
    assert area is not None
    expected = np.pi * r * r
    assert abs(area - expected) / expected < 0.03  # contour area ~ within 3%


def test_rim_area_matches_square_area():
    det = RimAreaDetector()
    side = 240
    area = det.detect(_square_image(side))
    assert area is not None
    assert abs(area - side * side) / (side * side) < 0.02


def test_rim_area_detects_dark_container_on_light_surface():
    """Both polarities: a dark bowl on a light table must still be measured."""
    det = RimAreaDetector()
    area = det.detect(_circle_image(120, fg=0, bg=255))
    assert area is not None
    expected = np.pi * 120 * 120
    assert abs(area - expected) / expected < 0.03


def test_rim_area_returns_none_on_blank_frame():
    det = RimAreaDetector()
    blank = Image.new("RGB", (IMG_W, IMG_H), (127, 127, 127))
    assert det.detect(blank) is None


# --------------------------- label -> prior mapping --------------------------
class _StubClassifier:
    def __init__(self, label, conf=0.9):
        self._label, self._conf = label, conf

    def predict(self, image):
        return self._label, self._conf


def test_label_mapping_plate_and_bowl():
    plate = TrainedContainerModel(_StubClassifier("plate"))
    bowl = TrainedContainerModel(_StubClassifier("bowl"))
    assert plate.classify(_circle_image(100))[0] == "dinner_plate"
    assert bowl.classify(_circle_image(100))[0] == "bowl_large"
    # mapping table is the single source of truth
    assert DEFAULT_LABEL_TO_PRIOR == {"plate": "dinner_plate", "bowl": "bowl_large"}


def test_unknown_label_passes_through_as_unknown():
    model = TrainedContainerModel(_StubClassifier("spork"))
    assert model.classify(_circle_image(100))[0] == "unknown"


def test_confidence_is_preserved():
    model = TrainedContainerModel(_StubClassifier("plate", conf=0.77))
    assert model.classify(_circle_image(100)) == ("dinner_plate", 0.77)


# ------------------------ end-to-end via PortionEstimator --------------------
def test_estimator_uses_container_prior_from_trained_model():
    container_model = TrainedContainerModel(_StubClassifier("plate"), RimAreaDetector())
    est = PortionEstimator(container_model=container_model)
    # A clear circular plate so the rim detector returns a real area; the food
    # mask is a smaller central patch.
    image = _circle_image(150)
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[200:280, 280:360] = True
    r = est.estimate(image, "pizza", mask=mask)

    assert r.method == "container_prior"
    assert r.reference == "dinner_plate"
    assert r.min_g <= r.point_g <= r.max_g
    assert (
        describe_scale_source(r.method, r.reference)
        == "assumed dinner plate (no reference object in frame)"
    )


def test_unknown_container_falls_back_to_shape_only():
    container_model = TrainedContainerModel(_StubClassifier("spork"), RimAreaDetector())
    est = PortionEstimator(container_model=container_model)
    r = est.estimate(_circle_image(150), "pizza")
    assert r.method == "shape_only"  # unknown -> no Stage-1 scale
