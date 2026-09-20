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

import math

import cv2
import numpy as np
import pytest
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


# --------------------- 2026-08 audit: rim-detector rules ---------------------
# Each rule below is pinned to a failure observed on real weighed photos, where
# the original "largest blob under a 95% cap" detector returned the granite
# *countertop* on 3/3 photos (border-hugging regions of 45-90% of the frame).


def test_rim_rejects_background_touching_borders():
    """A plate-sized blob that runs off the frame edges is background, not a rim."""
    arr = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    arr[:, : IMG_W // 2] = 255  # bright half-frame: touches 3 borders
    det = RimAreaDetector()
    assert det.detect(Image.fromarray(arr, "L").convert("RGB")) is None


def test_rim_uses_hull_area_when_food_bites_into_the_region():
    """A dark plate under bright food thresholds as a crescent; the rim's
    *enclosed* area is the convex hull, not the crescent."""
    arr = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)  # light table
    cv2.circle(arr, (IMG_W // 2, IMG_H // 2), 120, 0, thickness=-1)  # dark plate
    # bright "food" swallowing a big bite of the plate's colour region
    cv2.circle(arr, (IMG_W // 2 + 60, IMG_H // 2), 70, 255, thickness=-1)
    det = RimAreaDetector()
    area = det.detect(Image.fromarray(arr, "L").convert("RGB"))
    assert area is not None
    expected = np.pi * 120 * 120
    assert abs(area - expected) / expected < 0.10  # hull restores the full rim


def test_rim_rejects_candidate_smaller_than_the_food():
    """White plate on a light table: the only clean blob is the food itself.
    A 'container' smaller than the food on it is not a container."""
    arr = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    cv2.circle(arr, (IMG_W // 2, IMG_H // 2), 100, 40, thickness=-1)  # dark food blob
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    cv2.circle(mask.view(np.uint8), (IMG_W // 2, IMG_H // 2), 110, 1, thickness=-1)
    det = RimAreaDetector()
    assert det.detect(Image.fromarray(arr, "L").convert("RGB"), food_mask=mask) is None


def test_rim_rejects_candidate_disjoint_from_the_food():
    """A plate-like blob nowhere near the food (a napkin, a box) is not the
    container under it — containment, not just size, is required."""
    arr = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    cv2.circle(arr, (120, 120), 100, 40, thickness=-1)  # dark blob top-left
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[380:460, 480:600] = True  # food bottom-right, outside the blob
    det = RimAreaDetector()
    assert det.detect(Image.fromarray(arr, "L").convert("RGB"), food_mask=mask) is None


def test_rim_accepts_plate_with_partially_overflowing_food():
    """Tall food at an angle projects above the far rim (weighed donut photo:
    only 46% of the mask fell inside the plate ellipse). The containment floor
    must tolerate that while still holding the size rule."""
    arr = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
    cv2.circle(arr, (IMG_W // 2, IMG_H // 2), 150, 0, thickness=-1)  # dark plate
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    # food half-in, half-above the rim: containment ~0.5
    mask[IMG_H // 2 - 160 : IMG_H // 2, IMG_W // 2 - 40 : IMG_W // 2 + 40] = True
    det = RimAreaDetector()
    area = det.detect(Image.fromarray(arr, "L").convert("RGB"), food_mask=mask)
    assert area is not None
    expected = np.pi * 150 * 150
    assert abs(area - expected) / expected < 0.10


def test_detect_ellipse_reports_tilt_ratio():
    """An oblique plate projects an ellipse; minor/major ≈ cos(tilt) is the free
    tilt measurement the tall-food correction uses."""
    arr = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.ellipse(arr, (IMG_W // 2, IMG_H // 2), (160, 80), 0, 0, 360, 255, thickness=-1)
    det = RimAreaDetector()
    found = det.detect_ellipse(Image.fromarray(arr, "L").convert("RGB"))
    assert found is not None
    _area, tilt = found
    assert abs(tilt - 0.5) < 0.05  # 80/160
    # and a top-down circle reports ~1.0
    found2 = det.detect_ellipse(_circle_image(120))
    assert found2 is not None
    assert found2[1] > 0.95


# ---------------- §4: surface-growth rim detection (measured, off) --------------
def _plate_scene(plate_rgb=(240, 240, 240), bg=(70, 72, 75), size=(400, 400)):
    """A synthetic top-down scene: dark background, light round plate, dark food."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 120, plate_rgb, -1)  # plate
    cv2.circle(img, (w // 2, h // 2), 55, (60, 40, 30), -1)  # food
    food = np.zeros((h, w), bool)
    cv2.circle(food.view(np.uint8), (w // 2, h // 2), 55, 1, -1)
    return Image.fromarray(img), food


def test_surface_growth_finds_the_plate_around_the_food():
    """The mechanism works where its assumption holds: a uniform plate around food."""
    from app.ml.portion.container_model import SurfaceGrowthRimDetector

    img, food = _plate_scene()
    found = SurfaceGrowthRimDetector().detect_ellipse(img, food_mask=food)
    assert found is not None
    area, ratio = found
    # The plate is a r=120 disc; the hull of the grown region should be close.
    assert area == pytest.approx(math.pi * 120**2, rel=0.15)
    assert ratio == pytest.approx(1.0, abs=0.1)  # top-down => no tilt


def test_surface_growth_needs_a_food_seed():
    """Without a seed there is nothing to grow from — it must have no opinion."""
    from app.ml.portion.container_model import SurfaceGrowthRimDetector

    img, _ = _plate_scene()
    assert SurfaceGrowthRimDetector().detect_ellipse(img, food_mask=None) is None


def test_surface_growth_rejects_a_region_that_runs_off_the_frame():
    """If the surface does not separate from the counter, return None, not a guess.

    The whole safety argument for Stage 1 is that a doubtful rim is refused so the
    estimator falls to the honest wide band. A region touching several frame edges
    is the background, and was what made the countertop a ruler before the audit.
    """
    from app.ml.portion.container_model import SurfaceGrowthRimDetector

    # Plate colour == background colour: growth cannot stop at the rim.
    img, food = _plate_scene(plate_rgb=(70, 72, 75), bg=(70, 72, 75))
    assert SurfaceGrowthRimDetector().detect_ellipse(img, food_mask=food) is None


def test_chained_rim_detector_is_strictly_additive():
    """The fallback must never change a rim the primary already resolved."""
    from app.ml.portion.container_model import ChainedRimDetector

    class Primary:
        def detect_ellipse(self, image, food_mask=None):
            return (1234.0, 0.9)

    class Fallback:
        def detect_ellipse(self, image, food_mask=None):
            return (9999.0, 0.5)

    chain = ChainedRimDetector([Primary(), Fallback()])
    assert chain.detect_ellipse(Image.new("RGB", (64, 64)), food_mask=None) == (1234.0, 0.9)
    # ...and it does fall through when the primary declines.
    class Declines:
        def detect_ellipse(self, image, food_mask=None):
            return None

    assert ChainedRimDetector([Declines(), Fallback()]).detect_ellipse(
        Image.new("RGB", (64, 64))
    ) == (9999.0, 0.5)


def test_chained_rim_detector_survives_a_raising_detector():
    from app.ml.portion.container_model import ChainedRimDetector

    class Boom:
        def detect_ellipse(self, image, food_mask=None):
            raise RuntimeError("nope")

    class Ok:
        def detect_ellipse(self, image, food_mask=None):
            return (7.0, 1.0)

    assert ChainedRimDetector([Boom(), Ok()]).detect_ellipse(Image.new("RGB", (8, 8))) == (
        7.0,
        1.0,
    )


def test_surface_growth_stays_off_by_default():
    """Locks in a MEASURED negative result, so it cannot be switched on by drift.

    Enabling it took the weighed set from MAPE 18.2%/coverage 66.7% to
    37.5%/33.3%: it raises rim coverage 3/6 -> 6/6 but the rims are wrong (one
    hull hugged the food and missed the rim, another swallowed the kitchen scale
    under the plate). A wrong ruler is worse than no ruler. See the note on this
    setting in `app/config.py` for the full evidence.
    """
    from app.config import Settings

    assert Settings().rim_surface_growth_enabled is False
