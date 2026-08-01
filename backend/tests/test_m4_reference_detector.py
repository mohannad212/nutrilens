"""
tests/test_m4_reference_detector.py
======================================

Covers the COCO-based `ReferenceDetector` (`app/ml/portion/reference_detector.py`).
The real torchvision detector isn't exercised here (it downloads weights and is a
third-party model); instead we inject a stub `detect_fn` returning canned COCO
detections and assert the wrapper's own logic:

- only fork/spoon/knife are kept; other COCO classes (cell phone, bowl, person...)
  are ignored;
- detections below the score threshold are dropped;
- pixel length is the bounding-box **diagonal** (orientation-robust for thin
  cutlery), not a single side;
- a detector exception degrades to an empty list (estimator falls back);
- end-to-end, a `PortionEstimator` wired with the detector upgrades an estimate to
  `method="reference_object"` scaled from the utensil.
"""

import math

import numpy as np
from PIL import Image

from app.ml.portion import PortionEstimator, describe_scale_source
from app.ml.portion.reference_detector import (
    DEFAULT_COCO_TO_REFERENCE,
    CocoReferenceDetector,
    _box_diagonal,
)

IMG_W, IMG_H = 640, 480


def _img() -> Image.Image:
    return Image.new("RGB", (IMG_W, IMG_H), (200, 200, 200))


def _stub_detect_fn(detections):
    """detections: list of (coco_label, score, (x0,y0,x1,y1))."""

    def fn(image):
        return list(detections)

    return fn


# --------------------------------- geometry ----------------------------------
def test_box_diagonal_is_orientation_robust():
    # A 100x20 axis-aligned box vs the same rod at 45 deg (~85x85 box): the
    # diagonal stays close to the true ~102 length; a single side would not.
    axis = _box_diagonal((0, 0, 100, 20))
    diag45 = _box_diagonal((0, 0, 85, 85))
    assert math.isclose(axis, math.hypot(100, 20), rel_tol=1e-9)
    assert diag45 > 100  # ~120; a side (85) would badly under-read the length


# ----------------------------- mapping / filtering ---------------------------
def test_only_fork_spoon_knife_are_kept():
    det = CocoReferenceDetector(
        _stub_detect_fn(
            [
                ("fork", 0.9, (0, 0, 190, 30)),
                ("cell phone", 0.99, (0, 0, 200, 100)),  # ignored (out of scope)
                ("bowl", 0.95, (0, 0, 300, 300)),  # ignored
                ("spoon", 0.8, (10, 10, 40, 190)),
            ]
        )
    )
    out = dict(det.detect(_img()))
    assert set(out) == {"fork", "spoon"}
    assert DEFAULT_COCO_TO_REFERENCE == {"fork": "fork", "spoon": "spoon", "knife": "knife"}


def test_low_score_detections_dropped():
    det = CocoReferenceDetector(
        _stub_detect_fn([("knife", 0.3, (0, 0, 210, 20))]), score_threshold=0.5
    )
    assert det.detect(_img()) == []


def test_pixel_length_is_the_box_diagonal():
    det = CocoReferenceDetector(_stub_detect_fn([("fork", 0.9, (0, 0, 180, 40))]))
    out = det.detect(_img())
    assert len(out) == 1
    cls, length = out[0]
    assert cls == "fork"
    assert math.isclose(length, math.hypot(180, 40), rel_tol=1e-9)


def test_detector_exception_degrades_to_empty():
    def boom(image):
        raise RuntimeError("model died")

    assert CocoReferenceDetector(boom).detect(_img()) == []


# ------------------------ end-to-end via PortionEstimator --------------------
def test_estimator_uses_reference_object_from_detector():
    det = CocoReferenceDetector(_stub_detect_fn([("fork", 0.9, (0, 0, 190, 0))]))
    est = PortionEstimator(reference_detector=det)
    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[100:300, 100:400] = True
    r = est.estimate(_img(), "pizza", mask=mask)

    assert r.method == "reference_object"
    assert r.reference == "fork"
    assert r.min_g <= r.point_g <= r.max_g
    assert describe_scale_source(r.method, r.reference) == "measured from a fork in the photo"


def test_reference_beats_container_when_both_present():
    """Stage 2 wins over Stage 1: a fork present -> reference_object, not container."""

    class StubContainer:
        def classify(self, image):
            return ("dinner_plate", 0.9)

        def rim_area_px(self, image):
            return 150_000.0

    det = CocoReferenceDetector(_stub_detect_fn([("fork", 0.9, (0, 0, 190, 0))]))
    est = PortionEstimator(container_model=StubContainer(), reference_detector=det)
    r = est.estimate(_img(), "pizza", mask=np.ones((IMG_H, IMG_W), dtype=bool))
    assert r.method == "reference_object"
    assert r.reference == "fork"
