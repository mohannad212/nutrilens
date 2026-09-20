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
from pathlib import Path

import numpy as np
import pytest
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

        def rim_area_px(self, image, food_mask=None):
            return 150_000.0

    det = CocoReferenceDetector(_stub_detect_fn([("fork", 0.9, (0, 0, 190, 0))]))
    est = PortionEstimator(container_model=StubContainer(), reference_detector=det)
    r = est.estimate(_img(), "pizza", mask=np.ones((IMG_H, IMG_W), dtype=bool))
    assert r.method == "reference_object"
    assert r.reference == "fork"


# ------------------ §2: the reference table must not overpromise ----------------
def test_reference_table_lists_only_detectable_classes():
    """`REFERENCE_SIZES` must contain nothing the pipeline cannot actually detect.

    The table originally carried all 8 references from `05 §4` while only 3 were
    reachable. A lookup entry that no detector can ever emit is not a harmless
    stub — it reads as a working feature to anyone auditing the code, and it is
    the kind of claim a report reader checks. Detectable now means: the COCO
    detector's cutlery, plus `credit_card` via the geometric card detector.
    """
    from app.ml.portion import reference_sizes as rs
    from app.ml.portion.reference_detector import DEFAULT_COCO_TO_REFERENCE

    detectable = set(DEFAULT_COCO_TO_REFERENCE.values()) | {"credit_card"}
    assert set(rs.REFERENCE_SIZES) == detectable


def test_unimplemented_references_are_documented_but_unreachable():
    """The rejected references stay documented, but can never resolve a scale."""
    from app.ml.portion import reference_sizes as rs

    assert set(rs.UNIMPLEMENTED_REFERENCES) == {"smartphone", "coin", "chopsticks", "hand"}
    for name, (size, reason) in rs.UNIMPLEMENTED_REFERENCES.items():
        # The research is retained...
        assert size.real_cm > 0 and len(reason) > 40, name
        # ...but the lookup path cannot reach it, so no scale can come from one.
        assert rs.size_for(name) is None, name
    assert rs.best_reference(["hand", "coin", "smartphone"]) is None


# ------------------- §1: credit-card reference detector -------------------------
def _card_image(long_px, size=(1200, 900), bg=(90, 95, 100), fill=(235, 235, 240), at=(60, 60)):
    """A flat pasted ID-1-ratio rectangle on a plain field.

    The canvas is deliberately large relative to the card: rule 1 rejects any
    candidate over 8% of the frame area (at that size it is the table or the
    plate, not a card beside a meal), so a cramped test canvas would fail the
    detector for the wrong reason.
    """
    from PIL import ImageDraw

    from app.ml.portion.reference_detector import CARD_RATIO

    img = Image.new("RGB", size, bg)
    x, y = at
    short = round(long_px / CARD_RATIO)
    ImageDraw.Draw(img).rectangle([x, y, x + long_px, y + short], fill=fill)
    return img


def test_card_ratio_is_the_iso_figure():
    """85.60 x 53.98 mm (ISO/IEC 7810 ID-1) — the value the whole detector keys on."""
    from app.ml.portion.reference_detector import CARD_LONG_MM, CARD_RATIO

    assert CARD_LONG_MM == 85.60
    assert CARD_RATIO == pytest.approx(1.5857, abs=1e-4)


def test_card_detector_measures_the_long_edge():
    """A card of known pixel size is measured to within a few percent."""
    from app.ml.portion.reference_detector import CardReferenceDetector

    det = CardReferenceDetector()
    found = det.detect(_card_image(200))
    assert found, "a clean 200 px card must be detected"
    cls, px = found[0]
    assert cls == "credit_card"
    assert px == pytest.approx(200, rel=0.05)


def test_card_below_the_pixel_floor_is_rejected():
    """Under 80 px a card is too coarse to be a useful ruler — and is where the
    false positives live. Rejecting is correct: the pipeline falls to the next rung."""
    from app.ml.portion.reference_detector import CardReferenceDetector

    assert CardReferenceDetector().detect(_card_image(60)) == []


def test_non_card_rectangle_is_rejected():
    """A rectangle of the wrong aspect ratio must not be read as a card.

    This is the failure that matters: at a 12% tolerance a kitchen scale's
    display panel (ratio 1.42) was accepted in a real photo, which would have
    produced a confidently wrong scale. 1.42 must be rejected at the shipped 6%.
    """
    from PIL import ImageDraw

    from app.ml.portion.reference_detector import CardReferenceDetector

    img = Image.new("RGB", (1200, 900), (90, 95, 100))
    long_px = 200
    short = round(long_px / 1.42)  # the kitchen-scale panel's measured ratio
    ImageDraw.Draw(img).rectangle([60, 60, 60 + long_px, 60 + short], fill=(235, 235, 240))
    assert CardReferenceDetector().detect(img) == []


def test_card_detector_never_raises_on_junk_input():
    """A detector failure must degrade to no-reference, never fail /recognise."""
    from app.ml.portion.reference_detector import CardReferenceDetector

    assert CardReferenceDetector().detect(Image.new("RGB", (4, 4), (0, 0, 0))) == []
    assert CardReferenceDetector().detect(Image.new("L", (64, 64), 128)) == []


def test_card_outranks_a_fork_when_both_are_present():
    """Priority 1 vs 4: the exact 85.60 mm reference must win over an assumed 19 cm fork.

    This is the point of §1 — the card sidesteps the fork's ~10% length variance
    (and therefore ~20% mass error), so when both are in frame the card must be
    the one that sets the scale.
    """
    from app.ml.portion.reference_detector import CompositeReferenceDetector

    class Stub:
        def __init__(self, dets):
            self._d = dets

        def detect(self, image):
            return list(self._d)

    composite = CompositeReferenceDetector(
        [Stub([("credit_card", 200.0)]), Stub([("fork", 400.0)])]
    )
    est = PortionEstimator(reference_detector=composite)
    r = est.estimate(Image.new("RGB", (640, 480)), "pizza", mask=np.ones((480, 640), bool))
    assert r.method == "reference_object"
    assert r.reference == "credit_card"


def test_composite_survives_one_detector_failing():
    """A broken detector must not silence the working one."""
    from app.ml.portion.reference_detector import CompositeReferenceDetector

    class Boom:
        def detect(self, image):
            raise RuntimeError("model gone")

    class Ok:
        def detect(self, image):
            return [("fork", 300.0)]

    assert CompositeReferenceDetector([Boom(), Ok()]).detect(Image.new("RGB", (64, 64))) == [
        ("fork", 300.0)
    ]


def test_card_detector_finds_no_card_in_the_weighed_photos():
    """Regression lock on the false-positive audit (scripts/card_fp_audit.py).

    Every weighed photo is a negative — none contains a card. A detection here
    means a threshold was loosened and a confidently wrong ruler is back.
    """
    from app.config import settings
    from app.ml.portion.reference_detector import CardReferenceDetector

    images = Path(__file__).resolve().parents[1] / "data" / "weighed" / "images"
    photos = sorted(images.glob("*.jpg"))
    if not photos:
        pytest.skip("weighed image set not present")
    det = CardReferenceDetector(
        ratio_tol=settings.card_ratio_tol, min_long_px=settings.card_min_long_px
    )
    for p in photos:
        assert det.detect(Image.open(p)) == [], f"false positive on {p.name}"
