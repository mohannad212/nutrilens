"""
tests/test_m4_portion.py
===========================

WHY THIS FILE EXISTS
---------------------
M4's Definition of Done (`09-build-phases.md`): `/recognise` returns a sensible
`{min_g, max_g, point_g, method}` and the *portion pipeline unit tests pass*.
`05 §8` further pins the pipeline's testability: "Unit-tested with synthetic masks
+ known scales." This file does both — it drives the `PortionEstimator` maths with
hand-controlled masks and scales (so the grams are checked against first-principles
arithmetic, not the implementation), and it drives the real `POST /recognise` route
to confirm the `portion` object is wired into the response contract (`03 §4`).

HOW IT CONTROLS THE SCALE
--------------------------
Scale (cm-per-pixel) is otherwise resolved from cues in the photo. To make the
maths deterministic the unit tests use the `shape_only` fallback, whose scale is a
pure function of the image width and `settings.portion_shape_only_frame_width_cm`
— so a mask of exactly N pixels has an exactly-known area in cm². Stage 1 and
Stage 2 are exercised with tiny stub models that return canned pixel spans, the
same way the M3 tests stub the classifier.

WHAT EACH TEST VALIDATES
--------------------------
- shape_only maths: a known mask + known framing scale yields the grams computed
  independently here (disc volume × density), widened by the ±35% band.
- method upgrades: wiring a container model → `container_prior` (±25%); wiring a
  reference detector → `reference_object` (±12%), including the priority pick and
  the agree-within-tolerance averaging (`05 §4`).
- always a range: `min_g ≤ point_g ≤ max_g` and the band matches the method.
- clamp: a huge mask is clamped to the class's plausible upper bound (`05 §5`).
- graceful degradation: with no mask and no models, `estimate` still returns a
  valid `shape_only` range rather than raising (`05 §10`).
- route wiring: `/recognise` includes a `portion` object when an estimator is
  wired, and `portion` is null when it is not (classification half unaffected).
"""

import io
import math

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.deps import get_classifier, get_current_user, get_portion_estimator
from app.main import create_app
from app.ml.portion import PortionEstimator, describe_scale_source
from app.ml.portion.densities import density_of

IMG_W, IMG_H = 640, 480
RECOGNISE_URL = "/api/v1/recognise"


# --------------------------------- helpers -----------------------------------
def _img() -> Image.Image:
    return Image.new("RGB", (IMG_W, IMG_H), (200, 200, 200))


def _rect_mask(pixels: int) -> np.ndarray:
    """A boolean mask with exactly `pixels` True cells (raster-filled)."""
    assert 0 <= pixels <= IMG_W * IMG_H
    flat = np.zeros(IMG_W * IMG_H, dtype=bool)
    flat[:pixels] = True
    return flat.reshape((IMG_H, IMG_W))


def _shape_only_cm_per_px() -> float:
    return settings.portion_shape_only_frame_width_cm / IMG_W


class StubContainerModel:
    def __init__(self, container_class="dinner_plate", rim_area_px=200_000.0):
        self._class = container_class
        self._area = rim_area_px

    def classify(self, image):
        return (self._class, 0.9)

    def rim_area_px(self, image):
        return self._area


class StubReferenceDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):
        return list(self._detections)


# ------------------------- Stage 3 maths (shape_only) ------------------------
def test_shape_only_mass_matches_first_principles():
    """A known mask + known framing scale gives the independently-computed grams."""
    est = PortionEstimator()  # no aux models -> shape_only
    # 60k px keeps the whole ±35% band inside pizza's [80, 900] g clamp, so this
    # test checks the raw band maths without the clamp engaging (that is covered
    # separately by test_mass_is_clamped_to_class_bounds).
    mask = _rect_mask(60_000)
    r = est.estimate(_img(), "pizza", mask=mask)

    # Independent maths: area = pixels * (cm/px)^2; pizza is a flat disc (h=1.5),
    # density = default 0.90 (pizza is not individually tuned in densities.py).
    cm_per_px = _shape_only_cm_per_px()
    area_cm2 = 60_000 * cm_per_px**2
    volume = area_cm2 * 1.5  # flat_disc: V = A * h
    mass = volume * density_of("pizza")

    assert r.method == "shape_only"
    assert r.reference is None
    assert r.point_g == round(mass, 1)
    assert r.min_g == round(mass * (1 - settings.portion_eps_shape_only), 1)
    assert r.max_g == round(mass * (1 + settings.portion_eps_shape_only), 1)
    assert r.min_g <= r.point_g <= r.max_g


def test_always_returns_a_range():
    est = PortionEstimator()
    r = est.estimate(_img(), "french_fries", mask=_rect_mask(30_000))
    assert r.min_g < r.point_g < r.max_g  # never a bare single number (05 §1)


def _round_area(diameter_cm: float) -> float:
    return math.pi * (diameter_cm / 2.0) ** 2


# ----------------------- Stage 1: container_prior ----------------------------
def test_container_model_upgrades_to_container_prior():
    est = PortionEstimator(container_model=StubContainerModel("dinner_plate", 200_000.0))
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))

    # Area-based scale from a round dinner plate: cm/px = sqrt(A_real / A_px),
    # A_real = π(27/2)². Band is range-derived: (24/27)² .. (30/27)².
    cm_per_px = math.sqrt(_round_area(27.0) / 200_000.0)
    area_cm2 = 50_000 * cm_per_px**2
    mass = area_cm2 * 1.5 * density_of("pizza")
    lo, hi = (24.0 / 27.0) ** 2, (30.0 / 27.0) ** 2

    assert r.method == "container_prior"
    assert r.reference == "dinner_plate"
    assert r.point_g == round(mass, 1)
    assert r.min_g == round(mass * lo, 1)
    assert r.max_g == round(mass * hi, 1)


def test_container_band_is_range_derived_and_asymmetric():
    """The container band comes from the size range, squared -> asymmetric (05 §3)."""
    est = PortionEstimator(container_model=StubContainerModel("dinner_plate", 150_000.0))
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))

    cm_per_px = math.sqrt(_round_area(27.0) / 150_000.0)
    mass = 50_000 * cm_per_px**2 * 1.5 * density_of("pizza")
    lo, hi = (24.0 / 27.0) ** 2, (30.0 / 27.0) ** 2

    assert r.min_g == round(mass * lo, 1)
    assert r.max_g == round(mass * hi, 1)
    # Squaring is non-linear: the upside (+24%) exceeds the downside (-21%).
    assert (r.max_g - r.point_g) > (r.point_g - r.min_g)


def test_square_container_scales_by_area_not_circle():
    """A rect container (takeaway box) uses side², not a circle's π r² (05 §3)."""
    est = PortionEstimator(container_model=StubContainerModel("takeaway_box", 100_000.0))
    r = est.estimate(_img(), "pizza", mask=_rect_mask(40_000))

    # takeaway_box is rect: real footprint = 15² = 225 cm² (NOT π(15/2)²=176.7).
    cm_per_px_sq = math.sqrt(15.0**2 / 100_000.0)
    mass_sq = 40_000 * cm_per_px_sq**2 * 1.5 * density_of("pizza")
    assert r.method == "container_prior"
    assert r.reference == "takeaway_box"
    assert r.point_g == round(mass_sq, 1)

    # Prove it is shape-aware: a circular assumption would give a different mass.
    cm_per_px_circle = math.sqrt(_round_area(15.0) / 100_000.0)
    mass_circle = 40_000 * cm_per_px_circle**2 * 1.5 * density_of("pizza")
    assert round(mass_sq, 1) != round(mass_circle, 1)


def test_unknown_container_falls_through_to_shape_only():
    est = PortionEstimator(container_model=StubContainerModel("unknown", 500.0))
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))
    assert r.method == "shape_only"  # unknown container carries no scale (05 §3)


# ----------------------- Stage 2: reference_object ---------------------------
def test_reference_detector_upgrades_and_picks_by_priority():
    # A fork (priority 4) and a credit card (priority 1) are both present; the
    # more reliable credit card must win (05 §4).
    detector = StubReferenceDetector([("fork", 300.0), ("credit_card", 100.0)])
    est = PortionEstimator(reference_detector=detector)
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))

    cm_per_px = 8.56 / 100.0  # credit card long edge / px
    area_cm2 = 50_000 * cm_per_px**2
    mass = area_cm2 * 1.5 * density_of("pizza")

    assert r.method == "reference_object"
    assert r.reference == "credit_card"
    assert r.point_g == round(mass, 1)
    assert r.min_g == round(mass * (1 - settings.portion_eps_reference), 1)


def test_two_agreeing_references_are_averaged():
    # fork at 19cm/190px = 0.10 cm/px; spoon at 18cm/180px = 0.10 cm/px — identical,
    # well within tolerance, so the scale is their average (still 0.10).
    detector = StubReferenceDetector([("fork", 190.0), ("spoon", 180.0)])
    est = PortionEstimator(reference_detector=detector)
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))

    s_fork, s_spoon = 19.0 / 190.0, 18.0 / 180.0
    cm_per_px = (s_fork + s_spoon) / 2.0
    area_cm2 = 50_000 * cm_per_px**2
    mass = area_cm2 * 1.5 * density_of("pizza")

    assert r.method == "reference_object"
    assert r.reference == "fork"  # highest priority among the two
    assert r.point_g == round(mass, 1)


# ------------------------------- clamp (05 §5) -------------------------------
def test_mass_is_clamped_to_class_bounds():
    # A reference object that makes the plate look enormous would blow the mass
    # up; the per-class clamp must cap a fried egg at its 90 g ceiling.
    detector = StubReferenceDetector([("credit_card", 20.0)])  # tiny px -> big cm/px
    est = PortionEstimator(reference_detector=detector)
    r = est.estimate(_img(), "fried_egg", mask=_rect_mask(50_000))
    assert r.point_g == 90.0  # clamp_max_g for fried_egg


# --------------------- graceful degradation (05 §10) -------------------------
def test_no_mask_no_models_still_returns_estimate():
    """Even with nothing to go on, the pipeline returns a valid shape_only range."""
    est = PortionEstimator()
    r = est.estimate(_img(), "grilled_salmon")  # no mask -> segmentation fallback
    assert r.method == "shape_only"
    assert r.min_g <= r.point_g <= r.max_g
    assert r.point_g > 0


def test_wider_band_for_weaker_scale_cue():
    """reference_object (±12%) must be tighter than shape_only (±35%)."""

    def rel_width(r):
        return (r.max_g - r.min_g) / r.point_g

    mask = _rect_mask(50_000)
    shape_only = PortionEstimator().estimate(_img(), "pizza", mask=mask)
    ref = PortionEstimator(
        reference_detector=StubReferenceDetector([("credit_card", 100.0)])
    ).estimate(_img(), "pizza", mask=mask)
    assert rel_width(ref) < rel_width(shape_only)


# ----------------------------- route wiring (03 §4) --------------------------
def _client(portion_estimator=None):
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: _StubClassifier()
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_portion_estimator] = lambda: portion_estimator
    return TestClient(app)


class _StubClassifier:
    def predict(self, image, top_k=3):
        return [("pizza", 0.95), ("lasagna", 0.03), ("garlic_bread", 0.01)][:top_k]


def _jpeg_bytes():
    rng = np.random.default_rng(0)
    arr = rng.integers(60, 200, (IMG_H, IMG_W, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _upload(client):
    return client.post(RECOGNISE_URL, files={"image": ("food.jpg", _jpeg_bytes(), "image/jpeg")})


def test_recognise_includes_portion_when_estimator_wired():
    resp = _upload(_client(PortionEstimator()))
    assert resp.status_code == 200
    portion = resp.json()["items"][0]["portion"]
    assert portion is not None
    assert portion["method"] in {"reference_object", "container_prior", "shape_only"}
    assert portion["min_g"] <= portion["point_g"] <= portion["max_g"]
    # scale source is surfaced for the client/report (no models wired -> framing).
    assert portion["scale_source"] == "image framing only (no reference object or known container)"


def test_scale_source_describes_each_stage():
    """describe_scale_source turns (method, reference) into a readable phrase (05 §4)."""
    assert describe_scale_source("reference_object", "fork") == "measured from a fork in the photo"
    assert (
        describe_scale_source("reference_object", "credit_card")
        == "measured from a credit card in the photo"
    )
    assert (
        describe_scale_source("container_prior", "dinner_plate")
        == "assumed dinner plate (no reference object in frame)"
    )
    assert "image framing only" in describe_scale_source("shape_only", None)


def test_reference_estimate_reports_reference_scale_source():
    est = PortionEstimator(reference_detector=StubReferenceDetector([("fork", 190.0)]))
    r = est.estimate(_img(), "pizza", mask=_rect_mask(50_000))
    assert describe_scale_source(r.method, r.reference) == "measured from a fork in the photo"


def test_recognise_omits_portion_when_no_estimator():
    resp = _upload(_client(portion_estimator=None))
    assert resp.status_code == 200
    assert resp.json()["items"][0]["portion"] is None


def test_math_helpers_are_consistent():
    """Sanity: the equivalent-radius helper round-trips area (guards 05 §5 maths)."""
    from app.ml.portion.shape_priors import _radius_from_area

    area = 123.4
    r = _radius_from_area(area)
    assert math.isclose(math.pi * r * r, area, rel_tol=1e-9)
