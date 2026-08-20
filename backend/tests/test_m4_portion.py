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

import ast
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.deps import get_classifier, get_current_user, get_portion_estimator
from app.main import create_app
from app.ml.portion import PortionEstimator, describe_scale_source, shape_priors
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
    # 150k px keeps the whole ±35% band inside pizza's [80, 900] g clamp, so this
    # test checks the raw band maths without the clamp engaging (that is covered
    # separately by test_mass_is_clamped_to_class_bounds). The count is tied to
    # pizza's density: it was 60k while pizza fell through to the 0.90 default, and
    # had to rise when pizza got its own (lighter) 0.55, or the band's lower edge
    # would sink under the 80 g clamp and this test would be measuring the clamp.
    mask = _rect_mask(150_000)
    r = est.estimate(_img(), "pizza", mask=mask)

    # Independent maths: area = pixels * (cm/px)^2; pizza is a flat disc (h=1.5).
    # Density is read from the table rather than hard-coded so this stays a test of
    # the band arithmetic, not of the constant.
    cm_per_px = _shape_only_cm_per_px()
    area_cm2 = 150_000 * cm_per_px**2
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
    # 100k px (was 40k) so the mass clears pizza's 80 g clamp floor under its own
    # 0.55 density — otherwise both the square and circular masses below clamp to
    # the same 80.0 and the shape-awareness assertion passes for the wrong reason.
    r = est.estimate(_img(), "pizza", mask=_rect_mask(100_000))

    # takeaway_box is rect: real footprint = 15² = 225 cm² (NOT π(15/2)²=176.7).
    cm_per_px_sq = math.sqrt(15.0**2 / 100_000.0)
    mass_sq = 100_000 * cm_per_px_sq**2 * 1.5 * density_of("pizza")
    assert r.method == "container_prior"
    assert r.reference == "takeaway_box"
    assert r.point_g == round(mass_sq, 1)

    # Prove it is shape-aware: a circular assumption would give a different mass.
    cm_per_px_circle = math.sqrt(_round_area(15.0) / 100_000.0)
    mass_circle = 100_000 * cm_per_px_circle**2 * 1.5 * density_of("pizza")
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


# ------------------------- filled-bread family (05 §5) -----------------------
# A weighed 275 g chicken burger came back as 747 g. Neither `hamburger` nor any
# of its relatives had an entry in `densities.py` or `shape_priors.py`, so every
# one of them inherited `default` (0.90 g/cm³ — the mixed-food value, ~2x too
# heavy for anything built on bread) and `DEFAULT_SHAPE` (an `irregular_pile`
# whose height *grows* with footprint, which is wrong for a stacked item and makes
# mass scale as A^1.5). These two tests pin both halves of that fix.
FILLED_BREAD_FAMILY = [
    "hamburger",
    "club_sandwich",
    "grilled_cheese_sandwich",
    "pulled_pork_sandwich",
    "lobster_roll_sandwich",
    "hot_dog",
    "tacos",
    "breakfast_burrito",
    "chicken_quesadilla",
    "pizza",
    "donuts",
    "bread_pudding",
    "french_toast",
    "garlic_bread",
    "bruschetta",
]


def test_filled_bread_family_is_individually_tuned():
    """No member of the family may silently fall back to the generic defaults."""
    from app.ml.portion.densities import CLASS_DENSITY, DEFAULT_DENSITY
    from app.ml.portion.shape_priors import DEFAULT_SHAPE, SHAPE_PRIORS

    missing_density = [c for c in FILLED_BREAD_FAMILY if c not in CLASS_DENSITY]
    missing_shape = [c for c in FILLED_BREAD_FAMILY if c not in SHAPE_PRIORS]
    assert missing_density == [], f"no density entry: {missing_density}"
    assert missing_shape == [], f"no shape prior: {missing_shape}"

    for cls in FILLED_BREAD_FAMILY:
        # The whole point is that bread-based food is lighter than mixed food.
        assert density_of(cls) < DEFAULT_DENSITY, cls
        # `irregular_pile` is the model this family was wrongly using: it lets a
        # slightly-too-large mask inflate height as well as area.
        assert shape_priors.prior_for(cls).model is not DEFAULT_SHAPE.model, cls


def test_filled_bread_family_gives_plausible_servings():
    """A realistic footprint must yield a realistic number of grams.

    Guards the *values*, not just their presence — a density of 0.05 would pass
    the tuned-ness check above. Footprints are honest measured areas rather than
    "circle of diameter D": a hot dog occupies ~18x4.5 cm, and treating it as a
    15 cm disc would triple its area and make this test meaningless.
    """
    # (class, footprint cm2, plausible cooked-serving grams)
    cases = [
        ("hamburger", math.pi * 6.5**2, (180, 350)),
        ("club_sandwich", 150.0, (200, 400)),
        ("grilled_cheese_sandwich", 132.0, (120, 250)),
        ("pulled_pork_sandwich", math.pi * 6.5**2, (200, 400)),
        ("lobster_roll_sandwich", 90.0, (150, 300)),
        ("hot_dog", 81.0, (100, 220)),
        ("tacos", 168.0, (120, 300)),
        ("breakfast_burrito", 140.0, (250, 500)),
        ("chicken_quesadilla", math.pi * 9**2, (150, 350)),
        ("pizza", math.pi * 14**2, (400, 800)),
        ("donuts", math.pi * 5.5**2, (55, 140)),
        ("bread_pudding", 120.0, (120, 280)),
        ("french_toast", 200.0, (110, 250)),
        ("garlic_bread", 150.0, (60, 160)),
        ("bruschetta", 180.0, (50, 130)),
    ]
    assert {c for c, _, _ in cases} == set(FILLED_BREAD_FAMILY)

    for cls, area_cm2, (lo, hi) in cases:
        prior = shape_priors.prior_for(cls)
        volume = shape_priors.volume_cm3(prior.model, area_cm2, prior.h, prior.k)
        grams = shape_priors.clamp_mass(cls, volume * density_of(cls))
        assert lo <= grams <= hi, f"{cls}: {grams:.0f} g outside plausible {lo}-{hi} g"


# ------------- flat / shell-in / pastry families (05 §5) ---------------------
# The same audit that caught the burger found three families failing *harder*, and
# for the opposite reason. `DEFAULT_SHAPE`'s height grows with footprint
# (`h_eff = k·√A`), so the wider a food spreads the deeper it is assumed to be —
# backwards for anything sliced flat across a plate. Carpaccio over 380 cm² was
# modelled ~6.8 cm deep and pinned to the 1500 g clamp against a real ~120 g.
# Here the fix is the shape model; for the burger family it was the density.
#
# (class, footprint cm2, plausible cooked-serving grams)
FLAT_SHELL_PASTRY_CASES = [
    # thin / spread flat — dense food, wrongly modelled as a deep heap
    ("beef_carpaccio", 380.0, (80, 200)),
    ("peking_duck", 314.0, (180, 420)),
    ("foie_gras", 80.0, (50, 150)),
    # shell-in seafood — mask sees shells and pods, so effective density is low
    ("mussels", 254.0, (250, 550)),
    ("oysters", 380.0, (200, 450)),
    ("escargots", 200.0, (80, 200)),
    ("edamame", 154.0, (100, 250)),
    # airy baked / pastry
    ("beignets", 150.0, (70, 190)),
    ("churros", 180.0, (70, 190)),
    ("cannoli", 96.0, (70, 190)),
    ("macarons", 78.0, (30, 90)),
    ("baklava", 80.0, (60, 160)),
    ("cup_cakes", 77.0, (60, 170)),
]


def test_flat_shell_pastry_families_are_tuned_and_plausible():
    """These three families must not sit on the generic defaults."""
    from app.ml.portion.densities import CLASS_DENSITY
    from app.ml.portion.shape_priors import SHAPE_PRIORS

    for cls, area_cm2, (lo, hi) in FLAT_SHELL_PASTRY_CASES:
        assert cls in CLASS_DENSITY, f"{cls} has no density entry"
        assert cls in SHAPE_PRIORS, f"{cls} has no shape prior"
        prior = shape_priors.prior_for(cls)
        volume = shape_priors.volume_cm3(prior.model, area_cm2, prior.h, prior.k)
        grams = shape_priors.clamp_mass(cls, volume * density_of(cls))
        assert lo <= grams <= hi, f"{cls}: {grams:.0f} g outside plausible {lo}-{hi} g"


def test_flat_foods_do_not_deepen_as_they_spread():
    """The specific defect: a wider mask must not imply a deeper food.

    Under `IRREGULAR_PILE` mass scales as A^1.5, so doubling a flat food's
    footprint nearly tripled its mass. For a food that is genuinely flat, mass must
    scale linearly with area — this is what stops a slightly generous mask from
    running away, and it is the property `beef_carpaccio` lacked.
    """
    for cls in ("beef_carpaccio", "peking_duck", "foie_gras", "oysters", "macarons"):
        prior = shape_priors.prior_for(cls)
        single = shape_priors.volume_cm3(prior.model, 100.0, prior.h, prior.k)
        double = shape_priors.volume_cm3(prior.model, 200.0, prior.h, prior.k)
        assert double == pytest.approx(2 * single), f"{cls} is not area-linear"


# ----------------------------- egg dishes (05 §5) ----------------------------
def test_egg_dishes_are_tuned_and_plausible():
    """Caught by a weighed 284 g plate of halved eggs that estimated 714 g.

    Halved eggs are the clearest case for why `IRREGULAR_PILE` is the wrong
    default: they sit in a *single layer* bounded by the plate, so a wider
    arrangement must not imply a deeper serving. The scenarios below deliberately
    vary the footprint 3x for one class — a `√A` height model cannot stay in range
    across all of them, so this fails if anyone reverts the shape.
    """
    from app.ml.portion.densities import CLASS_DENSITY
    from app.ml.portion.shape_priors import SHAPE_PRIORS

    # (class, footprint cm2, plausible cooked-serving grams)
    cases = [
        ("deviled_eggs", 90.0, (110, 220)),  # 6 halves on a side plate
        ("deviled_eggs", 172.6, (210, 380)),  # 10 halves, the weighed 284 g plate
        ("deviled_eggs", 260.0, (330, 550)),  # 16 halves on a platter
        ("omelette", 150.0, (100, 200)),
        ("omelette", 216.0, (150, 280)),
        ("eggs_benedict", 157.0, (230, 420)),
        ("croque_madame", 168.0, (210, 390)),
    ]
    for cls, area_cm2, (lo, hi) in cases:
        assert cls in CLASS_DENSITY, f"{cls} has no density entry"
        assert cls in SHAPE_PRIORS, f"{cls} has no shape prior"
        prior = shape_priors.prior_for(cls)
        volume = shape_priors.volume_cm3(prior.model, area_cm2, prior.h, prior.k)
        grams = shape_priors.clamp_mass(cls, volume * density_of(cls))
        assert lo <= grams <= hi, f"{cls} @ {area_cm2:.0f} cm2: {grams:.0f} g not in {lo}-{hi}"


# ------------------- whole-label-set coverage (05 §5) ------------------------
def _food101_classes() -> list[str]:
    """The deployed label set, or skip — these guards are about *that* list."""
    path = Path(settings.model_dir) / "labels_food101.json"
    if not path.exists():
        pytest.skip(f"label set not present at {path}")
    labels = json.loads(path.read_text())
    return sorted(labels.values() if isinstance(labels, dict) else labels)


def test_every_class_has_a_density_and_a_shape_prior():
    """No class may rely on the generic fallbacks.

    `DEFAULT_DENSITY` (0.90) and `DEFAULT_SHAPE` (an `irregular_pile`) exist so an
    unknown class still yields *something* — that graceful degradation is tested
    in `test_no_mask_no_models_still_returns_estimate`. But for a class we ship,
    landing on them is a bug, not a default: it caused a 275 g burger to read
    747 g and a 284 g plate of eggs to read 714 g. This asserts the tables cover
    the deployed label set, so a newly added class fails loudly here rather than
    silently estimating ~3x heavy in production.
    """
    from app.ml.portion.densities import CLASS_DENSITY
    from app.ml.portion.shape_priors import SHAPE_PRIORS

    classes = _food101_classes()
    assert [c for c in classes if c not in CLASS_DENSITY] == []
    assert [c for c in classes if c not in SHAPE_PRIORS] == []


def test_no_class_gives_an_absurd_mass_for_a_normal_footprint():
    """A mid-sized plateful must be a believable number of grams for every class.

    Coverage alone is not correctness — a density of 0.01 would pass the test
    above. This is the cheap value check across all 101: at 150 cm² of food, every
    class must land in a plausible plated range. The ceiling is generous because
    the honest extreme is real: a large bowl of `pho` is mostly broth and weighs
    ~750 g.

    Deliberately **pre-clamp**. `clamp_mass` would drag a nonsense result back to
    the class's own floor or ceiling and hide exactly the defect this is looking
    for — an earlier version of this test asserted the clamped value and did not
    fail when a density was mutated to 0.01, because the 60 g floor caught it.
    Checking the raw `volume × density` tests the two tables themselves.
    """
    for cls in _food101_classes():
        prior = shape_priors.prior_for(cls)
        volume = shape_priors.volume_cm3(prior.model, 150.0, prior.h, prior.k)
        grams = volume * density_of(cls)  # NOT clamped — see the docstring
        assert 20.0 <= grams <= 900.0, f"{cls}: {grams:.0f} g for a 150 cm2 footprint"


def test_no_duplicate_keys_in_the_lookup_tables():
    """A repeated key silently keeps the last value and drops the earlier one.

    Both tables are long and grouped by food family, so the same class can easily
    be added twice under two different headings — which Python accepts. This
    parses the source rather than the imported dict, because by import time the
    duplicate is already gone and unobservable.
    """
    for module_path in (
        Path("app/ml/portion/densities.py"),
        Path("app/ml/portion/shape_priors.py"),
    ):
        if not module_path.exists():  # pragma: no cover - layout guard
            pytest.skip(f"{module_path} not found from the test's cwd")
        for node in ast.walk(ast.parse(module_path.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            assert dupes == [], f"{module_path.name} has duplicate keys: {dupes}"


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
