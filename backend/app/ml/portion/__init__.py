"""
app/ml/portion/__init__.py
=============================

WHY THIS FILE EXISTS
---------------------
This is the serving entry point for the whole portion-estimation pipeline
(`05-portion-estimation.md`) — the single object `recognition_service` calls to
turn "a photo + a predicted food class" into a calibrated **mass range in grams**.
The three stages (container prior, reference-object calibration, shape priors) and
the tables in this package are orchestrated here into one deterministic,
side-effect-free `estimate(...)` call, matching the serving interface sketched in
`05 §8`.

WHAT PROBLEM IT SOLVES
------------------------
A 2D photo has no absolute scale (`05 §1`). The estimator resolves scale from the
best available cue — a known-size reference object (best), else the container's
size prior, else a device-normalised framing heuristic (worst) — measures the
food's area in cm² from a segmentation mask, lifts that area to a volume via the
food's shape prior, multiplies by density to get mass, and finally widens that
mass into an honest range whose width reflects *which* scale stage produced it.
It always returns a range and records the `method`, so the caller never presents
fake precision (`05 §Output contract`).

HOW IT WORKS INTERNALLY
--------------------------
`PortionEstimator.estimate(image, food_class, mask=None)`:
  1. `_resolve_scale(image)` → `(cm_per_px, method, reference, band)`. It tries, in
     reliability order: Stage 2 reference detector (if wired) → Stage 1 container
     model (if wired) → shape-only framing heuristic. Each rung is optional; with
     no trained aux models wired the estimator degrades to `shape_only` and still
     returns a valid (wider) range — exactly the graceful degradation of `05 §10`.
     The container rung derives scale from *area* (`√(A_real/A_px)`, shape-aware —
     square boxes as well as round plates) and its `band` from the container's own
     min/max size range, squared (mass ∝ size²).
  2. `region = mask if given else self._segment(image)` — the food's pixel mask.
     A caller (or the trained segmenter) may pass a mask; otherwise a classical
     OpenCV fallback (`05 §6`) is used, and if even that fails we fall back to a
     central region of the frame rather than crashing.
  3. `_area(region, cm_per_px)` → area in cm² (pixel count × cm-per-px²).
  4. `_volume(food_class, area_cm2)` → cm³ via the class's shape prior (`05 §5`).
  5. `mass = volume × density_of(food_class)`, clamped to the class's plausible
     band, then widened by the stage's `band` into `{min_g, max_g, point_g}` — a
     tight symmetric band for a reference object, the range-derived asymmetric band
     for a container prior, the widest band for shape-only. The min/max bounds are
     themselves clamped to the class's plausible range.
Nothing is written to disk; the mask lives only in memory (`05 §6/§8`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Optional, duck-typed aux models (`container_model` / `reference_detector` /
  `segmenter` may all be `None`) rather than hard dependencies: the offline ML
  track delivers those weights on its own schedule (like the classifier in `04`),
  and the API must return *some* portion the moment segmentation works (`05 §10`).
  Wiring a model later strictly tightens the range without changing this contract.
- Deriving every geometric parameter from one measured area (a circle of
  equivalent area where a radius is needed): keeps the pipeline driven by a single
  observable and makes the unit tests — synthetic masks + known scales — able to
  pin the maths exactly (`05 §8`).

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "With no reference object and no trained container model, is this just a
    guess?"
A: It's an honest wide-band estimate: `method="shape_only"`, ε≈±35%, and the user
   always confirms/adjusts via the slider (FR06). The pipeline is built so that
   adding a reference object or the container model automatically narrows the band
   and upgrades the `method` — the range never pretends to more precision than the
   cue it came from (`05 §4, §9`).

Q: "Why return a range at all instead of the single `point_g`?"
A: Monocular volume from one uncalibrated image is inherently ambiguous (`05 §1`).
   The range is the truthful representation of that ambiguity; `point_g` is only
   the slider's default position, not a claim of exactness.

COMMON STUDENT MISTAKES
--------------------------
- Persisting the mask or the image to measure it — everything stays in memory.
- Forgetting to clamp: a segmentation that leaks onto the plate doubles the area
  and the mass; the per-class clamp (`shape_priors.clamp_mass`) rejects that.
- Treating `point_g` as ground truth in downstream code — it is the midpoint of a
  deliberately wide band the user is expected to adjust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from app.config import settings
from app.ml.portion import container_priors, reference_sizes, shape_priors
from app.ml.portion.densities import density_of

__all__ = [
    "PortionEstimate",
    "PortionEstimator",
    "PortionMethod",
    "ContainerModel",
    "ReferenceDetector",
    "Segmenter",
    "describe_scale_source",
]

# Which scale stage governed an estimate (05 §4). Kept in lock-step with the
# `Literal` on `schemas.recognition.Portion.method` so the pipeline and the wire
# contract can never drift.
PortionMethod = Literal["reference_object", "container_prior", "shape_only"]


def describe_scale_source(method: PortionMethod, reference: str | None) -> str:
    """Human-readable explanation of *how the scale was obtained* (05 §4).

    `method` + `reference` already encode this, but the app/report shouldn't have
    to map them itself — this is the single place that turns the pair into a phrase
    like "measured from a fork" vs "assumed dinner plate" vs "image framing only".
    It makes visible, in the response and in a demo, that a reference object (when
    present) is what actually does the work — the plate is only a fallback ruler.
    """
    label = (reference or "").replace("_", " ")
    if method == "reference_object":
        return f"measured from a {label} in the photo"
    if method == "container_prior":
        return f"assumed {label} (no reference object in frame)"
    return "image framing only (no reference object or known container)"


@dataclass(frozen=True)
class PortionEstimate:
    """The `/recognise` portion contract (`03 §4`, `05 §Output contract`).

    Always a range; `point_g` is the midpoint the slider defaults to. `method`
    records which scale stage governed the estimate; `reference` names the cue
    (a container or reference-object class) or is None for shape-only.
    """

    min_g: float
    max_g: float
    point_g: float
    method: PortionMethod
    reference: str | None


# ---- Optional aux-model interfaces (duck-typed; any may be absent) -----------
# The offline ML track (05 §7) delivers these as trained weights loaded at
# startup, mirroring the classifier. Until then the estimator runs without them.


class ContainerModel(Protocol):
    """Stage 1: classify the container and measure its rim area in pixels (05 §3)."""

    def classify(self, image: Image.Image) -> tuple[str, float]:
        """Return `(container_class, confidence)`."""

    def rim_area_px(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> float | None:
        """Return the container rim's enclosed area in pixels, or None.

        Area (not a single axis) so the scale derivation is shape-agnostic —
        square and rectangular containers work, not only round plates.
        `food_mask` (optional) lets the detector validate that a rim candidate
        actually contains the food — see `RimAreaDetector` rule 3.
        """


class ReferenceDetector(Protocol):
    """Stage 2: detect known-size reference objects and their pixel span (05 §4)."""

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        """Return `[(reference_class, pixel_length), ...]` for objects found."""


class Segmenter(Protocol):
    """Shared prerequisite: food-vs-background mask (05 §6)."""

    def segment(self, image: Image.Image) -> np.ndarray:
        """Return a boolean/0-1 mask the same H×W as the image."""


class PortionEstimator:
    """Orchestrates the 3-stage pipeline into one `estimate(...)` call (05 §8).

    All three aux models are optional. With none wired, `estimate` still returns
    a valid `shape_only` range from a classical segmentation + shape priors — the
    graceful-degradation contract of `05 §10`.
    """

    def __init__(
        self,
        container_model: ContainerModel | None = None,
        reference_detector: ReferenceDetector | None = None,
        segmenter: Segmenter | None = None,
    ) -> None:
        self.container_model = container_model
        self.reference_detector = reference_detector
        self.segmenter = segmenter

    # ---------------------------- public API ---------------------------------
    def estimate(
        self,
        image: Image.Image,
        food_class: str,
        mask: np.ndarray | None = None,
    ) -> PortionEstimate:
        """Estimate the portion mass range for one recognised food item.

        Deterministic and side-effect-free (`05 §8`): decodes nothing to disk,
        writes nothing, and returns the same result for the same inputs.
        """
        # Segment BEFORE resolving scale: the container rung validates its rim
        # candidate against the food mask (a container must contain its food),
        # which is what stops a food blob being mistaken for the plate when the
        # plate doesn't separate from the background (white plate, light table).
        region = mask if mask is not None else self._segment(image)
        cm_per_px, method, reference, band, tilt, rim_area_px = self._resolve_scale(
            image, region
        )
        area_cm2 = self._area(region, cm_per_px)

        # Tall-food tilt correction. The area-based scale is self-correcting for
        # camera tilt when the food is FLAT: plate and food foreshorten by the
        # same cos θ, and √(A_real/A_px) absorbs it. A tall mound breaks that —
        # its projected silhouette (side + top) barely shrinks with tilt, so its
        # area is over-recovered by ~1/cos θ. The rim ellipse's axis ratio IS
        # cos θ (a round plate at tilt θ projects an ellipse with exactly that
        # ratio), so multiplying the area back by it de-foreshortens tall
        # classes. Applied only to mound models (ELLIPSOID/DOME) under a
        # container scale; flat models need no correction and shape-only has no
        # tilt measurement. Measured on a weighed 203 g donut plate at ~60°:
        # uncorrected 337 g, corrected 167 g.
        prior = shape_priors.prior_for(food_class)
        if prior.model in (shape_priors.ShapeModel.ELLIPSOID, shape_priors.ShapeModel.DOME):
            area_cm2 *= tilt

        # §3: a bowl's fill depth, derived from the rim/surface area ratio rather
        # than assumed from a per-class constant. Requires a measured rim, so it
        # only ever applies under a container scale; off by default until there
        # are weighed bowls to validate it against (see `shape_priors`).
        h_override: float | None = None
        if (
            settings.portion_derive_bowl_depth
            and prior.model is shape_priors.ShapeModel.LIQUID_IN_BOWL
            and rim_area_px
        ):
            h_override = shape_priors.derived_fill_depth_cm(
                rim_area_px * (cm_per_px**2), area_cm2
            )

        volume_cm3 = self._volume(food_class, area_cm2, h_override=h_override)
        mass = volume_cm3 * density_of(food_class)
        mass = shape_priors.clamp_mass(food_class, mass)

        # `band` is a multiplicative (lo, hi) on mass whose width reflects the
        # governing scale stage — symmetric for reference/shape-only, and the
        # range-derived asymmetric band for a container prior (`05 §3/§4`).
        #
        # The band is taken around the *clamped* point and deliberately not
        # re-clamped itself. `clamp_mass` guards the point estimate against bad
        # geometry (a mask that leaked onto the plate); the band is a statement
        # about how uncertain that point is. Clamping both collapsed the interval
        # whenever the point pinned to a bound — `min_g == point_g` at the floor,
        # `max_g == point_g` at the ceiling — which reads as certainty exactly
        # where the pipeline is least sure, and breaks the `05 §1` contract that
        # an estimate is always a range. Found when a weighed 169 g fries plate
        # pinned to a clamp; the ceiling case was latent long before that, worked
        # around in the tests by hand-picking mask sizes that avoid the bounds.
        lo, hi = band
        return PortionEstimate(
            min_g=round(mass * lo, 1),
            max_g=round(mass * hi, 1),
            point_g=round(mass, 1),
            method=method,
            reference=reference,
        )

    # ---------------------------- Stage 1 + 2 --------------------------------
    def _resolve_scale(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, PortionMethod, str | None, tuple[float, float], float, float | None]:
        """Resolve cm-per-pixel from the best available cue (05 §4 priority).

        Order: reference object (tightest) → container prior → shape-only framing
        heuristic (widest). Returns `(cm_per_px, method, reference, band, tilt)`:
        `band` is the multiplicative `(lo, hi)` mass-uncertainty for that stage;
        `tilt` is the rim's projected axis ratio (≈cos of the camera tilt, 1.0
        when unmeasured) used by the tall-food correction in `estimate`;
        `rim_area_px` is the container's enclosed pixel area when Stage 1 measured
        one (None otherwise), which is what the §3 bowl-depth derivation needs.
        """
        # Stage 2 — reference object (strongest signal, tight symmetric band).
        ref_scale = self._reference_scale(image)
        if ref_scale is not None:
            cm_per_px, ref_class = ref_scale
            eps = settings.portion_eps_reference
            return (
                cm_per_px, "reference_object", ref_class, (1.0 - eps, 1.0 + eps), 1.0, None,
            )

        # Stage 1 — container size prior (band derived from its size range).
        container_scale = self._container_scale(image, food_mask)
        if container_scale is not None:
            cm_per_px, container_class, band, tilt, rim_area_px = container_scale
            return cm_per_px, "container_prior", container_class, band, tilt, rim_area_px

        # Fallback — device-normalised framing heuristic (05 §4 last resort).
        width_px = image.size[0]
        cm_per_px = settings.portion_shape_only_frame_width_cm / max(width_px, 1)
        eps = settings.portion_eps_shape_only
        return cm_per_px, "shape_only", None, (1.0 - eps, 1.0 + eps), 1.0, None

    def _reference_scale(self, image: Image.Image) -> tuple[float, str] | None:
        """Stage 2 scale from a detected reference object, or None (05 §4).

        Picks the highest-priority detected reference; if the next-priority one
        agrees within tolerance, averages the two scales for extra robustness.
        """
        if self.reference_detector is None:
            return None
        detections = self.reference_detector.detect(image)
        if not detections:
            return None

        # px length per detected class (keep the largest span if duplicated).
        px_by_class: dict[str, float] = {}
        for cls, px in detections:
            if px > 0 and (cls not in px_by_class or px > px_by_class[cls]):
                px_by_class[cls] = px

        best = reference_sizes.best_reference(list(px_by_class))
        if best is None:
            return None

        def scale_of(cls: str) -> float:
            size = reference_sizes.size_for(cls)
            assert size is not None  # best_reference only returns known classes
            return size.real_cm / px_by_class[cls]

        scale = scale_of(best)

        # Average with the next-best reference if it agrees within tolerance.
        others = [c for c in px_by_class if c != best]
        second = reference_sizes.best_reference(others)
        if second is not None:
            s2 = scale_of(second)
            if abs(s2 - scale) <= settings.portion_reference_agree_tol * scale:
                scale = (scale + s2) / 2.0
        return scale, best

    def _container_scale(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, str, tuple[float, float], float, float] | None:
        """Stage 1 scale + uncertainty band from the container prior, or None (05 §3).

        Scale is **area-based** — `cm_per_px = √(A_real / A_px)` — so it is
        shape-agnostic (a square plate uses `side²`, a round one `π r²`) rather
        than assuming a circular rim. The uncertainty band comes from the
        container's own min/max size range (squared, since mass ∝ size²), so a
        vaguely-sized container yields a correspondingly wider gram range. Returns
        None if there is no container model, no usable size, or no rim area.
        """
        if self.container_model is None:
            return None
        container_class, conf = self.container_model.classify(image)
        prior = container_priors.prior_for(container_class)
        if not container_priors.has_scale(prior):
            return None
        # Prefer the ellipse variant (area + tilt); duck-typed fallback keeps
        # older/stub container models (area-only) working with tilt = 1.0.
        rim_ellipse = getattr(self.container_model, "rim_ellipse", None)
        if rim_ellipse is not None:
            found = rim_ellipse(image, food_mask=food_mask)
            if found is None:
                return None
            area_px, tilt = found
        else:
            area_px = self.container_model.rim_area_px(image, food_mask=food_mask)
            tilt = 1.0
        if area_px is None or area_px <= 0:
            return None
        real_area = container_priors.real_area_cm2(prior, "typical")
        cm_per_px = math.sqrt(real_area / area_px)
        # The size range gives the band for a *correctly identified* container.
        # The classifier is also the thing most likely to be wrong (a bowl read as
        # a plate is 2.25x in mass), so its residual doubt widens the band on top
        # of that — free when it is confident, visibly wider when it is not.
        band = container_priors.widen_for_confidence(
            container_priors.uncertainty_band(prior), conf
        )
        return cm_per_px, container_class, band, max(0.3, min(float(tilt), 1.0)), area_px

    # ------------------------- segmentation (§6) -----------------------------
    def _segment(self, image: Image.Image) -> np.ndarray:
        """Return the food-region mask: trained segmenter, else classical, else box.

        Falls back through `05 §6`'s preference order and never raises — a failed
        segmentation yields a central-region mask so the pipeline still produces a
        (wider, honest) estimate rather than erroring the whole `/recognise` call.
        """
        if self.segmenter is not None:
            try:
                mask = np.asarray(self.segmenter.segment(image))
                if mask.any():
                    return mask.astype(bool)
            except Exception:
                pass  # fall through to classical
        classical = self._classical_segment(image)
        if classical is not None and classical.any():
            return classical
        return self._central_region(image)

    @staticmethod
    def _classical_segment(image: Image.Image) -> np.ndarray | None:
        """GrabCut on a central rectangle (OpenCV) — the no-training fallback (`05 §6`).

        Segments the central plated region from the surrounding table by seeding
        GrabCut with everything outside a 10% margin as definite background and the
        interior as probable foreground, then refining with GrabCut's colour model.
        This is a real improvement over "largest contour" (which grabbed whichever
        blob was biggest, usually the plate), but it is still classical: on a busy
        table or a small off-centre portion it may include the plate or miss food —
        documented limits (`05 §6`). The trained segmenter (`segmenter.py`) is the
        accurate path; this only runs when those weights are absent. Returns None on
        any failure so the caller drops to the central-region mask.
        """
        try:
            import cv2  # local import: keeps the module importable without cv2

            rgb = np.asarray(image.convert("RGB"))
            h, w = rgb.shape[:2]
            mx, my = int(w * 0.1), int(h * 0.1)
            rect = (mx, my, max(w - 2 * mx, 1), max(h - 2 * my, 1))
            gc_mask = np.zeros((h, w), np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            # Keep definite- and probable-foreground pixels.
            fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            return fg if fg.any() else None
        except Exception:
            return None

    @staticmethod
    def _central_region(image: Image.Image) -> np.ndarray:
        """Last-resort mask: the central 50%×50% box of the frame (`05 §6`)."""
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        y0, y1 = height // 4, 3 * height // 4
        x0, x1 = width // 4, 3 * width // 4
        mask[y0:y1, x0:x1] = True
        return mask

    # ---------------------------- Stage 3 (§5) -------------------------------
    @staticmethod
    def _area(region: np.ndarray, cm_per_px: float) -> float:
        """Food area in cm²: mask pixel count × (cm-per-pixel)²."""
        pixel_count = float(np.count_nonzero(region))
        return pixel_count * (cm_per_px**2)

    @staticmethod
    def _volume(food_class: str, area_cm2: float, h_override: float | None = None) -> float:
        """Volume in cm³ from the class's shape prior (`05 §5`).

        `h_override` replaces the class's height prior with a *measured* one — used
        only by the §3 bowl-depth derivation, where the fill depth is computed from
        the container rim rather than assumed.
        """
        prior = shape_priors.prior_for(food_class)
        h = prior.h if h_override is None else h_override
        return shape_priors.volume_cm3(prior.model, area_cm2, h, prior.k)
