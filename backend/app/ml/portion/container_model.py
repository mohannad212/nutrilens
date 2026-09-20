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
    """Classical-CV enclosed pixel area of the container rim (`05 §3`).

    Three rules, each pinned to an observed failure on real weighed photos
    (2026-08 audit — the original "largest blob under a 95% cap" version returned
    the *countertop* on 3/3 real photos, and would happily have returned the food
    itself on a white plate):

    1. **Border filter** — a background region always runs off the frame edge; a
       plate the photographer framed does not. Candidates whose bounding box
       touches more than `max_border_touches` frame edges are rejected. This is
       what actually separates plate from countertop; the old area cap did not
       (a granite background at 89.7% of frame sailed under a 95% cap).
    2. **Convex-hull area** — the food carves bites out of the plate's colour
       region, so the raw contour under-measures the rim's *enclosed* area
       (a dark plate under two bright donuts thresholds as a crescent). The
       hull restores the full rim footprint.
    3. **A container must contain its food** — when the plate and background do
       not separate (white plate on a light counter), the largest clean blob is
       the *food*. Given the food mask, a rim candidate must cover a meaningful
       share of it (`min_food_containment`) and be meaningfully bigger than it
       (`min_food_area_ratio`); a "plate" smaller than the food on it is not a
       plate. The containment floor is deliberately loose (0.4, not ~0.9):
       tall food photographed at an angle projects *above* the far rim — on a
       real weighed donut photo only 46% of the donut mask fell inside the
       plate ellipse. Rejecting disjoint candidates (a napkin: ~0.0) is the
       rule's whole job; the size ratio is what rejects the food itself.
       Without a plausible candidate we return None — the estimator's wide
       shape-only band is more honest than a confidently wrong scale.

    `detect_ellipse` additionally reports the rim's projected axis ratio
    (minor/major of a fitted ellipse). A round plate photographed at tilt θ
    projects an ellipse with ratio ≈ cos θ, so this is a free tilt measurement
    the estimator uses to de-foreshorten *tall* foods (see `__init__.py`).
    """

    def __init__(
        self,
        min_area_frac: float = 0.05,
        max_area_frac: float = 0.80,
        max_border_touches: int = 1,
        min_food_containment: float = 0.4,
        min_food_area_ratio: float = 1.15,
    ):
        # Ignore blobs smaller than this fraction of the frame (noise) or larger
        # than this (the background itself, not the container).
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        # A framed plate may nudge one edge; background hugs 3-4 of them.
        self.max_border_touches = max_border_touches
        # Fraction of the food mask a rim candidate must cover, and how much
        # larger than the food it must be, to count as the container under it.
        self.min_food_containment = min_food_containment
        self.min_food_area_ratio = min_food_area_ratio

    def detect(self, image: Image.Image, food_mask: np.ndarray | None = None) -> float | None:
        """Return the container rim's enclosed pixel area, or None.

        `food_mask` (same H×W as the image, truthy on food pixels) enables rule 3
        above. Never raises — on any failure (or no plausible region) returns None
        so the estimator falls through to shape-only rather than erroring
        `/recognise`.
        """
        found = self.detect_ellipse(image, food_mask=food_mask)
        return found[0] if found is not None else None

    def detect_ellipse(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """Return `(hull_area_px, axis_ratio)` for the best rim candidate, or None.

        `axis_ratio` is minor/major of the ellipse fitted to the winning hull —
        ≈1.0 for a top-down shot, ≈cos(tilt) for an oblique one. Same never-raises
        contract as `detect`.
        """
        try:
            import cv2  # local import: keeps the module importable without cv2

            gray = np.asarray(image.convert("L"))
            h, w = gray.shape
            frame_area = float(h * w)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_px = float(np.count_nonzero(food_mask)) if food_mask is not None else None

            # Try both polarities: the container may be brighter OR darker than
            # the surface it sits on.
            best: tuple[float, float] | None = None
            for b in (binary, 255 - binary):
                contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    hull = cv2.convexHull(c)
                    hull_area = float(cv2.contourArea(hull))
                    if not (
                        self.min_area_frac * frame_area
                        <= hull_area
                        <= self.max_area_frac * frame_area
                    ):
                        continue
                    x, y, bw, bh = cv2.boundingRect(c)
                    touches = (
                        int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
                    )
                    if touches > self.max_border_touches:
                        continue
                    if mask_px is not None and mask_px > 0:
                        if hull_area < self.min_food_area_ratio * mask_px:
                            continue
                        filled = np.zeros((h, w), np.uint8)
                        cv2.fillConvexPoly(filled, hull, 1)
                        covered = float(np.count_nonzero(filled.astype(bool) & food_mask))
                        if covered < self.min_food_containment * mask_px:
                            continue
                    if best is None or hull_area > best[0]:
                        best = (hull_area, self._axis_ratio(cv2, hull))
            return best
        except Exception:
            return None

    @staticmethod
    def _axis_ratio(cv2: Any, hull: np.ndarray) -> float:
        """Minor/major axis ratio of the ellipse fitted to the hull, in (0, 1]."""
        try:
            if len(hull) >= 5:  # fitEllipse needs ≥5 points
                (_cx, _cy), (a, b), _angle = cv2.fitEllipse(hull)
                if a > 0 and b > 0:
                    return float(min(a, b) / max(a, b))
        except Exception:
            pass
        return 1.0  # unknown tilt -> assume top-down (no correction)


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

    def rim_area_px(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> float | None:
        return self.rim_detector.detect(image, food_mask=food_mask)

    def rim_ellipse(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """`(hull_area_px, axis_ratio)` — area plus the free tilt measurement."""
        return self.rim_detector.detect_ellipse(image, food_mask=food_mask)


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
def load_container_model(
    weights_path: str, labels_path: str, device: str, surface_growth: bool = True
) -> TrainedContainerModel:
    """Load the TorchScript container classifier + build the full `ContainerModel`.

    Cached so the lifespan's startup load isn't repeated. Raises whatever
    `torch.jit.load` raises if the weights are missing/corrupt — the caller
    (lifespan) decides how to degrade.

    `surface_growth` adds the backlog-§4 fallback rim detector *behind* the Otsu
    one, so Otsu's answers are unchanged and only the photos it declined get a
    second opinion.
    """
    import torch

    dev = torch.device(device)
    module = torch.jit.load(weights_path, map_location=dev).eval()
    idx_to_class: dict[str, str] = json.loads(Path(labels_path).read_text())
    classifier = ContainerClassifier(module, idx_to_class, _build_container_transform(), dev)
    rim: Any = RimAreaDetector()
    if surface_growth:
        rim = ChainedRimDetector([rim, SurfaceGrowthRimDetector()])
    return TrainedContainerModel(classifier, rim_detector=rim)


class SurfaceGrowthRimDetector:
    """Rim area grown outward from the food mask (backlog §4), not thresholded.

    WHY A SECOND DETECTOR.
    `RimAreaDetector` is Otsu + contours, and it finds a rim on only 3 of 6
    weighed photos. Its failure mode is specific and understood: it needs the
    plate to separate from the background *globally*. A black plate on granite
    works; a **white plate on speckled granite fails**, because the bright grains
    bridge plate and counter into one blob that the border rule then (correctly)
    rejects. The audit established that no amount of further Otsu tuning fixes
    this, and that morphological opening appears to fix it while actually
    outlining the plate *plus the kitchen scale*. So this is not another
    threshold — it is a different question.

    THE DIFFERENT QUESTION.
    Instead of "which pixels are plate?", ask "what surface is the food sitting
    on?". The food mask (from the trained segmenter, IoU 0.925) is a reliable
    seed we already compute. The pixels in a thin ring *just outside* the food
    are, by construction, the container: food rests on its container. So we can
    sample the container's actual colour from that ring and grow the region that
    matches it, rather than trying to guess a global threshold that separates two
    surfaces we have never measured. Nothing here is decided by image-wide
    statistics, which is exactly what made Otsu brittle.

    WHY IT IS ADDITIVE, NOT A REPLACEMENT.
    This runs only when `RimAreaDetector` returns None. Otsu is genuinely correct
    where it fires (it also supplies the tilt ratio that de-foreshortens tall
    food), and the backlog's rule stands: rejecting a doubtful rim is correct
    behaviour, so a new detector must only ever *add* candidates that pass the
    same safety rules — never rescue one the old rules threw out. Every rule
    below is inherited from `RimAreaDetector` for that reason.

    HOW THE COLOUR TOLERANCE IS CHOSEN — AND WHY IT IS NOT A CONSTANT.
    The obvious design is one tolerance: grow every pixel within distance `t` of
    the ring's colour. Measured over the weighed set, that is a knife-edge. The
    grown area as a function of `t` moves 27% -> 40% -> 42% -> nothing on the
    spaghetti photo, and every value of `t` that rescued one photo lost another;
    there is no broad stable plateau. Picking the `t` that happened to make six
    photos resolve would be fitting a constant to the test set — precisely the
    "tuning a prior to a single photo" mistake the audit kept catching.

    So the tolerance is not chosen; it is *found per photo* by a criterion that
    needs no constant: **grow until the region spills, then take the last size
    before it spilled.** A container is a bounded object the photographer framed,
    so the correct region is the largest one that still satisfies every safety
    rule — above all, that it does not run off the frame edge. The ladder is swept
    coarse-to-fine and the largest passing candidate wins. A photo where the
    surface never separates from the counter simply never produces a passing
    candidate at any rung, and we return None, which is the honest answer.

    HONEST LIMIT.
    A ring sampled around food that overflows its container (a burger filling its
    plate, a sandwich on a board) samples the *table*, and would grow the table.
    The border-touch and area-fraction rules are what stop that becoming a ruler,
    and they are the same rules that stopped the countertop being used before.
    """

    def __init__(
        self,
        ring_px: int = 12,
        ring_gap_px: int = 4,
        tol_ladder: tuple[float, ...] = (8.0, 11.0, 14.0, 18.0, 22.0, 26.0, 30.0),
        min_area_frac: float = 0.05,
        max_area_frac: float = 0.80,
        max_border_touches: int = 1,
        min_food_area_ratio: float = 1.15,
    ):
        # Ring geometry: skip `ring_gap_px` outward first (segmentation edges are
        # soft — sampling right at the boundary picks up food/plate blend pixels),
        # then sample a `ring_px`-wide annulus of true container surface.
        self.ring_px = ring_px
        self.ring_gap_px = ring_gap_px
        # Colour tolerances to try, tight to loose. Not a tuned constant — the
        # detector keeps the LARGEST region that still passes every safety rule,
        # so the ladder only has to span the plausible range. See the class
        # docstring for why a single fixed tolerance was rejected.
        self.tol_ladder = tol_ladder
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_border_touches = max_border_touches
        self.min_food_area_ratio = min_food_area_ratio

    def detect(self, image: Image.Image, food_mask: np.ndarray | None = None) -> float | None:
        found = self.detect_ellipse(image, food_mask=food_mask)
        return found[0] if found is not None else None

    def detect_ellipse(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        """`(hull_area_px, axis_ratio)` for the grown container region, or None.

        Requires `food_mask` — without a seed there is nothing to grow from, and
        this detector has no opinion at all (returns None, estimator falls to
        shape-only). Never raises.
        """
        if food_mask is None:
            return None
        try:
            import cv2

            food = np.asarray(food_mask).astype(bool)
            mask_px = float(np.count_nonzero(food))
            if mask_px <= 0:
                return None

            rgb = np.asarray(image.convert("RGB"))
            h, w = rgb.shape[:2]
            frame_area = float(h * w)
            if food.shape != (h, w):
                return None

            # CIELAB, not RGB: L*a*b* is approximately perceptually uniform, so a
            # single distance threshold means the same thing on a white plate and
            # a dark one. In RGB the same numeric tolerance is far too loose in
            # bright regions and far too tight in shadow.
            lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB)

            food_u8 = food.astype(np.uint8)
            gap = cv2.dilate(food_u8, self._kernel(cv2, self.ring_gap_px)).astype(bool)
            outer = cv2.dilate(
                food_u8, self._kernel(cv2, self.ring_gap_px + self.ring_px)
            ).astype(bool)
            ring = outer & ~gap
            if np.count_nonzero(ring) < 50:  # too little surface to characterise
                return None

            samples = lab[ring].astype(np.float32)
            # Median rather than mean: a ring that clips a fork, a shadow or a
            # spill must not drag the reference colour with it.
            centre = np.median(samples, axis=0)
            dist = np.linalg.norm(lab.astype(np.float32) - centre, axis=2)

            best: tuple[float, float] | None = None
            for tol in self.tol_ladder:
                found = self._grow_at(cv2, dist, food, tol, h, w, frame_area, mask_px)
                # Largest passing region wins: growth stops being valid the moment
                # it spills off the frame, so the last size before that is the
                # container.
                if found is not None and (best is None or found[0] > best[0]):
                    best = found
            return best
        except Exception:
            return None

    def _grow_at(
        self, cv2: Any, dist: np.ndarray, food: np.ndarray, tol: float,
        h: int, w: int, frame_area: float, mask_px: float,
    ) -> tuple[float, float] | None:
        """One rung of the ladder: grow at colour tolerance `tol`, or None."""
        try:
            import numpy as np

            similar = ((dist <= tol) | food).astype(np.uint8)
            # Close small gaps (specular highlights, the food's own shadow on the
            # plate) so the container is one component rather than a ring of
            # fragments. This is NOT the morphological opening the audit rejected:
            # that one was applied to a global threshold and silently merged the
            # plate with the kitchen scale. Here the component must still contain
            # the food seed and still pass the border rule.
            similar = cv2.morphologyEx(similar, cv2.MORPH_CLOSE, self._kernel(cv2, 5))

            # Keep only the connected component the food actually sits in — this
            # is the "same connected surface" the region must belong to, and it
            # is what stops a similarly-coloured object elsewhere in frame (a
            # napkin, a second plate) joining the measurement.
            n, labels = cv2.connectedComponents(similar)
            if n < 2:
                return None
            seed_labels, counts = np.unique(labels[food], return_counts=True)
            seed_labels, counts = seed_labels[seed_labels > 0], counts[seed_labels > 0]
            if seed_labels.size == 0:
                return None
            component = (labels == int(seed_labels[int(np.argmax(counts))])).astype(np.uint8)

            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            best: tuple[float, float] | None = None
            for c in contours:
                hull = cv2.convexHull(c)
                hull_area = float(cv2.contourArea(hull))
                # --- the SAME safety rules as RimAreaDetector, deliberately ---
                if not (
                    self.min_area_frac * frame_area
                    <= hull_area
                    <= self.max_area_frac * frame_area
                ):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                touches = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
                if touches > self.max_border_touches:
                    continue
                # A "plate" no bigger than the food on it is not a plate.
                if hull_area < self.min_food_area_ratio * mask_px:
                    continue
                if best is None or hull_area > best[0]:
                    best = (hull_area, RimAreaDetector._axis_ratio(cv2, hull))
            return best
        except Exception:
            return None

    @staticmethod
    def _kernel(cv2: Any, r: int) -> Any:
        k = max(int(r), 1) * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


class ChainedRimDetector:
    """Try each rim detector in order; first plausible answer wins (backlog §4).

    Order matters and is not arbitrary: `RimAreaDetector` (Otsu) runs first
    because where it fires it is right, and it measures the tilt ratio the tall-
    food correction depends on. `SurfaceGrowthRimDetector` only sees the photos
    Otsu declined. This makes the addition strictly monotonic — no photo that
    already had a rim can get a different one — so a regression in the weighed
    MAPE can only come from a newly-rescued row, which is easy to inspect.
    """

    def __init__(self, detectors: list) -> None:
        self.detectors = detectors

    def detect(self, image: Image.Image, food_mask: np.ndarray | None = None) -> float | None:
        found = self.detect_ellipse(image, food_mask=food_mask)
        return found[0] if found is not None else None

    def detect_ellipse(
        self, image: Image.Image, food_mask: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        for d in self.detectors:
            try:
                found = d.detect_ellipse(image, food_mask=food_mask)
            except Exception:
                continue
            if found is not None:
                return found
        return None
