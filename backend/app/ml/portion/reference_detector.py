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


# =============================================================================
# Credit-card reference detector (backlog §1) — classical geometry, no training
# =============================================================================
# WHY A SECOND DETECTOR RATHER THAN A CLASS ON THE FIRST.
# A fork needs ML because it is a *shape* — there is no closed-form description of
# "fork". A card is the opposite: a rigid rectangle whose aspect ratio is fixed by
# standard. COCO has no card class and training one would need a dataset, when
# `approxPolyDP` + `minAreaRect` already describe the object exactly.
#
# WHY IT IS WORTH THE TROUBLE. It is the only *exact* reference available:
#   card  85.60 mm, ISO/IEC 7810 ID-1, zero variance, every bank card on earth
#   fork  19 cm ASSUMED; real forks run 17-21 cm -> ~10% length variance, and
#         since mass scales with area that is ~20% mass error.
# That is why `reference_sizes.py` ranks `credit_card` priority 1 and `fork` 4.
#
# THE WHOLE ENGINEERING PROBLEM IS FALSE POSITIVES. A wrongly-detected card is a
# confidently wrong ruler, which is strictly worse than no ruler at all (the
# pipeline degrades honestly to a wider band when Stage 2 finds nothing). The two
# thresholds below are what buy that safety, and both were measured, not guessed:
#   * ratio tolerance 6% (not 12%). At 12% the kitchen scale's display panel
#     (ratio 1.42) was detected as a card in the donut photo.
#   * min long edge 80 px. Below that a card is too few pixels to be a useful
#     ruler anyway, and small ragged quadrilaterals are where the false positives
#     live.
# At 6% + 80 px the prototype found ZERO false positives across the weighed set.
#
# KNOWN LIMIT — READ BEFORE TRUSTING THIS. The tolerance was validated against a
# *pasted flat rectangle*, not a photographed card. A real card has rounded
# corners, glare, shadow and perspective skew, all of which push the apparent
# ratio off 1.5857; a tilted card's projected ratio is off by 1/cos(theta) in one
# axis. So this detector is expected to MISS tilted real cards rather than
# hallucinate them — which is the correct direction to fail, but it means the hit
# rate is unproven on real photos. Do not loosen `ratio_tol` to raise that hit
# rate without re-running the false-positive check over the whole weighed set:
# loosening is exactly what brought the kitchen scale back.

CARD_LONG_MM, CARD_SHORT_MM = 85.60, 53.98
CARD_RATIO = CARD_LONG_MM / CARD_SHORT_MM  # 1.5857...


class CardReferenceDetector:
    """`ReferenceDetector` protocol impl: finds an ID-1 card by geometry alone.

    Emits `[("credit_card", long_edge_px)]` — `reference_sizes.py` maps that key to
    8.56 cm. Returns `[]` on anything it is not sure about, and never raises: a
    missing card must degrade to the next scale rung, never fail `/recognise`.
    """

    def __init__(
        self,
        ratio_tol: float = 0.06,
        min_long_px: float = 80.0,
        min_frac: float = 0.0008,
        max_frac: float = 0.08,
        solidity_min: float = 0.90,
    ):
        self.ratio_tol = ratio_tol
        self.min_long_px = min_long_px
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.solidity_min = solidity_min

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        cards = self.find_cards(image)
        if not cards:
            return []
        # If several survive, the largest is the best ruler (most pixels across
        # the known 8.56 cm = the smallest relative measurement error).
        return [("credit_card", cards[0]["long_px"])]

    def find_cards(self, image: Image.Image) -> list[dict]:
        """Candidate cards, largest first. Each dict carries `long_px` and `ratio`.

        Separate from `detect` so the false-positive audit script can inspect
        *why* something was accepted, not just that it was.
        """
        try:
            import cv2
            import numpy as np

            bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return []

        try:
            h, w = bgr.shape[:2]
            frame = float(h * w)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Bilateral (not Gaussian): smooths surface texture — granite speckle,
            # tablecloth weave — while keeping the card's straight edges sharp,
            # which is the one feature the whole detector depends on.
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            found: list[dict] = []
            # Three Canny settings: a card may be lighter or darker than the
            # surface it sits on, and one threshold pair misses the low-contrast
            # case (a white card on a white counter).
            for lo, hi in ((30, 90), (50, 150), (75, 200)):
                edges = cv2.Canny(gray, lo, hi)
                # Close 1-2 px gaps so a card outline broken by glare still forms
                # a single closed contour.
                edges = cv2.morphologyEx(
                    edges,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                )
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    cand = self._as_card(c, frame, cv2, np)
                    if cand is not None:
                        found.append(cand)

            # The three edge settings usually find the same card three times.
            deduped: list[dict] = []
            for d in sorted(found, key=lambda d: -d["area_px"]):
                if all(
                    abs(d["long_px"] - k["long_px"]) / k["long_px"] > 0.15
                    for k in deduped
                ):
                    deduped.append(d)
            return deduped
        except Exception:
            return []

    def _as_card(self, contour, frame_area: float, cv2, np) -> dict | None:
        """Accept one contour as a card, or None. Every rejection is a named rule."""
        area = cv2.contourArea(contour)
        # Rule 1: plausible size in frame. Too small is noise; too large is the
        # table, the plate, or the frame border itself.
        if not (self.min_frac * frame_area <= area <= self.max_frac * frame_area):
            return None
        # Rule 2: it must actually be a quadrilateral, and convex — a card is.
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None
        # Rule 3: solidity — guards against ragged edge fragments that happen to
        # simplify to four vertices but enclose a much smaller real area.
        hull_area = cv2.contourArea(cv2.convexHull(approx))
        if hull_area <= 0 or area / hull_area < self.solidity_min:
            return None
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(approx)
        long_px, short_px = max(rw, rh), min(rw, rh)
        # Rule 4: big enough to be a useful ruler. This is the threshold that
        # killed the remaining false positives (see the header note).
        if long_px < self.min_long_px or short_px <= 0:
            return None
        # Rule 5: the ISO aspect ratio, within a deliberately tight tolerance.
        ratio = long_px / short_px
        if abs(ratio - CARD_RATIO) / CARD_RATIO > self.ratio_tol:
            return None
        return {
            "long_px": float(long_px),
            "short_px": float(short_px),
            "ratio": float(ratio),
            "area_px": float(area),
            "center": (float(cx), float(cy)),
            "angle": float(ang),
        }


class CompositeReferenceDetector:
    """Runs several reference detectors and concatenates their detections.

    The estimator already ranks whatever it is handed by `reference_sizes`
    priority (card 1 > cutlery 4) and averages a second reference in when the two
    agree, so this needs no ranking logic of its own — it only has to make both
    detectors' output reach `_reference_scale` in one list. One failing detector
    must not silence the others, hence the per-detector guard.
    """

    def __init__(self, detectors: list) -> None:
        self.detectors = detectors

    def detect(self, image: Image.Image) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for d in self.detectors:
            try:
                out.extend(d.detect(image))
            except Exception:
                continue
        return out
