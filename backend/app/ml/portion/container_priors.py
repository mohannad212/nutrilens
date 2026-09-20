"""
app/ml/portion/container_priors.py
=====================================

WHY THIS FILE EXISTS
---------------------
Stage 1 of the portion pipeline (`05 §3`) classifies the *container* a food sits
on/in (plate, bowl, cup...) to get a first, coarse guess at physical scale. A
container class on its own means nothing to the maths — it only helps because we
know roughly how big a "dinner plate" is in the real world. This file is that
knowledge: the lookup from a container class to its typical real-world
diameter/width in centimetres, with a plausible range.

WHAT PROBLEM IT SOLVES
------------------------
A single uncalibrated photo has no absolute scale (`05 §1`). If we can find the
container's rim in the image and measure its *pixel area*, then a shape-aware real
footprint area (disc for a round plate, square for a box) gives a first
`cm_per_px = √(A_real / A_px)` estimate — one that works for square/rectangular
containers, not just circles. This table supplies the "known real size" and shape
per container class, and — because plates genuinely vary — a min/max range so the
uncertainty propagates honestly into the final gram range.

HOW IT WORKS INTERNALLY
--------------------------
`CONTAINER_PRIORS` maps each of the 8 container classes from `05 §3` to a
`ContainerPrior(typical_cm, min_cm, max_cm, shape)`. `real_area_cm2()` turns a
size + shape into a footprint area (for area-based scaling); `uncertainty_band()`
turns the min/max range into an *asymmetric* mass band (mass ∝ size², so the range
is squared). `unknown` carries no usable size — it signals "Stage 1 could not
help, fall through to Stage 2 with a wider band" (`05 §3` last row). `prior_for()`
never raises on an unseen label — it returns the `unknown` sentinel so the
estimator degrades gracefully rather than crashing on a new container string.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- A plain data table (chosen) over hard-coded numbers inside the estimator:
  these are *measured facts* about tableware, not algorithm logic; keeping them
  as reviewable data (with a source comment) means a defender can point at one
  file and say "here are our physical assumptions", and they can be tuned without
  touching the pipeline code.
- Storing a range, not just a mean: the whole honesty principle of `05` is that
  every stage widens or narrows a band. A dinner plate is 24–30 cm, not exactly
  27; encoding that lets the final estimate own its real uncertainty.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Where do these numbers come from?"
A: Standard tableware dimensions (dinner plates ~26–28 cm, side plates ~19–21 cm,
   mugs ~8 cm rim, etc.). They are priors, deliberately given as ranges because
   real crockery varies; the reference-object stage (`05 §4`) overrides them with
   a precise scale whenever a known object is in frame.

Q: "What if the container is misclassified?"
A: The error stays bounded because we use a *range* and, when a Stage-2 reference
   object is present, that stronger signal dominates (`05 §4`). Container prior is
   the fallback, not the final word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Container footprint shape, used to convert a linear size (cm) into a real area
# (cm²) — a round plate is a disc, a takeaway box / wrap is treated as a square.
# `05 §3` only tabulated a diameter/width; the shape is what lets the estimator
# scale from area (shape-agnostic) instead of assuming everything is a circle.
SHAPE_ROUND = "round"
SHAPE_RECT = "rect"


@dataclass(frozen=True)
class ContainerPrior:
    """Typical + plausible-range real-world size (cm) and footprint shape.

    `typical_cm`/`min_cm`/`max_cm` are the diameter (round) or side (rect); the
    min/max span is what the estimator propagates (squared) into the portion
    uncertainty band, so a vaguely-sized container honestly yields a wider range.
    """

    typical_cm: float
    min_cm: float
    max_cm: float
    shape: str = SHAPE_ROUND


# The 8 container classes from `05 §3`. Values are the "typical diameter/width"
# and "range used" columns of that table. `unknown` has no size — it means Stage
# 1 could not contribute a scale and the estimator must rely on Stage 2 (or fall
# to shape-only) with a wider uncertainty band.
CONTAINER_PRIORS: dict[str, ContainerPrior] = {
    "dinner_plate": ContainerPrior(27.0, 24.0, 30.0, SHAPE_ROUND),
    "side_plate": ContainerPrior(20.0, 18.0, 22.0, SHAPE_ROUND),
    "bowl_small": ContainerPrior(13.0, 11.0, 15.0, SHAPE_ROUND),
    "bowl_large": ContainerPrior(18.0, 16.0, 21.0, SHAPE_ROUND),
    "cup_mug": ContainerPrior(8.0, 7.0, 9.0, SHAPE_ROUND),  # rim diameter
    "wrap_flat": ContainerPrior(22.0, 18.0, 26.0, SHAPE_RECT),  # elongated / flat
    "takeaway_box": ContainerPrior(15.0, 12.0, 18.0, SHAPE_RECT),  # square-ish box
    "unknown": ContainerPrior(0.0, 0.0, 0.0, SHAPE_ROUND),  # sentinel: no scale
}

# Returned for any container label we have never seen — same meaning as
# `unknown`: Stage 1 contributed nothing.
_UNKNOWN = CONTAINER_PRIORS["unknown"]


def prior_for(container_class: str) -> ContainerPrior:
    """Return the size prior for a container class, or the `unknown` sentinel.

    Never raises on an unrecognised label so the estimator can degrade
    gracefully (`05 §10`) instead of failing the whole `/recognise` call.
    """
    return CONTAINER_PRIORS.get(container_class, _UNKNOWN)


def has_scale(prior: ContainerPrior) -> bool:
    """True if this prior carries a usable size (i.e. is not the unknown sentinel)."""
    return prior.typical_cm > 0.0


def _footprint_area(dim_cm: float, shape: str) -> float:
    """Real footprint area (cm²) of a container of linear size `dim_cm`.

    Round → disc `π (d/2)²`; rect → square `s²`. This is what makes the estimator
    shape-aware: a square plate's area is `side²`, not a circle's `π r²`, so we no
    longer implicitly assume every container is round.
    """
    if shape == SHAPE_RECT:
        return dim_cm * dim_cm
    return math.pi * (dim_cm / 2.0) ** 2


def real_area_cm2(prior: ContainerPrior, which: str = "typical") -> float:
    """Footprint area for the `typical` / `min` / `max` size of a container.

    Used both to derive an area-based scale (`cm_per_px = √(A_real / A_px)`, which
    works for any shape) and — via the min/max variants — to size the portion
    uncertainty band.
    """
    dim = {"typical": prior.typical_cm, "min": prior.min_cm, "max": prior.max_cm}[which]
    return _footprint_area(dim, prior.shape)


def uncertainty_band(prior: ContainerPrior) -> tuple[float, float]:
    """Multiplicative mass band `(lo, hi)` implied by the size range (`05 §3/§4`).

    Mass ∝ area ∝ (linear size)², so a diameter/side that could be anywhere in
    `[min_cm, max_cm]` (point estimate at `typical_cm`) yields an *asymmetric*
    mass band `((min/typ)², (max/typ)²)`. A tightly-known container gives a narrow
    band; a vague one (e.g. a collapsed "plate" class) correctly gives a wide one,
    so the small-vs-large ambiguity shows up honestly in the gram range instead of
    being hidden behind a flat constant.
    """
    typ = prior.typical_cm
    if typ <= 0.0:  # unknown sentinel — no meaningful band
        return 1.0, 1.0
    lo = (prior.min_cm / typ) ** 2
    hi = (prior.max_cm / typ) ** 2
    return lo, hi


# The largest single mistake Stage 1 can make is a bowl/plate mixup: applying the
# 27 cm `dinner_plate` prior to an 18 cm `bowl_large` scales mass by (27/18)² =
# **2.25x**. This is not hypothetical — the 2026-08 audit caught the soup bowl
# classified `dinner_plate` at 0.809 confidence (`13-portion-accuracy-backlog.md`
# §5). Until the classifier is retrained on a size taxonomy (§5c), the honest move
# is to keep its argmax but let its own doubt widen the band.
CONTAINER_CONFUSION_RATIO = 2.25


def confidence_widen(confidence: float) -> float:
    """Multiplicative band widening implied by the classifier's own confidence.

    `1 + (1 - p)·(RATIO - 1)`: at p=1.0 the band is untouched, and as p falls the
    band stretches toward the full bowl/plate confusion ratio. At p=0.99 that is
    ×1.01 (effectively free when the model is sure); at p=0.81 — the confidence on
    the misclassified soup bowl — it is ×1.24.

    WHY THIS SHAPE. The alternative is a mixture band spanning bowl-min to
    plate-max, which is **3.5x wide** for *every* photo regardless of confidence —
    too wide to show a user, and it throws away the information that the model is
    usually right. Committing to argmax and widening by the residual doubt keeps
    the common (confident) case tight while making the uncertain case visibly
    uncertain. It is a statement about a real, named source of error, not a
    coverage-buying widen (see the backlog's COMMON MISTAKES).
    """
    p = max(0.0, min(float(confidence), 1.0))
    return 1.0 + (1.0 - p) * (CONTAINER_CONFUSION_RATIO - 1.0)


def widen_for_confidence(
    band: tuple[float, float], confidence: float
) -> tuple[float, float]:
    """Stretch a container band outward by `confidence_widen(confidence)`.

    Applied to the *mass* band symmetrically in log-space (lo ÷ w, hi × w) — the
    size prior's own min/max asymmetry is preserved, and the classifier's doubt is
    layered on top of it rather than replacing it.
    """
    lo, hi = band
    w = confidence_widen(confidence)
    return lo / w, hi * w
