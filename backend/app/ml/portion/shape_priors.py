"""
app/ml/portion/shape_priors.py
=================================

WHY THIS FILE EXISTS
---------------------
Stage 3 of the portion pipeline (`05 §5`) turns a food's measured pixel footprint
(an area A in cm², once a scale is known) into a *volume* in cm³. You cannot do
that from area alone — a 100 cm² blob of soup, a scoop of rice, and a pizza slice
have very different volumes for the same footprint. This file encodes the missing
third dimension per food: which geometric *shape model* the class resembles, its
height prior, and a sane min/max clamp on the resulting mass.

WHAT PROBLEM IT SOLVES
------------------------
Volume = shape_model(area, height_prior). A pizza is a thin `flat_disc`
(V ≈ A × small h); a scoop of mashed potato is a `dome` (V from a hemisphere);
fries are an `irregular_pile` that heaps higher as it spreads (h grows with √A).
Picking the right model per class — and clamping the result to a physically
plausible band — is what keeps the gram estimate sensible instead of off by an
order of magnitude when segmentation over/under-shoots.

HOW IT WORKS INTERNALLY
--------------------------
- `ShapeModel` enumerates the six models from `05 §5`.
- `volume_cm3(model, area_cm2, h, k)` implements each model's formula
  (disc/dome/ellipsoid/cylinder/pile/liquid). Parameters come from the mask area
  and the class's height prior `h` (and heap factor `k` for piles).
- `SHAPE_PRIORS` maps each tuned Food-101 class to a `ShapePrior(model, h, k,
  clamp_min_g, clamp_max_g)`. Unlisted classes get `DEFAULT_SHAPE` (an
  `irregular_pile` with `h=1.2`, `k=0.35`, generous clamp) so any of the 101
  classes yields *some* estimate (`05 §5` "sensible default").
- `prior_for()` and `clamp_mass()` are the non-raising accessors the estimator
  calls.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- A small set of parameterised geometric primitives (chosen) over a learned
  volume regressor: with no depth sensor and a modest dataset, closed-form shape
  priors are interpretable, need no extra training data, and fail gracefully. The
  report can state exactly why a class was modelled as a dome vs a disc.
- Per-class mass clamps rather than trusting the raw formula: a segmentation that
  leaks into the plate can double the area and thus the volume; clamping to a
  class-plausible band (a single fried egg ∈ [30, 90] g) rejects those blow-ups
  cheaply (`05 §5`).

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Isn't a fixed height prior crude?"
A: Yes — it is a *prior*, deliberately coarse, and it feeds a *range* the user
   confirms, not a final number. The uncertainty band around it (chosen by which
   scale stage ran) is where that crudeness is accounted for honestly (`05 §9`).

Q: "Why does an irregular_pile's height grow with √area?"
A: A heaped serving of fries or salad doesn't stay a constant thickness as it
   grows — more food both spreads and mounds. `h_eff = k·√A` captures that the
   pile gets taller as its footprint grows, which a flat `A × h` would miss.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class ShapeModel(str, enum.Enum):
    """The six geometric volume models from `05 §5`."""

    FLAT_DISC = "flat_disc"
    DOME = "dome"
    ELLIPSOID = "ellipsoid"
    CYLINDER = "cylinder"
    IRREGULAR_PILE = "irregular_pile"
    LIQUID_IN_BOWL = "liquid_in_bowl"


def _radius_from_area(area_cm2: float) -> float:
    """Radius of the circle with the given area — A = π r²  →  r = √(A/π)."""
    return math.sqrt(max(area_cm2, 0.0) / math.pi)


def volume_cm3(model: ShapeModel, area_cm2: float, h: float, k: float) -> float:
    """Volume (cm³) for a shape model given footprint area, height prior h, heap k.

    Formulas mirror the `05 §5` table. Parameters are derived from the mask area
    (a circle of equivalent area where a radius is needed) so a single measured A
    drives every model consistently.
    """
    area_cm2 = max(area_cm2, 0.0)
    r = _radius_from_area(area_cm2)
    if model is ShapeModel.FLAT_DISC:
        return area_cm2 * h  # V = A × h (thin slab)
    if model is ShapeModel.DOME:
        return (2.0 / 3.0) * math.pi * r**3  # hemisphere from footprint radius
    if model is ShapeModel.ELLIPSOID:
        # semi-axes a=b=r (from footprint), c=h/2 → V = (4/3)π·r²·(h/2)
        return (4.0 / 3.0) * math.pi * r * r * (h / 2.0)
    if model is ShapeModel.CYLINDER:
        return math.pi * r * r * h  # standing cylinder, footprint radius r
    if model is ShapeModel.IRREGULAR_PILE:
        h_eff = k * math.sqrt(area_cm2)  # heaps taller as it spreads
        return area_cm2 * h_eff
    if model is ShapeModel.LIQUID_IN_BOWL:
        # No container volume without Stage 1 — approximate the fill as a shallow
        # disc of depth h (a bowl's typical fill depth). Documented in 05 §5.
        return area_cm2 * h
    raise ValueError(f"Unknown shape model: {model}")  # pragma: no cover


@dataclass(frozen=True)
class ShapePrior:
    """Per-class geometry: shape model, height prior, heap factor, mass clamp."""

    model: ShapeModel
    h: float  # height prior (cm)
    k: float  # heap factor (only used by IRREGULAR_PILE)
    clamp_min_g: float
    clamp_max_g: float


# Default for any Food-101 class we haven't individually tuned (`05 §5`): a
# heaped serving, gentle height prior, and a generous clamp so nothing sensible
# is rejected while gross segmentation blow-ups still are.
DEFAULT_SHAPE = ShapePrior(
    ShapeModel.IRREGULAR_PILE, h=1.2, k=0.35, clamp_min_g=10.0, clamp_max_g=1500.0
)

# Tuned classes. `h`/`k` follow `05 §5` (defaults h≈1.2, k≈0.35); clamps are
# class-plausible cooked-portion bounds to reject wild masks (e.g. fried egg).
SHAPE_PRIORS: dict[str, ShapePrior] = {
    # flat discs / slabs
    "pizza": ShapePrior(ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=80.0, clamp_max_g=900.0),
    "pancakes": ShapePrior(ShapeModel.FLAT_DISC, h=1.2, k=0.0, clamp_min_g=60.0, clamp_max_g=500.0),
    "waffles": ShapePrior(ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=60.0, clamp_max_g=500.0),
    "fried_egg": ShapePrior(ShapeModel.FLAT_DISC, h=1.0, k=0.0, clamp_min_g=30.0, clamp_max_g=90.0),
    "steak": ShapePrior(ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=120.0, clamp_max_g=500.0),
    "grilled_salmon": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.2, k=0.0, clamp_min_g=100.0, clamp_max_g=450.0
    ),
    # domes / scoops
    "mashed_potatoes": ShapePrior(
        ShapeModel.DOME, h=4.0, k=0.0, clamp_min_g=80.0, clamp_max_g=500.0
    ),
    "ice_cream": ShapePrior(ShapeModel.DOME, h=4.0, k=0.0, clamp_min_g=40.0, clamp_max_g=350.0),
    "guacamole": ShapePrior(ShapeModel.DOME, h=3.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0),
    # ellipsoids (round-ish discrete items)
    "apple_pie": ShapePrior(
        ShapeModel.ELLIPSOID, h=3.5, k=0.0, clamp_min_g=80.0, clamp_max_g=350.0
    ),
    "dumplings": ShapePrior(
        ShapeModel.ELLIPSOID, h=3.0, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0
    ),
    "falafel": ShapePrior(ShapeModel.ELLIPSOID, h=3.5, k=0.0, clamp_min_g=50.0, clamp_max_g=350.0),
    # cylinders (rolls / logs)
    "sushi": ShapePrior(ShapeModel.CYLINDER, h=2.5, k=0.0, clamp_min_g=30.0, clamp_max_g=300.0),
    "burrito": ShapePrior(ShapeModel.CYLINDER, h=5.0, k=0.0, clamp_min_g=150.0, clamp_max_g=650.0),
    "hot_dog": ShapePrior(ShapeModel.CYLINDER, h=3.5, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0),
    "spring_rolls": ShapePrior(
        ShapeModel.CYLINDER, h=3.0, k=0.0, clamp_min_g=40.0, clamp_max_g=250.0
    ),
    # irregular piles (heaped, spreading)
    "french_fries": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.4, clamp_min_g=60.0, clamp_max_g=500.0
    ),
    "caesar_salad": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.5, clamp_min_g=50.0, clamp_max_g=500.0
    ),
    "greek_salad": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.5, clamp_min_g=50.0, clamp_max_g=500.0
    ),
    "fried_rice": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.35, clamp_min_g=80.0, clamp_max_g=600.0
    ),
    "spaghetti_bolognese": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.4, clamp_min_g=100.0, clamp_max_g=700.0
    ),
    "pad_thai": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.4, clamp_min_g=100.0, clamp_max_g=650.0
    ),
    # liquids in a bowl
    "french_onion_soup": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=4.0, k=0.0, clamp_min_g=150.0, clamp_max_g=600.0
    ),
    "hot_and_sour_soup": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=4.0, k=0.0, clamp_min_g=150.0, clamp_max_g=600.0
    ),
    "miso_soup": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=4.0, k=0.0, clamp_min_g=120.0, clamp_max_g=500.0
    ),
    "clam_chowder": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=4.0, k=0.0, clamp_min_g=150.0, clamp_max_g=600.0
    ),
    "pho": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=5.0, k=0.0, clamp_min_g=250.0, clamp_max_g=900.0
    ),
}


def prior_for(food_class: str) -> ShapePrior:
    """Return the shape prior for a Food-101 class, or the default heaped model."""
    return SHAPE_PRIORS.get(food_class, DEFAULT_SHAPE)


def clamp_mass(food_class: str, mass_g: float) -> float:
    """Clamp an estimated mass to the class's physically-plausible band (`05 §5`)."""
    prior = prior_for(food_class)
    return max(prior.clamp_min_g, min(mass_g, prior.clamp_max_g))
