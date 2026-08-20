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
    # thin / spread flat
    #
    # `DEFAULT_SHAPE` is at its worst here, and for a reason worth stating: an
    # `irregular_pile` sets height as `k·√A`, so the *wider* a food spreads the
    # *deeper* it is assumed to be. That is exactly backwards for anything sliced
    # and fanned across a plate. Carpaccio covering 380 cm² was being modelled as
    # ~6.8 cm deep — seven centimetres of paper-thin raw beef — which pinned it to
    # the 1500 g clamp against a real ~120 g serving. A flat disc with a small
    # fixed `h` is the whole fix; their densities were never the problem.
    "beef_carpaccio": ShapePrior(
        ShapeModel.FLAT_DISC, h=0.35, k=0.0, clamp_min_g=60.0, clamp_max_g=250.0
    ),
    "peking_duck": ShapePrior(
        ShapeModel.FLAT_DISC, h=1.0, k=0.0, clamp_min_g=100.0, clamp_max_g=500.0
    ),
    "foie_gras": ShapePrior(
        ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=40.0, clamp_max_g=200.0
    ),
    # shell-in seafood
    #
    # Split by how they are served, not by what they are: oysters and escargots
    # arrive as a single flat layer (a disc), whereas mussels and edamame really do
    # heap, so they keep `IRREGULAR_PILE` — but at a gentler `k`, since a pile of
    # shells slumps flatter than a pile of fries. The heavy lifting for this family
    # is the density: the mask measures shells and pods, not the meat inside.
    "oysters": ShapePrior(ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=60.0, clamp_max_g=600.0),
    "escargots": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    "mussels": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=100.0, clamp_max_g=700.0
    ),
    "edamame": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=40.0, clamp_max_g=400.0
    ),
    # airy baked / pastry
    #
    # `CYLINDER` here is V = A·h, i.e. a full prism of the footprint — this module
    # derives every model from area alone, so a cylinder *lying down* is not
    # modelled as round in cross-section. For churros and cannoli that overstates
    # volume by ~4/π, so their `h` is set below the true stick diameter to absorb
    # it rather than leaving a known 1.27x bias in the result.
    "beignets": ShapePrior(ShapeModel.ELLIPSOID, h=4.0, k=0.0, clamp_min_g=30.0, clamp_max_g=250.0),
    "cup_cakes": ShapePrior(
        ShapeModel.ELLIPSOID, h=5.0, k=0.0, clamp_min_g=30.0, clamp_max_g=250.0
    ),
    "churros": ShapePrior(ShapeModel.CYLINDER, h=2.0, k=0.0, clamp_min_g=30.0, clamp_max_g=250.0),
    "cannoli": ShapePrior(ShapeModel.CYLINDER, h=3.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0),
    "macarons": ShapePrior(ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=15.0, clamp_max_g=150.0),
    "baklava": ShapePrior(ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=30.0, clamp_max_g=250.0),
    # egg dishes
    #
    # Halved eggs are the clearest case in the whole table for why
    # `IRREGULAR_PILE` is the wrong default. They are laid out in a *single layer*
    # — the plate is what limits them, not stacking — so spreading them over a
    # wider plate must not imply a deeper serving. `h` is the geometry of one
    # half: a large egg is ~4.4 cm across its short axis, so a lengthwise half
    # stands ~2.2 cm, plus a little for a deviled filling mound.
    "deviled_eggs": ShapePrior(
        ShapeModel.ELLIPSOID, h=2.5, k=0.0, clamp_min_g=40.0, clamp_max_g=500.0
    ),
    "omelette": ShapePrior(ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=60.0, clamp_max_g=450.0),
    "eggs_benedict": ShapePrior(
        ShapeModel.ELLIPSOID, h=5.0, k=0.0, clamp_min_g=120.0, clamp_max_g=550.0
    ),
    "croque_madame": ShapePrior(
        ShapeModel.FLAT_DISC, h=3.5, k=0.0, clamp_min_g=120.0, clamp_max_g=500.0
    ),
    # domes / scoops
    "mashed_potatoes": ShapePrior(
        ShapeModel.DOME, h=4.0, k=0.0, clamp_min_g=80.0, clamp_max_g=500.0
    ),
    "ice_cream": ShapePrior(ShapeModel.DOME, h=4.0, k=0.0, clamp_min_g=40.0, clamp_max_g=350.0),
    "guacamole": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=1.8, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    # filled bread (burgers / sandwiches / tacos / rolls)
    #
    # These all used to fall through to `DEFAULT_SHAPE`, an `irregular_pile` whose
    # effective height *grows* with footprint (`h_eff = k·√A`). That is the wrong
    # model for a stacked item: a burger does not get taller as it gets wider, and
    # because `IRREGULAR_PILE` makes mass ∝ A^1.5 it also turns any small
    # over-segmentation into a large mass error. A bounded height prior makes mass
    # ∝ A instead, so the estimate degrades linearly with the mask.
    #
    # `ELLIPSOID` for the domed ones (V = ⅔·A·h) because a bun is not a full prism
    # of its footprint; `FLAT_DISC` (V = A·h) for the genuinely slab-shaped ones,
    # where the food really does fill its bounding box; `CYLINDER` for the logs.
    "hamburger": ShapePrior(
        ShapeModel.ELLIPSOID, h=6.0, k=0.0, clamp_min_g=90.0, clamp_max_g=600.0
    ),
    "pulled_pork_sandwich": ShapePrior(
        ShapeModel.ELLIPSOID, h=6.0, k=0.0, clamp_min_g=120.0, clamp_max_g=550.0
    ),
    "tacos": ShapePrior(ShapeModel.ELLIPSOID, h=4.0, k=0.0, clamp_min_g=60.0, clamp_max_g=450.0),
    "club_sandwich": ShapePrior(
        ShapeModel.FLAT_DISC, h=5.0, k=0.0, clamp_min_g=120.0, clamp_max_g=600.0
    ),
    "grilled_cheese_sandwich": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=70.0, clamp_max_g=350.0
    ),
    "chicken_quesadilla": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=70.0, clamp_max_g=450.0
    ),
    "lobster_roll_sandwich": ShapePrior(
        ShapeModel.CYLINDER, h=5.0, k=0.0, clamp_min_g=100.0, clamp_max_g=450.0
    ),
    "breakfast_burrito": ShapePrior(
        ShapeModel.CYLINDER, h=5.0, k=0.0, clamp_min_g=150.0, clamp_max_g=650.0
    ),
    # other baked goods that were also still on the default pile
    "bread_pudding": ShapePrior(
        ShapeModel.FLAT_DISC, h=3.5, k=0.0, clamp_min_g=80.0, clamp_max_g=400.0
    ),
    "french_toast": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0
    ),
    "garlic_bread": ShapePrior(
        ShapeModel.FLAT_DISC, h=3.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    # Thinner than garlic_bread despite looking similar: a bruschetta base is a
    # sliced baguette round, not a split-and-buttered half loaf.
    "bruschetta": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    "donuts": ShapePrior(ShapeModel.ELLIPSOID, h=3.5, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0),
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
    # ---- remaining classes, completing coverage of all 101 ------------------
    # plated meat / seafood — single-layer foods, so a fixed height, not `k·√A`
    "scallops": ShapePrior(ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0),
    "ceviche": ShapePrior(ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0),
    "baby_back_ribs": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=150.0, clamp_max_g=700.0
    ),
    "filet_mignon": ShapePrior(
        ShapeModel.FLAT_DISC, h=3.5, k=0.0, clamp_min_g=120.0, clamp_max_g=450.0
    ),
    "pork_chop": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=120.0, clamp_max_g=450.0
    ),
    "prime_rib": ShapePrior(
        ShapeModel.FLAT_DISC, h=3.5, k=0.0, clamp_min_g=150.0, clamp_max_g=600.0
    ),
    "sashimi": ShapePrior(ShapeModel.FLAT_DISC, h=1.2, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0),
    "cheese_plate": ShapePrior(
        ShapeModel.FLAT_DISC, h=1.5, k=0.0, clamp_min_g=50.0, clamp_max_g=400.0
    ),
    # `beef_tartare` and `tuna_tartare` are the exception in this group: they are
    # ring-moulded mounds, not a flat layer. `ELLIPSOID` rather than `DOME`
    # because `DOME` ignores `h` entirely — it builds a hemisphere from the
    # footprint radius, which for a 10 cm ring implies a 5 cm dome.
    "beef_tartare": ShapePrior(
        ShapeModel.ELLIPSOID, h=3.0, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0
    ),
    "tuna_tartare": ShapePrior(
        ShapeModel.ELLIPSOID, h=3.0, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0
    ),
    "chicken_wings": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=80.0, clamp_max_g=600.0
    ),
    # fried / battered — these genuinely heap, so the pile model stays; the fix
    # for them is the density (voids), not the geometry.
    "onion_rings": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=50.0, clamp_max_g=400.0
    ),
    "fried_calamari": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=60.0, clamp_max_g=450.0
    ),
    "samosa": ShapePrior(ShapeModel.ELLIPSOID, h=3.5, k=0.0, clamp_min_g=50.0, clamp_max_g=400.0),
    "gyoza": ShapePrior(ShapeModel.ELLIPSOID, h=2.5, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0),
    "takoyaki": ShapePrior(ShapeModel.ELLIPSOID, h=3.5, k=0.0, clamp_min_g=60.0, clamp_max_g=400.0),
    "crab_cakes": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0
    ),
    "fish_and_chips": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=200.0, clamp_max_g=900.0
    ),
    "nachos": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.35, clamp_min_g=60.0, clamp_max_g=500.0
    ),
    "poutine": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.30, clamp_min_g=150.0, clamp_max_g=700.0
    ),
    # saucy mains / stews — `LIQUID_IN_BOWL` (V = A·h) models a bowl filled to a
    # typical depth. The mask sees the surface, so height is a fill depth, not a
    # heap that grows with area.
    "chicken_curry": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=2.0, k=0.0, clamp_min_g=150.0, clamp_max_g=700.0
    ),
    "lobster_bisque": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=2.0, k=0.0, clamp_min_g=150.0, clamp_max_g=600.0
    ),
    "shrimp_and_grits": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=2.0, k=0.0, clamp_min_g=150.0, clamp_max_g=650.0
    ),
    "bibimbap": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=2.0, k=0.0, clamp_min_g=200.0, clamp_max_g=800.0
    ),
    "ramen": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=2.5, k=0.0, clamp_min_g=200.0, clamp_max_g=900.0
    ),
    "huevos_rancheros": ShapePrior(
        ShapeModel.FLAT_DISC, h=1.8, k=0.0, clamp_min_g=150.0, clamp_max_g=700.0
    ),
    "hummus": ShapePrior(ShapeModel.FLAT_DISC, h=1.2, k=0.0, clamp_min_g=50.0, clamp_max_g=350.0),
    # pasta / dumpling mains — a single saucy layer, not a mound
    "ravioli": ShapePrior(ShapeModel.FLAT_DISC, h=1.8, k=0.0, clamp_min_g=100.0, clamp_max_g=500.0),
    "gnocchi": ShapePrior(ShapeModel.FLAT_DISC, h=1.8, k=0.0, clamp_min_g=100.0, clamp_max_g=500.0),
    "lasagna": ShapePrior(ShapeModel.FLAT_DISC, h=3.0, k=0.0, clamp_min_g=150.0, clamp_max_g=550.0),
    "macaroni_and_cheese": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=100.0, clamp_max_g=600.0
    ),
    "spaghetti_carbonara": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.20, clamp_min_g=120.0, clamp_max_g=600.0
    ),
    "risotto": ShapePrior(ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=100.0, clamp_max_g=600.0),
    "paella": ShapePrior(ShapeModel.FLAT_DISC, h=2.5, k=0.0, clamp_min_g=150.0, clamp_max_g=700.0),
    # cakes and sweet baked — a served slice, so a slab of fixed depth
    "chocolate_cake": ShapePrior(
        ShapeModel.FLAT_DISC, h=4.0, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0
    ),
    "carrot_cake": ShapePrior(
        ShapeModel.FLAT_DISC, h=4.0, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0
    ),
    "red_velvet_cake": ShapePrior(
        ShapeModel.FLAT_DISC, h=4.0, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0
    ),
    "strawberry_shortcake": ShapePrior(
        ShapeModel.FLAT_DISC, h=4.5, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0
    ),
    "tiramisu": ShapePrior(ShapeModel.FLAT_DISC, h=3.5, k=0.0, clamp_min_g=60.0, clamp_max_g=350.0),
    "cheesecake": ShapePrior(
        ShapeModel.FLAT_DISC, h=4.0, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0
    ),
    # set / creamy desserts
    "creme_brulee": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=1.5, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0
    ),
    "chocolate_mousse": ShapePrior(
        ShapeModel.LIQUID_IN_BOWL, h=4.0, k=0.0, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    "panna_cotta": ShapePrior(
        ShapeModel.ELLIPSOID, h=3.0, k=0.0, clamp_min_g=60.0, clamp_max_g=300.0
    ),
    "frozen_yogurt": ShapePrior(
        ShapeModel.ELLIPSOID, h=6.0, k=0.0, clamp_min_g=50.0, clamp_max_g=350.0
    ),
    # non-leafy salads — these are sliced or dense, so not a loose leafy heap
    "caprese_salad": ShapePrior(
        ShapeModel.FLAT_DISC, h=2.0, k=0.0, clamp_min_g=80.0, clamp_max_g=400.0
    ),
    "seaweed_salad": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.25, clamp_min_g=40.0, clamp_max_g=300.0
    ),
    "beet_salad": ShapePrior(
        ShapeModel.IRREGULAR_PILE, h=1.2, k=0.28, clamp_min_g=60.0, clamp_max_g=450.0
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
