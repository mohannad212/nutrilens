"""
app/ml/portion/densities.py
==============================

WHY THIS FILE EXISTS
---------------------
The portion pipeline estimates a food's *volume* (cm³) from its pixel footprint
and a scale (`05 §5`), but nutrition and the user think in *grams*. Converting
volume to mass needs one physical property per food: its density (g/cm³). This
file is that lookup — from a Food-101 class to a density — so the final step
`mass_g = volume_cm3 × density` has a defensible number to multiply by.

WHAT PROBLEM IT SOLVES
------------------------
Foods differ enormously in density: a loose leafy salad is ~0.25 g/cm³ while
cheese is ~1.10 — a 4x swing that would dominate any mass estimate if guessed.
Hard-coding a single density would make every low-density food read far too heavy
(or dense foods too light). Mapping each class to a food-*group* density (from
food-science references, `05 §5`) keeps the estimate physically grounded, and a
sensible default (0.90) covers classes we haven't individually tuned.

HOW IT WORKS INTERNALLY
--------------------------
- `GROUP_DENSITY` holds the per-group values from `05 §5`
  (cooked rice/grains, lean meat, leafy salad, bread, soup, cheese, fried potato,
  and a `default`).
- `CLASS_DENSITY` maps individual Food-101 classes to a density (usually by
  pointing at a group value) for the classes where group membership is obvious.
- `density_of()` resolves a class: explicit class value first, else the default.
  It never raises — an unmapped class gets the documented default so the pipeline
  degrades gracefully (`05 §10`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Two-level table (group values + class→density) rather than a flat 101-row
  table of raw floats: the group layer names the food-science source once and
  keeps related foods consistent (all fried-potato dishes share one density), so
  a tweak to "fried potato" updates fries and poutine together.
- A single documented `default` (0.90) instead of raising on unknown classes:
  the honesty model of `05` is graceful degradation with a wider band, not a hard
  failure — a class with no tuned density still yields a usable (if looser) range.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Aren't these densities approximate?"
A: Yes, deliberately. They are food-group averages; real density varies with
   preparation and air content. That approximation is exactly why the pipeline
   returns a *range* the user confirms, never a fake-precise single gram value
   (`05 §1, §9`).

Q: "Why 0.90 for the default?"
A: It is a mid-range value for mixed cooked foods — close to water (1.0) but
   accounting for air gaps in typical plated portions. It's the `default` row of
   the `05 §5` table.
"""

from __future__ import annotations

# Food-group densities (g/cm³) from `05 §5`, values from food-science references.
GROUP_DENSITY: dict[str, float] = {
    "grains": 0.85,  # cooked rice / grains / pasta
    "lean_meat": 1.05,  # lean cooked meat / fish
    "leafy_salad": 0.25,  # loose leaves
    "bread": 0.30,  # bread / baked goods
    "liquid": 1.00,  # soup / broth / milk
    "cheese": 1.10,
    "fried_potato": 0.55,  # fries / crisps
    "default": 0.90,  # unlisted mixed foods
}

DEFAULT_DENSITY = GROUP_DENSITY["default"]

# Per-class density (g/cm³) for Food-101 classes where the food group is clear.
# Most entries just point at a `GROUP_DENSITY` value so related foods stay
# consistent. Anything not listed falls back to `DEFAULT_DENSITY` in `density_of`.
CLASS_DENSITY: dict[str, float] = {
    # grains / rice / pasta
    "fried_rice": GROUP_DENSITY["grains"],
    "risotto": GROUP_DENSITY["grains"],
    "paella": GROUP_DENSITY["grains"],
    "spaghetti_bolognese": GROUP_DENSITY["grains"],
    "spaghetti_carbonara": GROUP_DENSITY["grains"],
    "macaroni_and_cheese": GROUP_DENSITY["grains"],
    "ramen": GROUP_DENSITY["grains"],
    "pad_thai": GROUP_DENSITY["grains"],
    # lean meat / fish
    "grilled_salmon": GROUP_DENSITY["lean_meat"],
    "steak": GROUP_DENSITY["lean_meat"],
    "filet_mignon": GROUP_DENSITY["lean_meat"],
    "pork_chop": GROUP_DENSITY["lean_meat"],
    "prime_rib": GROUP_DENSITY["lean_meat"],
    "chicken_wings": GROUP_DENSITY["lean_meat"],
    "sashimi": GROUP_DENSITY["lean_meat"],
    "tuna_tartare": GROUP_DENSITY["lean_meat"],
    # leafy salads
    "caesar_salad": GROUP_DENSITY["leafy_salad"],
    "greek_salad": GROUP_DENSITY["leafy_salad"],
    "caprese_salad": GROUP_DENSITY["leafy_salad"],
    "beet_salad": GROUP_DENSITY["leafy_salad"],
    "seaweed_salad": GROUP_DENSITY["leafy_salad"],
    # bread / baked
    "garlic_bread": GROUP_DENSITY["bread"],
    "french_toast": GROUP_DENSITY["bread"],
    "pancakes": GROUP_DENSITY["bread"],
    "waffles": GROUP_DENSITY["bread"],
    "bruschetta": GROUP_DENSITY["bread"],
    # liquids / soups
    "french_onion_soup": GROUP_DENSITY["liquid"],
    "hot_and_sour_soup": GROUP_DENSITY["liquid"],
    "miso_soup": GROUP_DENSITY["liquid"],
    "clam_chowder": GROUP_DENSITY["liquid"],
    "pho": GROUP_DENSITY["liquid"],
    # cheese-heavy
    "cheese_plate": GROUP_DENSITY["cheese"],
    "cheesecake": GROUP_DENSITY["cheese"],
    # fried potato
    "french_fries": GROUP_DENSITY["fried_potato"],
    "poutine": GROUP_DENSITY["fried_potato"],
}


def density_of(food_class: str) -> float:
    """Return the density (g/cm³) for a Food-101 class, or the documented default.

    Never raises on an unknown class — an unmapped food gets `DEFAULT_DENSITY`
    so the pipeline still returns a (wider) range rather than failing (`05 §10`).
    """
    return CLASS_DENSITY.get(food_class, DEFAULT_DENSITY)
