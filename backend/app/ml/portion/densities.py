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
    # Filled bread items (burgers, sandwiches, tacos, rolls). Not the same thing as
    # `bread`: that is the bulk density of the crumb alone, whereas a burger is a
    # bun *plus* a denser filling, and the shape priors measure it as one solid
    # body. This is therefore an **effective** density against the item's bounding
    # geometric volume, not a material density — the two are only equal for foods
    # that fill their own bounding box. Cross-checked against published burger
    # weights (a plain fast-food hamburger ~100 g at ~9.5 cm across, a Big Mac
    # ~219 g at ~10 cm) divided by the ellipsoid volume this module would compute
    # for them, which lands the ratio at 0.45-0.55.
    "bread_composite": 0.45,
    # Shell-in seafood served in the shell (mussels, oysters, escargots) or in the
    # pod (edamame). The segmentation mask sees shells and pods, and a heap of them
    # is mostly void, so the effective density over that footprint is far below
    # either the meat's or the shell's own density.
    "shellfish_in_shell": 0.35,
    # Fried or puffed dough (beignets, churros). Lighter than `bread` because the
    # frying sets an open, largely hollow structure.
    "fried_dough": 0.30,
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
    "sashimi": GROUP_DENSITY["lean_meat"],
    "tuna_tartare": GROUP_DENSITY["lean_meat"],
    # leafy salads (loose leaves only — see the non-leafy ones further down)
    "caesar_salad": GROUP_DENSITY["leafy_salad"],
    "greek_salad": GROUP_DENSITY["leafy_salad"],
    # bread / baked (crumb only — nothing substantial layered on top)
    "garlic_bread": GROUP_DENSITY["bread"],
    "french_toast": GROUP_DENSITY["bread"],
    "pancakes": GROUP_DENSITY["bread"],
    "waffles": GROUP_DENSITY["bread"],
    "bruschetta": GROUP_DENSITY["bread"],
    # filled bread — bun/tortilla/slices plus a filling, measured as one body.
    # These were the largest remaining gap in this table: every one of them used
    # to fall through to `default` (0.90), the mixed-food value, which is ~2x too
    # heavy for anything built on bread and made the estimator read roughly 3x
    # over on a weighed burger.
    "hamburger": GROUP_DENSITY["bread_composite"],
    "club_sandwich": GROUP_DENSITY["bread_composite"],
    "grilled_cheese_sandwich": GROUP_DENSITY["bread_composite"],
    "pulled_pork_sandwich": GROUP_DENSITY["bread_composite"],
    "lobster_roll_sandwich": GROUP_DENSITY["bread_composite"],
    "hot_dog": GROUP_DENSITY["bread_composite"],
    "tacos": GROUP_DENSITY["bread_composite"],
    "chicken_quesadilla": GROUP_DENSITY["bread_composite"],
    # Denser members of the same family, kept explicit rather than forced onto the
    # group value: a burrito's filling is packed tighter than a burger's, pizza is
    # dough compressed under sauce and cheese, bread pudding is custard-soaked,
    # and a donut is fried dough that is mostly air.
    "breakfast_burrito": 0.55,
    "pizza": 0.55,
    "bread_pudding": 0.60,
    "donuts": 0.35,
    "nachos": 0.35,  # loose chips + toppings; a heap with large voids
    # liquids / soups
    "french_onion_soup": GROUP_DENSITY["liquid"],
    "hot_and_sour_soup": GROUP_DENSITY["liquid"],
    "miso_soup": GROUP_DENSITY["liquid"],
    "clam_chowder": GROUP_DENSITY["liquid"],
    "pho": GROUP_DENSITY["liquid"],
    # cheese-heavy
    "cheese_plate": GROUP_DENSITY["cheese"],
    # fried potato
    "french_fries": GROUP_DENSITY["fried_potato"],
    "poutine": GROUP_DENSITY["fried_potato"],
    # thin sliced / spread flat — solid protein, so these are *denser* than the
    # 0.90 default. Their old error came from the shape prior, not from here.
    "beef_carpaccio": GROUP_DENSITY["lean_meat"],
    "peking_duck": 1.00,  # slightly under lean_meat: duck skin carries fat
    "foie_gras": 0.95,  # dense but fatty, so just below lean meat
    # shell-in seafood
    "mussels": GROUP_DENSITY["shellfish_in_shell"],
    "escargots": GROUP_DENSITY["shellfish_in_shell"],
    "edamame": 0.30,  # pods heap with more void than a shell pile
    "oysters": 0.50,  # single layer on the half shell, so less void than a heap
    # airy baked / pastry
    "beignets": GROUP_DENSITY["fried_dough"],
    "churros": GROUP_DENSITY["fried_dough"],
    "macarons": 0.35,  # meringue shells around a thin ganache
    "cup_cakes": 0.40,  # sponge plus frosting
    "cannoli": 0.45,  # fried shell, but packed with ricotta
    "baklava": 0.70,  # the outlier here: filo compressed under nuts and syrup
    # egg dishes — these span an unusually wide range, so they are set per class
    # rather than shared. Set egg is close to water (~1.03); an omelette is the
    # same material aerated and folded, so against its bounding slab it measures
    # far lighter. Averaging them into one group value would be wrong for both.
    "deviled_eggs": 1.00,  # solid halved egg plus a little whipped filling
    "omelette": 0.60,  # aerated and folded, so its slab is part air
    "eggs_benedict": 0.65,  # muffin and hollandaise dilute the egg
    "croque_madame": 0.55,  # bread-dominated, with an egg on top
    # ---- remaining classes, completing coverage of all 101 -------------------
    # fried / battered — the shared property is voids, not the food itself. An
    # onion ring is a hollow tube; calamari and gyoza trap air between pieces.
    "onion_rings": 0.25,  # hollow tubes heaped: mostly air
    "fried_calamari": 0.40,
    "samosa": 0.50,
    "gyoza": 0.65,
    "takoyaki": 0.65,
    "crab_cakes": 0.80,
    "spring_rolls": 0.50,
    "dumplings": 0.65,
    "falafel": 0.65,
    "fish_and_chips": 0.45,  # battered fish beside chips, spread over a plate
    # saucy mains / stews — near water, since sauce fills the gaps
    "chicken_curry": 1.00,
    "lobster_bisque": 1.00,
    "shrimp_and_grits": 0.95,
    "bibimbap": GROUP_DENSITY["grains"],
    "huevos_rancheros": 0.70,
    "hummus": 1.05,  # a dense purée, not a heap
    "guacamole": 0.95,
    # pasta / dumpling mains
    "ravioli": 0.90,
    "gnocchi": 0.90,
    "lasagna": 0.95,
    "sushi": 0.95,  # packed rice with fish
    # plated meat / seafood
    "scallops": GROUP_DENSITY["lean_meat"],
    "ceviche": 1.00,
    "beef_tartare": GROUP_DENSITY["lean_meat"],
    "baby_back_ribs": 0.75,  # bone and gaps pull it below lean meat
    "chicken_wings": 0.50,  # heaped, bone-in, with large voids between wings
    # cakes and sweet baked — sponge is largely air, so well under `default`
    "chocolate_cake": 0.40,
    "carrot_cake": 0.45,
    "red_velvet_cake": 0.45,
    "strawberry_shortcake": 0.45,
    "tiramisu": 0.60,  # soaked sponge and mascarpone, denser than plain cake
    "apple_pie": 0.60,
    "cheesecake": 0.95,  # a baked custard, not the solid `cheese` value
    # set / creamy desserts
    "creme_brulee": 1.05,
    "panna_cotta": 1.00,
    "chocolate_mousse": 0.55,  # aerated
    "frozen_yogurt": 0.60,  # aerated by churning
    "ice_cream": 0.55,
    # salads that are not leaves — the `leafy_salad` value is wrong for these:
    # caprese is tomato and mozzarella slices, and seaweed salad is dense and wet.
    "caprese_salad": 0.90,
    "seaweed_salad": 0.60,
    "beet_salad": 0.55,
}


def density_of(food_class: str) -> float:
    """Return the density (g/cm³) for a Food-101 class, or the documented default.

    Never raises on an unknown class — an unmapped food gets `DEFAULT_DENSITY`
    so the pipeline still returns a (wider) range rather than failing (`05 §10`).
    """
    return CLASS_DENSITY.get(food_class, DEFAULT_DENSITY)
