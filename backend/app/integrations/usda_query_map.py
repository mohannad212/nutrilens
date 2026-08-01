"""
app/integrations/usda_query_map.py
=====================================

WHY THIS FILE EXISTS
---------------------
The classifier speaks Food-101 (`grilled_salmon`, `beignets`, `pad_thai`). USDA
FoodData Central speaks food-science English ("Fish, salmon, Atlantic, farmed,
cooked, dry heat"). Nothing in either system translates between them, so this
file is that translation layer: one function, `build_query`, that turns a
Food-101 class plus the user's confirmed cooking method into the search string
`08 §5` sends to USDA.

WHAT PROBLEM IT SOLVES
------------------------
Searching USDA for the raw class name gives bad answers in three distinct ways,
and each needs a different fix:

1. **The class name already contains a cooking method.** `grilled_salmon`
   searched verbatim, for a user who told us they *baked* it, returns grilled
   values. The class's adjective is a Food-101 dataset artefact, not the user's
   statement about their food; the cooking method they picked in the app is.
   So the adjective is stripped and the confirmed method is appended
   (`08 §5`'s own example: `grilled_salmon` + `grilled` -> "salmon cooked grilled").
2. **The class is a dish name USDA has never heard of.** `beignets`,
   `takoyaki`, `croque_madame` and friends do not exist as Foundation or
   SR Legacy entries. Searching them returns nothing, which surfaces to the user
   as a 404 on a food they can plainly see on their plate. The override table
   below maps those onto the closest term USDA *does* index.
3. **Underscores are not word separators to a search engine.** `french_onion_soup`
   is one token; "french onion soup" is three.

WHY A LOOKUP TABLE AND NOT SOMETHING CLEVERER
------------------------------------------------
- **A hand-written table (chosen).** 101 classes is a closed, known set that
  never grows at runtime. A table is inspectable, diffable, and every entry can be
  justified individually in a defence. When a mapping is wrong, the fix is one
  line and it is obviously correct.
- **Embedding similarity / fuzzy string matching against USDA's food names.**
  Needs USDA's whole food table downloaded and an embedding model loaded to solve
  a 101-row problem, and it fails silently and unpredictably — "beignets" would
  match some arbitrary pastry with no way to see that coming.
- **Asking an LLM to name the food.** Adds a network dependency and a source of
  non-determinism to a value that is multiplied into a calorie figure. `08 §6`
  forbids fabricating nutrition; a generated search term is one step removed from
  exactly that.
- **Giving up and requiring users to search manually.** Defeats the point of the
  app (`00`: photo to log in three interactions).

HOW THE OVERRIDES ARE CHOSEN
-------------------------------
Each override is deliberately **method-neutral** — `ceviche` maps to "fish", not
"fish raw", even though ceviche is raw. The cooking method is appended
separately, so baking the method into the override would produce contradictory
queries ("fish raw cooked grilled") for a user who picked something else. Keeping
the override to the base ingredient means the two halves compose cleanly.

Where a dish has no single-ingredient equivalent, the override names its dominant
components ("macaroni cheese", "shrimp grits"). This is an approximation and is
worth saying so plainly: NutriLens reports the nutrition of the closest food USDA
has lab data for, not of the exact dish photographed. That is a stated limitation
of the portion+lookup approach, and it is why every response carries the
`03 §5` disclaimer.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "These mappings are unverified guesses. How do you know they work?"
A: They are best-effort and are *not* individually validated against the live API
   in this milestone (there is no API key wired yet). What is tested here is the
   query *construction* logic — stripping, composition, deduplication — which is
   the part with branches in it. Validating the 101 mappings against real USDA
   responses is a one-off manual exercise against the deployed key
   (`scripts/` smoke check), and a wrong mapping degrades to a poor match or a
   clean 404, never to a fabricated number.

Q: "Why strip the class's own adjective instead of trusting it?"
A: Because FR06 makes the user the authority on their own meal. They explicitly
   pick a cooking method; the dataset label is a guess made by whoever collected
   Food-101. When the two disagree, the user wins.

Q: "Why dedupe tokens?"
A: Composition can repeat a word — `fried_rice` strips to "rice", and the `fried`
   method appends "cooked fried", but `onion_rings` + fried would give "onion
   rings cooked fried" while `sashimi` + raw would give "fish raw raw". Repeated
   terms are noise at best and skew relevance scoring at worst.

COMMON STUDENT MISTAKES
--------------------------
- Sending the raw class name with underscores and wondering why every search
  misses.
- Baking the cooking method into the override table, so the user's own choice is
  silently overridden by the dataset's.
- Treating a missing override as an error. Most classes (`pizza`, `hamburger`,
  `steak`) need no mapping at all — the fallback path is the common path.
"""

from __future__ import annotations

# Cooking-method adjectives that appear *inside* Food-101 class names. These are
# stripped before the user's confirmed method is appended (see the module
# docstring, reason 1). Only tokens that are genuinely preparation methods belong
# here — `deviled` (deviled_eggs) and `frozen` (frozen_yogurt) are part of the
# dish's identity, not a method the user chose, so they stay.
_METHOD_ADJECTIVES: frozenset[str] = frozenset(
    {"grilled", "fried", "baked", "roasted", "boiled", "steamed", "raw"}
)

# `03 §10` cooking_method -> the phrase appended to the search query.
# "cooked" is included because USDA's SR Legacy descriptions consistently use it
# ("Chicken, broilers, breast, meat only, cooked, roasted"), so it pulls results
# toward prepared foods rather than raw ingredients.
# `other` maps to the empty string on purpose: the user has told us the method is
# none of the listed ones, so the honest query is the food alone rather than a
# guessed preparation.
_METHOD_PHRASES: dict[str, str] = {
    "grilled": "cooked grilled",
    "fried": "cooked fried",
    "boiled": "cooked boiled",
    "baked": "cooked baked",
    "roasted": "cooked roasted",
    "steamed": "cooked steamed",
    "raw": "raw",
    "other": "",
}

# Food-101 class -> USDA-searchable base term, for the classes where the class
# name alone finds nothing useful (module docstring, reason 2). Method-neutral by
# construction. Classes absent from this table fall back to the class name with
# underscores replaced by spaces, which is correct for the majority
# (`pizza`, `hamburger`, `steak`, `guacamole`, `hummus`, ...).
USDA_QUERY_OVERRIDES: dict[str, str] = {
    # --- dishes USDA indexes only by their main ingredient ---
    "baby_back_ribs": "pork ribs",
    "beef_carpaccio": "beef",
    "beef_tartare": "beef ground",
    "ceviche": "fish",
    "escargots": "snail",
    "filet_mignon": "beef tenderloin",
    "foie_gras": "goose liver pate",
    "fried_calamari": "squid",
    "peking_duck": "duck",
    "prime_rib": "beef rib",
    "sashimi": "fish",
    "scallops": "scallop",
    "seaweed_salad": "seaweed",
    "takoyaki": "octopus",
    "tuna_tartare": "tuna",
    "edamame": "soybeans green",
    # --- names USDA spells differently ---
    "beignets": "doughnut",
    "donuts": "doughnut",
    "cup_cakes": "cake cupcake",
    "french_fries": "potato french fried",
    "frozen_yogurt": "yogurt frozen",
    "macarons": "cookie meringue",
    "creme_brulee": "custard dessert",
    "panna_cotta": "cream dessert",
    "gnocchi": "pasta potato",
    "onion_rings": "onion rings breaded",
    # --- soups: USDA leads with "Soup," so the term order matters ---
    "clam_chowder": "soup clam chowder",
    "french_onion_soup": "soup onion",
    "hot_and_sour_soup": "soup hot sour",
    "lobster_bisque": "soup lobster bisque",
    "miso_soup": "soup miso",
    "pho": "soup noodle beef",
    # --- composite dishes: name the dominant components (see docstring) ---
    "breakfast_burrito": "burrito breakfast",
    "caprese_salad": "cheese mozzarella tomato",
    "cheese_plate": "cheese",
    "chicken_quesadilla": "quesadilla chicken",
    "club_sandwich": "sandwich turkey bacon",
    "crab_cakes": "crab cake",
    "croque_madame": "sandwich ham cheese",
    "deviled_eggs": "egg",
    "fish_and_chips": "fish battered",
    "grilled_cheese_sandwich": "sandwich cheese",
    "huevos_rancheros": "egg tortilla salsa",
    "lobster_roll_sandwich": "lobster sandwich",
    "macaroni_and_cheese": "macaroni cheese",
    "pad_thai": "noodles rice",
    "poutine": "potato french fried gravy",
    "pulled_pork_sandwich": "pork shoulder sandwich",
    "red_velvet_cake": "cake chocolate frosted",
    "shrimp_and_grits": "shrimp grits",
    "spaghetti_bolognese": "spaghetti meat sauce",
    "spaghetti_carbonara": "spaghetti cheese egg",
    "spring_rolls": "egg roll vegetable",
    "strawberry_shortcake": "cake shortcake strawberry",
    "gyoza": "dumpling pork",
}


def base_term(food101_class: str) -> str:
    """The method-neutral USDA search term for a Food-101 class.

    Override table first; otherwise the class name with underscores turned into
    spaces and any cooking-method adjective stripped, so the user's confirmed
    method is the only preparation signal in the final query.
    """
    override = USDA_QUERY_OVERRIDES.get(food101_class)
    if override is not None:
        return override

    tokens = [t for t in food101_class.split("_") if t]
    kept = [t for t in tokens if t.lower() not in _METHOD_ADJECTIVES]
    # A class that is *only* an adjective would strip to nothing; fall back to the
    # unstripped name rather than searching for an empty string.
    if not kept:
        kept = tokens
    return " ".join(kept)


def _dedupe(tokens: list[str]) -> list[str]:
    """Drop repeated tokens, preserving first-occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def build_query(food101_class: str, cooking_method: str | None = None) -> str:
    """Build the USDA `/foods/search` query string for a class + cooking method.

    `08 §5`'s worked example holds: ``build_query("grilled_salmon", "grilled")``
    returns ``"salmon cooked grilled"``.

    An unknown or `None` cooking method contributes nothing rather than raising —
    the request schema (`03 §5`) has already restricted the value to the enum, and
    a search for the food alone is a strictly better failure mode than a 500.
    """
    term = base_term(food101_class)
    phrase = _METHOD_PHRASES.get(cooking_method or "", "")
    tokens = f"{term} {phrase}".lower().split()
    return " ".join(_dedupe(tokens))
