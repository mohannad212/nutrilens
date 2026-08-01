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
   (`08 §5`'s own example: `grilled_salmon` + `grilled` -> "salmon cooked
   grilled", which the next section then marks up as `+salmon cooked grilled`).
2. **The class is a dish name USDA has never heard of.** `beignets`,
   `takoyaki`, `croque_madame` and friends do not exist as Foundation or
   SR Legacy entries. Searching them returns nothing, which surfaces to the user
   as a 404 on a food they can plainly see on their plate. The override table
   below maps those onto the closest term USDA *does* index.
3. **Underscores are not word separators to a search engine.** `french_onion_soup`
   is one token; "french onion soup" is three.

WHY FOOD TERMS ARE MARKED REQUIRED WITH `+`
----------------------------------------------
This is the single most important thing in the file, and it was learned by
running the queries against the live API rather than by reasoning about them.

USDA's search ranks on the whole query with every term optional by default. The
cooking-method words are short, extremely common in food descriptions, and
therefore dominate the score. The results are not subtly off — they are a
different food entirely:

    "salmon cooked grilled"  ->  Kielbasa, fully cooked, grilled       (337 kcal)
    "pizza cooked baked"     ->  Pork, cured, bacon, cooked, baked
    "hamburger cooked grilled" -> Kielbasa, fully cooked, grilled
    "guacamole raw"          ->  Abiyuch, raw
    "chicken curry cooked boiled" -> Chicken, feet, boiled

Every one of those would have been shown to the user as their meal's nutrition,
with no error and nothing to suggest anything had gone wrong. That is precisely
the failure `08 §6` forbids — a fabricated number wearing the label of a real
one — reached without anyone fabricating anything.

Prefixing the food terms with USDA's `+` operator makes them **required**, and
the same searches return salmon, pizza and hamburger. For the foods USDA has no
entry for at all (`guacamole`, `sushi`) it returns *nothing*, which is the
outcome we want: no match becomes an honest `404 NO_NUTRITION_MATCH` that the app
can explain, instead of a confident wrong answer. Trading availability for
truthfulness is the whole design stance of the nutrition path.

(Quoted phrases — `"salmon" cooked` — were also tried. USDA answers those with a
non-JSON error body, which the client correctly reports as an upstream failure,
so quoting is not an option.)

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
Q: "These mappings are guesses. How do you know they work?"
A: They were validated against the live API with `scripts/usda_smoke_check.py`,
   which runs every Food-101 class through the real search and prints the food
   USDA actually returns. That exercise is what produced the `+`-operator finding
   above and corrected the overrides that resolved to the wrong food. Re-running
   it is how you check the table after any change.

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
- Treating a missing override as an error. Roughly half the classes (`hamburger`,
  `steak`, `lasagna`, `waffles`) need no mapping at all — the fallback path is a
  common path, not an edge case.
- Adding an override without re-running the smoke check. Several entries here
  replaced earlier guesses that looked reasonable and resolved to the wrong food
  (`pizza` -> Goldfish crackers, `ramen` -> the dry noodle block).
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
#
# Every entry below was checked against the live API with
# `scripts/usda_smoke_check.py`; the comments record the food USDA actually
# returns, so a future edit can be compared against what it replaces.
USDA_QUERY_OVERRIDES: dict[str, str] = {
    # --- dishes USDA indexes only by their main ingredient ---
    "baby_back_ribs": "pork ribs",
    "beef_carpaccio": "beef tenderloin",  # plain "beef raw" -> cured corned beef
    "beef_tartare": "beef ground",
    "ceviche": "fish",
    "escargots": "snail",
    "filet_mignon": "beef tenderloin",
    "foie_gras": "goose liver pate",
    "fried_calamari": "squid",
    "guacamole": "avocado",  # USDA has no "guacamole" at all
    "peking_duck": "duck",
    "prime_rib": "beef rib",
    "sashimi": "fish",
    "scallops": "scallop",
    "seaweed_salad": "seaweed",
    "takoyaki": "octopus",
    "tuna_tartare": "tuna",
    "edamame": "soybeans green",
    # `oyster` alone matches "Emu, oyster" and "Mushrooms, oyster" — the
    # shellfish only surfaces once the mollusc family is named.
    "oysters": "mollusks oyster",
    # --- names USDA spells differently ---
    "pizza": "pizza cheese",  # bare "pizza" -> Goldfish *crackers*
    "beignets": "doughnut",
    "donuts": "doughnut",
    "churros": "doughnut",
    "baklava": "danish pastry",
    "cannoli": "danish pastry",
    "samosa": "turnover",
    "omelette": "omelet",  # USDA uses the American spelling
    "cup_cakes": "cake cupcake",
    "french_fries": "potato french fried",
    "frozen_yogurt": "yogurt frozen",
    "macarons": "cookie meringue",
    "creme_brulee": "desserts flan",  # -> "Desserts, flan, caramel custard"
    "panna_cotta": "cream dessert",
    "gnocchi": "pasta",
    "onion_rings": "onion rings breaded",
    "pancakes": "pancakes plain",  # bare "pancakes" -> "Potato pancakes"
    "apple_pie": "pie apple prepared",  # "pie apple" -> canned pie *filling*
    "bruschetta": "bread toasted",
    "tiramisu": "cake cheesecake",
    "bread_pudding": "pudding",
    # --- rice dishes with no USDA entry of their own ---
    # All four are approximated by their base grain; see the docstring note on
    # composite dishes. The rice dominates the mass, and USDA indexes nothing
    # closer.
    "sushi": "rice white cooked",
    "risotto": "rice white cooked",
    "paella": "rice white cooked",
    "bibimbap": "rice white cooked",
    # --- soups: USDA leads with "Soup," so the term order matters ---
    "clam_chowder": "soup clam chowder",
    "french_onion_soup": "soup onion",
    "hot_and_sour_soup": "soup hot sour",
    "lobster_bisque": "soup bisque",  # -> tomato bisque; the cream drives the macros
    "miso_soup": "soup vegetable canned",  # "miso" alone -> the *paste*, ~5x too concentrated
    "ramen": "soup noodle",  # bare "ramen" -> the *dry* block, ~10x a prepared bowl
    "pho": "soup noodle beef",
    # --- composite dishes: name the dominant components (see docstring) ---
    "breakfast_burrito": "burrito breakfast",
    "caprese_salad": "cheese mozzarella",
    "cheese_plate": "cheese",
    "chicken_curry": "chicken",  # USDA has no curry; "chicken curry" -> chicken *spread*
    "chicken_quesadilla": "quesadilla chicken",
    "club_sandwich": "sandwich club",  # -> "Fast foods, ... bacon and tomato club sandwich"
    "crab_cakes": "crab cake",
    "croque_madame": "sandwich ham cheese",
    "deviled_eggs": "egg whole",
    "eggs_benedict": "egg whole",
    "huevos_rancheros": "egg whole",
    "fish_and_chips": "fish battered",
    "grilled_cheese_sandwich": "cheese cheddar",
    "lobster_roll_sandwich": "lobster sandwich",
    "macaroni_and_cheese": "macaroni cheese",
    "pad_thai": "noodles rice",
    "poutine": "potato french fried",
    "pulled_pork_sandwich": "pork shoulder",
    "red_velvet_cake": "cake chocolate frosted",
    "shrimp_and_grits": "shrimp grits",
    "spaghetti_bolognese": "spaghetti meat sauce",
    "spaghetti_carbonara": "spaghetti cheese egg",
    "spring_rolls": "egg roll vegetable",
    "strawberry_shortcake": "cake shortcake strawberry",
    "gyoza": "dumpling pork",
    # Leaf salads. USDA indexes no composite salad, and the plausible-looking
    # alternatives are worse than the leaf: "caesar salad" matches the *dressing*
    # (542 kcal/100 g), "greek salad" matches Greek *yogurt*, and a bare "salad"
    # matches tuna salad. The leaf is the honest dominant ingredient — it
    # under-reports the dressing, oil and cheese, which is a real and documented
    # limitation of the whole lookup approach rather than of these two entries.
    "caesar_salad": "lettuce",
    "greek_salad": "lettuce",
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
    """The primary USDA query for a class + cooking method.

    Food terms are marked **required** with USDA's `+` operator; the cooking
    method is left optional so it influences ranking without excluding anything.
    ``build_query("grilled_salmon", "grilled")`` returns
    ``"+salmon cooked grilled"``.

    An unknown or `None` cooking method contributes nothing rather than raising —
    the request schema (`03 §5`) has already restricted the value to the enum, and
    a search for the food alone is a strictly better failure mode than a 500.
    """
    return build_query_chain(food101_class, cooking_method)[0]


def build_query_chain(food101_class: str, cooking_method: str | None = None) -> list[str]:
    """Progressively relaxed queries, most precise first (see the module docstring).

    The caller tries each in order and stops at the first that yields a usable
    match. Three rungs, and each exists because the one above it fails in a
    specific, observed way:

    1. ``+every +food +term  method phrase`` — all food terms required. Correct
       when USDA indexes the dish under all of them (`+soup +onion` -> "Soup,
       onion, canned, condensed").
    2. ``+primary term  method phrase`` — only the first food term required.
       Rung 1 over-constrains for composite overrides: `+rice +fish` (sushi)
       matches nothing at all, because no single USDA food is indexed under both.
    3. ``+primary term`` — no method words at all. They are dropped last because
       they are the ranking poison described above: with them present and nothing
       required, "guacamole raw" returns *Abiyuch, raw*.

    The list is deduplicated, so a single-token food with no cooking method
    produces one query rather than three identical ones.
    """
    term = base_term(food101_class)
    phrase = _METHOD_PHRASES.get(cooking_method or "", "")
    food_tokens = _dedupe(term.lower().split())
    if not food_tokens:  # defensive: base_term never returns empty in practice
        return [term.lower()]

    phrase_tokens = [t for t in phrase.lower().split() if t not in food_tokens]
    suffix = (" " + " ".join(phrase_tokens)) if phrase_tokens else ""

    all_required = " ".join(f"+{t}" for t in food_tokens)
    primary_required = f"+{food_tokens[0]}"

    chain = [
        f"{all_required}{suffix}",
        f"{primary_required}{suffix}",
        primary_required,
    ]
    return _dedupe(chain)


# Which words in a USDA description indicate the preparation the user confirmed.
# Used to re-rank candidates, not to filter them (`usda_client.select_best`).
#
# "dry heat" and "moist heat" are here because SR Legacy almost never says
# "grilled" or "boiled" — it classifies by heat transfer, so "Fish, salmon,
# chinook, cooked, dry heat" is the grilled/baked/roasted entry and looking only
# for the literal method word would miss every one of them.
METHOD_DESCRIPTION_HINTS: dict[str, tuple[str, ...]] = {
    "grilled": ("grilled", "broiled", "dry heat"),
    "fried": ("fried", "pan-fried"),
    "baked": ("baked", "dry heat"),
    "roasted": ("roasted", "dry heat"),
    "boiled": ("boiled", "simmered", "moist heat"),
    "steamed": ("steamed", "moist heat"),
    "raw": ("raw",),
    "other": (),
}


def description_matches_method(description: str, cooking_method: str | None) -> bool:
    """Whether a USDA description looks like the preparation the user confirmed."""
    hints = METHOD_DESCRIPTION_HINTS.get(cooking_method or "", ())
    if not hints:
        return False
    lowered = description.lower()
    return any(hint in lowered for hint in hints)
