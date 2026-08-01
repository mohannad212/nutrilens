"""
app/schemas/nutrition.py
===========================

WHY THIS FILE EXISTS
---------------------
`03` calls itself a contract: "Request/response shapes here are authoritative;
the Pydantic schemas in `app/schemas/` and the TypeScript types in
`mobile/src/types/` must match exactly." This file is the backend half of that
promise for `POST /nutrition` (`03 §5`) — the request Pydantic will reject before
any service runs, and the response shape the mobile Nutrition Summary screen
reads.

WHAT PROBLEM IT SOLVES
------------------------
Two distinct jobs that happen to live in the same file.

The **request** model is the validation boundary. Everything past it may assume
the cooking method is one of eight values and the portion is a positive, plausible
mass. Without that, `portion_g: -50` reaches the scaling arithmetic and produces
negative calories, and `portion_g: 1e9` produces a number that overflows
`NUMERIC(7,2)` when the meal is written — a 500 originating four modules away from
the bad input.

The **response** model is what makes the payload's guarantees structural rather
than remembered. The disclaimer in particular is a required field with a constant
default (`core/constants.py`), so a response object *cannot be constructed
without it*. That is the difference between "we always remember to add the
disclaimer" and "the code cannot express a response that lacks one" — and `03 §5`
plus the wellbeing NFR require the latter.

WHY BOTH `per_100g` AND `scaled` ARE RETURNED
------------------------------------------------
`scaled` alone would be enough to render the screen. `per_100g` is included
because it is the *evidence*: it is the number USDA actually published, and
together with `portion_g` it lets anyone — a user, a marker, a test — verify the
arithmetic in their head. It also means the app can re-scale locally when the user
nudges the portion slider without a second round trip, which matters for the
three-interaction target in `00`.

WHY `usda_fdc_id` IS NULLABLE HERE
-------------------------------------
`03 §5`'s example shows a string, and it usually is one. It can legitimately be
absent because `food_item.usda_fdc_id` is UNIQUE while the cache key is
`(food101_class, cooking_method)` — two cooking methods that resolve to the same
USDA food can only record the provenance id once (the reasoning is in
`integrations/nutrition_cache.py`). Typing it as non-nullable would make the
schema lie about a state the database can genuinely be in, and Pydantic would
raise a `ValidationError` while serialising a perfectly good response.

WHY `food101_class` IS NOT VALIDATED AGAINST THE 101 LABELS
--------------------------------------------------------------
It is checked for *shape* (length and character set) but not for membership. The
label list lives in `models/labels_food101.json`, which the training track ships
via a GitHub Release and which may legitimately be absent (`04 §7`, and the
lifespan in `main.py` already degrades gracefully when it is). Making a request
schema depend on a file that might not exist would turn a missing model artefact
into a 500 on every nutrition call. An unknown class instead flows through to the
lookup and returns the documented `404 NO_NUTRITION_MATCH`, which is both accurate
and the same answer a real-but-unmatchable class gets.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why cap the portion at 5 kg rather than leaving it open?"
A: `portion_confirmed_g` is `NUMERIC(6,1)` (`02 §2.4`), so values at or above
   100 000 g cannot be stored at all — an uncapped field moves that failure from a
   clear 422 at the edge to a database error mid-transaction. 5 kg is far beyond
   any single photographed item while leaving every real meal untouched.

Q: "Why is `calories` an `int` when the database column is `NUMERIC(7,2)`?"
A: The column stores the computed value at full precision for later summation;
   the *response* rounds to whole kcal per `03 §1`, because a calorie figure
   presented to a tenth implies a precision the estimate does not have.

COMMON STUDENT MISTAKES
--------------------------
- Making the disclaimer an optional field, so it is present exactly as often as
  each route remembers it.
- Accepting `cooking_method: str` and validating it inside the service, which
  produces a 500-flavoured failure instead of FastAPI's structured 422.
- Letting the response model expose the ORM row directly, which would publish
  `cached_at` and any column added later without anyone deciding to.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import NUTRITION_DISCLAIMER
from app.core.enums import CookingMethod
from app.services.nutrition_service import NutritionResult

# `02 §2.4` stores portions as NUMERIC(6,1). The cap is a sanity bound well inside
# that, not the column's true limit — see the docstring.
MAX_PORTION_G = 5000.0


class MacroSet(BaseModel):
    """One block of macronutrients (`03 §5`'s `per_100g` / `scaled` objects)."""

    model_config = ConfigDict(from_attributes=True)

    calories: int = Field(ge=0, description="Whole kcal, per `03 §1` rounding")
    protein_g: float = Field(ge=0, description="Grams, 1 decimal")
    carbs_g: float = Field(ge=0, description="Grams, 1 decimal")
    fat_g: float = Field(ge=0, description="Grams, 1 decimal")


class NutritionRequest(BaseModel):
    """`POST /nutrition` request body (`03 §5`)."""

    food101_class: str = Field(
        min_length=1,
        max_length=50,  # matches food_item.food101_class VARCHAR(50) (02 §2.3)
        pattern=r"^[a-z0-9_]+$",
        description="Raw Food-101 class as returned by /recognise, e.g. 'grilled_salmon'",
    )
    cooking_method: CookingMethod = Field(
        description="User-confirmed preparation; appended to the USDA query (FR07)"
    )
    portion_g: float = Field(
        gt=0,
        le=MAX_PORTION_G,
        description="User-confirmed mass in grams (FR06)",
    )


class NutritionResponse(BaseModel):
    """`POST /nutrition` response body (`03 §5`)."""

    food_id: int
    food_name: str
    usda_fdc_id: str | None = Field(
        default=None,
        description="USDA FoodData Central id; null when provenance could not be recorded",
    )
    cooking_method: str
    per_100g: MacroSet
    portion_g: float
    scaled: MacroSet
    source: str = Field(description='"usda" on a live lookup, "cache" on a cached read')
    # Required with a constant default: a response cannot be built without it.
    disclaimer: str = NUTRITION_DISCLAIMER

    @classmethod
    def from_result(cls, result: NutritionResult) -> "NutritionResponse":
        """Map the service's dataclass onto the contract shape.

        Kept here rather than in the router so the field-by-field correspondence
        with `03 §5` lives beside the fields themselves. The import direction
        (schema -> service) matches `schemas/recommendation.py`: services return
        plain values and never know what the JSON looks like.
        """
        return cls(
            food_id=result.food_id,
            food_name=result.food_name,
            usda_fdc_id=result.usda_fdc_id,
            cooking_method=result.cooking_method,
            per_100g=MacroSet.model_validate(result.per_100g),
            portion_g=result.portion_g,
            scaled=MacroSet.model_validate(result.scaled),
            source=result.source,
        )
