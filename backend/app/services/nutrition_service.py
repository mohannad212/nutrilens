"""
app/services/nutrition_service.py
====================================

WHY THIS FILE EXISTS
---------------------
FR07 and FR08: given a food the user has confirmed, a cooking method they picked,
and a portion they adjusted, produce the macronutrients for *that* plate. This is
the module that composes the two halves — the cache (`08 §6`) and the USDA client
(`08 §5`) — and does the one piece of arithmetic the whole feature rests on.

WHAT PROBLEM IT SOLVES
------------------------
The cache and the client each know one thing and neither knows the policy. The
client knows how to ask USDA and how to fail; it cannot see that a stale row
sitting in the database is a better answer than the 502 it is about to raise. The
cache knows what it holds; it cannot decide whether holding something 200 days old
is good enough. `08 §6` specifies a fallback *order*, and an order can only be
enforced by something that sees both options. That is this file, and it is the
reason it exists rather than the endpoint calling the client directly.

THE LOOKUP POLICY, IN ORDER
------------------------------
1. **Fresh complete cache row** -> return it, `source: "cache"`. No network, no
   quota spent. After warm-up this is the overwhelmingly common path.
2. **No row, or an incomplete one** -> call USDA, cache the result, return it with
   `source: "usda"`. A USDA failure here is terminal and surfaces as the
   documented 429/502/404: there is nothing to fall back *to*, and inventing a
   number is what `08 §6` forbids.
3. **A stale complete row** -> try USDA to refresh it. On success, the row is
   updated and returned as `"usda"`. **On failure, the stale row is returned
   anyway**, as `"cache"`. This is the case that makes the whole TTL worth having:
   the failure mode of a 180-day-old lab value for salmon is that it is slightly
   out of date, and the failure mode of a 502 is that the user cannot log their
   dinner. The first is obviously preferable, and it is exactly the Reliability
   NFR `08 §6` cites.

WHAT IS DELIBERATELY *NOT* A FALLBACK
----------------------------------------
Falling back to the same food with a **different cooking method** — serving
grilled salmon's numbers for fried salmon because that row happens to be cached.
It is tempting, it would cut the miss rate, and it is rejected: frying adds oil,
so those numbers are not an approximation of the answer, they are a different
food's answer presented as this one's. `08 §6`'s "never fabricate" covers values
that were never measured for the thing being reported, not just values pulled out
of the air. When we do not have it, we say so.

THE SCALING ARITHMETIC
-------------------------
USDA reports per 100 g. The user's portion is in grams. So every macro is
`per_100g * portion_g / 100`, and that is the entire model — nutrition is linear
in mass, which is true for a homogeneous food and is the assumption the whole
per-100 g convention is built on.

Rounding follows `03 §1`: calories to whole kcal, macros to one decimal. It
happens **once, at the end**, never on intermediate values. Rounding per-100 g
first and then scaling would compound the error by the portion multiplier — at
250 g, a 0.5 kcal rounding error on the per-100 g figure becomes 1.25 kcal of
drift, and the same logic applied per item then accumulates across a meal.

WHY `Decimal` IS COERCED TO `float`
--------------------------------------
`02 §2.3` stores macros as `NUMERIC(7,2)`, which SQLAlchemy returns as
`decimal.Decimal`. Python refuses to multiply a `Decimal` by a `float`, so a
portion of `210.0` against a cached `Decimal("206.00")` raises `TypeError` — and
only on the cache-hit path, which is exactly the path that unit tests using freshly
constructed objects tend to miss. Coercing at the boundary keeps the arithmetic in
one numeric type. The precision that `NUMERIC` buys is real but it is a *storage*
property; a float carries ~15 significant digits, and these values are rounded to
one decimal before anyone sees them.

WHY THE RESULT IS A DATACLASS, NOT A PYDANTIC MODEL
------------------------------------------------------
The same reason `recommendation_service` returns one: `meal_service` is a second
consumer that needs the numbers but not the HTTP response shape (it writes them
to `meal_item` columns). Returning a plain value object means the service is
testable without constructing a response, and the schema layer stays the only
place that knows what the JSON looks like.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Nutrition is linear in mass — is that actually true?"
A: For a single homogeneous food, yes, and that is the assumption behind the
   per-100 g convention USDA itself publishes. Where it breaks down is the step
   before this one: whether the *portion* was estimated correctly (`05`), and
   whether the photographed dish matches the USDA entry chosen for it
   (`usda_query_map`). Both are acknowledged limitations, and both are why the
   `03 §5` disclaimer is on every response.

Q: "Why is `source` in the response at all? Users do not care."
A: Two reasons that are not about users. It is the assertion that makes cache
   behaviour testable at the HTTP boundary rather than by inspecting the database,
   and it is what lets the evaluation demonstrate the Reliability NFR — showing
   USDA unreachable and nutrition still arriving as `"cache"` is only convincing
   if the response says which path it took.

Q: "What if the user requests a portion of 0 g?"
A: The schema rejects it before this code runs (`gt=0`). A zero-gram item is not a
   meaningful log entry, and permitting it would put a row of zeroes into the
   recommendation engine's averages.

COMMON STUDENT MISTAKES
--------------------------
- Rounding per-100 g values before scaling, so the error grows with the portion.
- Catching the USDA errors here and returning zeros or a "sorry" object, which
  turns a clear 502 into nutrition data the user has no reason to distrust.
- Committing the session in this service. It is called inside `POST /meals`'s
  transaction (`meal_service`), where a commit halfway through would persist the
  cache write but leave the meal itself unwritten if a later item failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import UsdaRateLimitedError, UsdaUnavailableError
from app.db.models import FoodItem
from app.integrations import nutrition_cache as cache
from app.integrations.usda_client import UsdaClient

logger = logging.getLogger(__name__)

GRAMS_PER_REFERENCE_PORTION = 100.0


@dataclass(frozen=True)
class Macros:
    """One set of macronutrients, already rounded per `03 §1`."""

    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass(frozen=True)
class NutritionResult:
    """What `03 §5` returns, before the schema layer names the fields for JSON."""

    food_id: int
    food_name: str
    usda_fdc_id: str | None
    cooking_method: str
    per_100g: Macros
    portion_g: float
    scaled: Macros
    source: str  # "usda" | "cache"


def _f(value: float | Decimal | None) -> float:
    """Coerce a possibly-`Decimal` column value to `float` (see module docstring)."""
    return 0.0 if value is None else float(value)


def _round_macros(calories: float, protein: float, carbs: float, fat: float) -> Macros:
    """Apply `03 §1`'s rounding: whole kcal, one decimal for grams."""
    return Macros(
        calories=int(round(calories)),
        protein_g=round(protein, 1),
        carbs_g=round(carbs, 1),
        fat_g=round(fat, 1),
    )


def scale_to_portion(row: FoodItem, portion_g: float) -> tuple[Macros, Macros]:
    """Scale a cached row's per-100 g macros to `portion_g`.

    Returns `(per_100g, scaled)`, both rounded only at the end.
    """
    calories = _f(row.calories_per_100g)
    protein = _f(row.protein_per_100g)
    carbs = _f(row.carbs_per_100g)
    fat = _f(row.fat_per_100g)

    factor = portion_g / GRAMS_PER_REFERENCE_PORTION
    return (
        _round_macros(calories, protein, carbs, fat),
        _round_macros(calories * factor, protein * factor, carbs * factor, fat * factor),
    )


def resolve_food_item(
    db: Session,
    food101_class: str,
    cooking_method: str,
    usda: UsdaClient,
) -> tuple[FoodItem, str]:
    """Get the cached `food_item` for this food, consulting USDA if needed.

    Returns `(row, source)` where source is `"cache"` or `"usda"`, following the
    fallback order in `08 §6` (and this module's docstring). Raises
    `UsdaRateLimitedError` / `UsdaUnavailableError` / `NoNutritionMatchError` only
    when there is no cached value to fall back to.

    Shared with `meal_service`, which needs the row (for `meal_item.food_id`)
    rather than a response object.
    """
    cached = cache.get_cached(db, food101_class, cooking_method)

    if cached is not None and not cache.is_stale(cached, settings.nutrition_cache_ttl_days):
        return cached, "cache"

    try:
        result = usda.search(food101_class, cooking_method)
    except (UsdaRateLimitedError, UsdaUnavailableError):
        if cached is not None:
            # The Reliability NFR (`08 §6`): a stale lab value beats an outage.
            logger.warning(
                "USDA unreachable; serving stale cached nutrition for %s/%s",
                food101_class,
                cooking_method,
            )
            return cached, "cache"
        raise

    return cache.upsert(db, food101_class, cooking_method, result), "usda"


def get_nutrition(
    db: Session,
    food101_class: str,
    cooking_method: str,
    portion_g: float,
    usda: UsdaClient,
) -> NutritionResult:
    """FR07/FR08 — macros for one confirmed item, scaled to its confirmed portion."""
    row, source = resolve_food_item(db, food101_class, cooking_method, usda)
    per_100g, scaled = scale_to_portion(row, portion_g)

    return NutritionResult(
        food_id=row.food_id,
        food_name=row.food_name,
        usda_fdc_id=row.usda_fdc_id,
        cooking_method=cooking_method,
        per_100g=per_100g,
        portion_g=portion_g,
        scaled=scaled,
        source=source,
    )
