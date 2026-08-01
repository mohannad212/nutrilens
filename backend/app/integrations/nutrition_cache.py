"""
app/integrations/nutrition_cache.py
======================================

WHY THIS FILE EXISTS
---------------------
`08 §6` opens with the sentence that defines this module: "the `food_item` table
**is** the cache". There is no Redis, no in-memory dict, no second store — the
same table that gives `meal_item` its foreign key is also what stops NutriLens
calling USDA twice for the same food. This file owns the two operations that view
implies: read a usable cached row, and write one back after a USDA lookup.

WHAT PROBLEM IT SOLVES
------------------------
Three things, and it is worth separating them because they justify the design
independently:

1. **Latency.** A USDA round trip is 200-800 ms. The Performance NFR budgets 5 s
   for the *whole* photo-to-nutrition path, most of which is already spent on
   inference. After warm-up almost every lookup should be a local indexed read.
2. **Quota.** `USDA_RATE_LIMIT_PER_HOUR` is a shared, finite budget
   (`core/rate_limit.py`). A cache hit costs none of it.
3. **Reliability.** `08 §6` states this explicitly: if USDA is down but the value
   was cached, the user still gets nutrition. This is the NFR that turns a
   third-party outage from an outage of *our* app into a degradation.

WHY A DATABASE TABLE AND NOT A REAL CACHE
--------------------------------------------
- **The `food_item` table (chosen).** The rows have to exist anyway —
  `meal_item.food_id` is a foreign key into them (`02 §2.4`), so a meal logged in
  July still resolves its food name in December. Given that, a separate cache
  would be a *second* copy of data the schema already stores permanently, with
  its own invalidation problem.
- **Redis / an LRU dict.** Faster per read, and the obvious choice if this were
  purely ephemeral. But it is not: eviction would silently turn a logged meal's
  nutrition into a dangling reference, and the row would have to be written to
  Postgres regardless.
- **No cache, always call USDA.** Simplest, and wrong on all three counts above.

HOW IT WORKS INTERNALLY
--------------------------
The lookup key is `(food101_class, cooking_method)`, backed by the unique index
`idx_food_item_class_method` (`02 §2.3`). "Grilled chicken" and "fried chicken"
are deliberately distinct rows: the cooking method changes the nutrition, so
collapsing them would be a correctness bug dressed as a cache optimisation.

**A row is only a cache hit if all four macros are present.** `seed_food101_classes`
pre-creates 101 rows with null macros so the class list exists before any lookup
(`02 §5`); treating one of those as a hit would return an empty nutrition object as
though it were data. `get_cached` therefore filters on the macro columns, not on
the row's existence.

**Freshness.** `cached_at` supports the optional TTL `08 §6` describes (default 180
days). A stale row is still returned — flagged as stale — because the caller's
correct response to "stale, and USDA is down" is to serve the stale value, not to
fail. That decision belongs to `nutrition_service`, so this module reports the
fact and does not act on it.

THE PLACEHOLDER-CLAIMING RULE
--------------------------------
`08 §6`'s write path says to "update if the row exists with null macros from
seeding". Those seeded rows have `cooking_method = NULL`, so the first real lookup
for a class -- say `(pizza, "baked")` -- does not match one by key. Rather than
leaving 101 permanently empty rows beside the real ones, `upsert` claims the
class's placeholder if it is still unclaimed, filling in its cooking method and
macros. Later methods for the same class (`(pizza, "fried")`) insert new rows
normally. The consequence, stated plainly: which cooking method inherits the
placeholder depends on which the user picks first. That is invisible in the API —
every row is looked up by key, never by "is this the placeholder" — and the
alternative is a table where two thirds of the rows are dead.

THE `usda_fdc_id` UNIQUE CONSTRAINT, AND WHY IT IS DEGRADED RATHER THAN ENFORCED
----------------------------------------------------------------------------------
`02 §2.3` marks `usda_fdc_id` UNIQUE, but the cache key is
`(food101_class, cooking_method)`. Those two constraints genuinely conflict:
searching "salmon cooked grilled" and "salmon cooked baked" very plausibly returns
the *same* USDA food, so two legitimately distinct cache rows want the same
`fdc_id`, and the second insert would violate the unique index.

The resolution here is to keep the schema's constraint intact and store `NULL` for
the later row's `usda_fdc_id`, logging when that happens. The reasoning: the
column is *provenance*, not data. Losing it on a duplicate costs the ability to
trace one row back to its USDA entry; the macros, which are what the user
actually sees, are unaffected. The alternatives were worse — dropping the unique
constraint contradicts the authoritative DDL in `02`, and refusing to cache the
second method would send every subsequent lookup for it to USDA forever. This is
a schema tension worth naming in the report rather than papering over.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Is this cache per-user? Could one user see another's data through it?"
A: It is shared and deliberately so — `food_item` is reference data, not user
   data (`02 §1`, which is also why its foreign key is `ON DELETE RESTRICT`).
   Nothing user-specific is stored in it: no user id, no portion, no timestamp of
   anyone's meal. Two users eating grilled salmon read the same lab values.

Q: "What invalidates an entry?"
A: Time, via `cached_at` and the TTL — and nothing else, because the underlying
   fact (the lab-measured composition of a food) does not change. USDA revising a
   value is the only real invalidation event, and a 180-day refresh is a
   proportionate response to something that happens on the order of years.

Q: "Why does `get_cached` not just return the row and let the caller check the
    macros?"
A: Because every caller would have to remember, and one that forgot would ship
   nulls to the user as nutrition. Making an incomplete row *not a hit* puts the
   rule in one place.

COMMON STUDENT MISTAKES
--------------------------
- Treating "a row exists" as "the cache has data", so the seeded placeholders
  return empty nutrition.
- Keying the cache on `food101_class` alone, so fried chicken reports grilled
  chicken's calories.
- Writing to the cache before checking whether the row already exists, and
  hitting the unique index on the second lookup for the same food.
- Committing inside the cache layer, which takes the transaction decision away
  from the service that knows whether the surrounding work succeeded.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import FoodItem
from app.integrations.usda_client import UsdaFood

logger = logging.getLogger(__name__)


def _as_naive_utc(value: datetime) -> datetime:
    """Normalise a timestamp for comparison against `datetime.now(UTC)`.

    `cached_at` is `TIMESTAMP` without a time zone (`02 §2.3`) and comes back
    naive from Postgres, while the app's clock is timezone-aware. Comparing the
    two directly raises `TypeError`, so the aware side is what gets normalised at
    the comparison site; this helper handles a row that is aware anyway (SQLite
    in tests can return either).
    """
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def has_complete_macros(row: FoodItem) -> bool:
    """True when the row carries all four macros — the definition of a cache hit.

    A seeded placeholder (`02 §5`) exists but has none of them, and must never be
    mistaken for data.
    """
    return (
        row.calories_per_100g is not None
        and row.protein_per_100g is not None
        and row.carbs_per_100g is not None
        and row.fat_per_100g is not None
    )


def is_stale(row: FoodItem, ttl_days: int) -> bool:
    """True when the row is older than the TTL. `ttl_days <= 0` disables expiry."""
    if ttl_days <= 0 or row.cached_at is None:
        return False
    age_limit = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=ttl_days)
    return _as_naive_utc(row.cached_at) < age_limit


def get_row(db: Session, food101_class: str, cooking_method: str) -> FoodItem | None:
    """The `food_item` row for this exact cache key, complete or not."""
    return db.execute(
        select(FoodItem).where(
            FoodItem.food101_class == food101_class,
            FoodItem.cooking_method == cooking_method,
        )
    ).scalar_one_or_none()


def get_cached(db: Session, food101_class: str, cooking_method: str) -> FoodItem | None:
    """A **usable** cached row, or `None`.

    Incomplete rows are not hits (see the module docstring). Staleness is *not*
    filtered here: `nutrition_service` needs to see a stale row so it can fall
    back to it when USDA is unreachable (`08 §6`).
    """
    row = get_row(db, food101_class, cooking_method)
    if row is None or not has_complete_macros(row):
        return None
    return row


def _unclaimed_placeholder(db: Session, food101_class: str) -> FoodItem | None:
    """A seeded `(class, NULL)` row that no cooking method has claimed yet."""
    row = (
        db.execute(
            select(FoodItem).where(
                FoodItem.food101_class == food101_class,
                FoodItem.cooking_method.is_(None),
            )
        )
        .scalars()
        .first()
    )
    if row is None or has_complete_macros(row):
        return None
    return row


def _fdc_id_is_free(db: Session, fdc_id: str, exclude_food_id: int | None) -> bool:
    """Whether `usda_fdc_id` can be stored without violating its unique index."""
    existing = (
        db.execute(select(FoodItem.food_id).where(FoodItem.usda_fdc_id == fdc_id)).scalars().first()
    )
    return existing is None or existing == exclude_food_id


def upsert(db: Session, food101_class: str, cooking_method: str, usda: UsdaFood) -> FoodItem:
    """Write a USDA result into the cache and return the row (`08 §6` write path).

    Resolution order: the exact `(class, method)` row, else the class's unclaimed
    seeded placeholder, else a new row. Flushes so `food_id` is populated for the
    caller, but does **not** commit — the surrounding request owns that decision.
    """
    row = get_row(db, food101_class, cooking_method)
    if row is None:
        row = _unclaimed_placeholder(db, food101_class)
        if row is not None:
            row.cooking_method = cooking_method

    if row is None:
        row = FoodItem(food101_class=food101_class, cooking_method=cooking_method)
        db.add(row)

    row.food_name = usda.description
    row.calories_per_100g = usda.calories_per_100g
    row.protein_per_100g = usda.protein_per_100g
    row.carbs_per_100g = usda.carbs_per_100g
    row.fat_per_100g = usda.fat_per_100g
    row.cached_at = datetime.now(UTC).replace(tzinfo=None)

    # Provenance, best-effort: the column is UNIQUE but the cache key is not, so a
    # food shared by two cooking methods can only record its FDC id once. See the
    # module docstring for why the macros are kept and the id is what gives way.
    if _fdc_id_is_free(db, usda.fdc_id, getattr(row, "food_id", None)):
        row.usda_fdc_id = usda.fdc_id
    else:
        logger.info(
            "USDA fdc_id %s already recorded on another cache row; storing "
            "%s/%s without provenance",
            usda.fdc_id,
            food101_class,
            cooking_method,
        )
        row.usda_fdc_id = None

    db.flush()
    return row
