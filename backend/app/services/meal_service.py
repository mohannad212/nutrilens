"""
app/services/meal_service.py
===============================

WHY THIS FILE EXISTS
---------------------
FR10 — persisting a confirmed meal — and with it the moment the core loop closes.
Everything before this point is ephemeral: the photo is discarded (`02`'s privacy
invariant), the predictions live in one response, the portion is a slider
position. This is where a user's meal becomes a row that history (FR11) reads and
the recommendation engine (FR12) draws on.

WHAT PROBLEM IT SOLVES
------------------------
The client knows what the user confirmed; the server decides what is true. Those
have to be kept apart, and this module is the seam. A `POST /meals` body arrives
carrying food classes, cooking methods and portions — all legitimate user input —
and *no* nutrition, because nutrition is not the client's to assert. The server
looks every macro up itself (`02 §6`), computes each item's contribution, sums the
meal, and writes the result.

WHY THE CLIENT IS NEVER TRUSTED WITH THE NUMBERS
---------------------------------------------------
`03 §6` says it outright — "Server recomputes `total_calories` (never trusts
client)" — and `02 §6` makes it a data-integrity rule. Three reasons, in
increasing order of seriousness:

1. **Drift.** A client that computes locally will eventually round differently, or
   run an old version, and a user's history becomes internally inconsistent with
   no way to tell which rows are wrong.
2. **Downstream corruption.** FR12's insights are averages over `meal_item`
   macros. A client-supplied figure propagates from one meal into every insight
   the user ever sees, and the engine has no way to detect it.
3. **It is trivially forgeable.** A meal log is a health record the user acts on;
   the API accepting "this pizza was 40 kcal" would make the feature a fiction.
   Not adversarially interesting for a single-user account, but the principle —
   derived values are the server's — is what keeps the data meaningful.

So the request schema simply has nowhere to put a calorie figure. The safest way
to not trust an input is to not accept it.

HOW IT WORKS INTERNALLY — ONE TRANSACTION
--------------------------------------------
`create_meal` runs as a single unit of work:

1. Resolve each item's `food_item` through `nutrition_service.resolve_food_item`,
   which is cache-first and may call USDA (`08 §6`). Reusing that function rather
   than reimplementing the lookup is what guarantees `POST /nutrition` and
   `POST /meals` can never disagree about a food's macros — a discrepancy the user
   would see directly, having just read the numbers on the previous screen.
2. Scale each item to its confirmed portion and write a `meal_item` row.
3. Sum the items into `meal_log.total_calories`.
4. Commit once, at the end.

The commit placement is the important part. If item three's lookup 502s, items one
and two must not be left behind as a phantom meal, and the `meal_log` header must
not exist without its items. One commit means the meal either exists completely or
does not exist at all.

WHY `user_confirmed` IS SET HERE AND ONLY HERE
-------------------------------------------------
`02 §2.2` gives `meal_log.user_confirmed` a default of `FALSE`, and `02 §6` defines
an unconfirmed meal as a draft that is excluded from history and from the
recommendation window. NutriLens never writes a draft — a meal row is only created
by this endpoint, which by definition is the user pressing "Log This Meal"
(`03 §6`: "Persist a **confirmed** meal"). The column is therefore set `TRUE` on
insert. It exists for the schema to express the distinction and for future
partial-save flows, not because a `FALSE` row is ever written today.

WHY A FUTURE `logged_at` IS REFUSED
--------------------------------------
`09` M7 asks for this explicitly: "M5 should also reject or ignore a
client-supplied future `logged_at`". The engine already filters future timestamps
out of its 7-day window, but a meal dated next month would sit in history forever,
appear at the top of a newest-first list, and re-enter the window when its date
arrives. `03 §6`'s request body has no `logged_at` field at all, which is the
strongest form of the fix — the server clock is the only source. `create_meal`
accepts an optional `logged_at` for **tests and the demo seed script**, and
clamps a future value to now rather than raising, because the caller is trusted
code and silently drifting data is the failure being prevented.

WHY THE RECOMMENDATION HOOK CANNOT FAIL THE MEAL
---------------------------------------------------
`09` M7 built `recommendation_service.on_meal_logged` and left the call site to
M5. It is invoked **after** the commit, and it never raises (the service wraps
itself). The ordering is deliberate: insights are *derived* from the meal, so a
failure to regenerate them must not discard the meal that triggered it. The user's
data is the thing of value; the insights can be rebuilt on the next refresh.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "What if the user's confirmed portion is outside the estimated range?"
A: It is accepted and logged as a warning, per `02 §6` — deliberately *not* a hard
   constraint. FR06 makes the user the authority on their own portion; the range
   is the model's estimate, and a user who knows they ate 400 g of a dish the
   model bounded at 250 g is correcting us, not making an error.

Q: "Why does the response include `meals_logged_total` and
    `recommendations_unlocked`?"
A: `03 §6` specifies both, and they save a round trip at the one moment they
   change. The app has just logged a meal and is navigating Home; without them it
   would have to call `/recommendations` to discover whether the seventh meal just
   unlocked the feature.

Q: "Two devices log the same meal at once — what happens?"
A: Two independent meals, which is correct: there is no idempotency key in
   `03 §6` because a user genuinely may eat two identical meals. Deduplication
   would have to guess at intent, and guessing wrong deletes real data.

COMMON STUDENT MISTAKES
--------------------------
- Accepting `total_calories` from the request body "to save a lookup".
- Committing per item, so a mid-meal failure leaves a partial meal in history.
- Recomputing the total from the response objects instead of from the persisted
  `meal_item` rows, so a rounding difference between the two goes unnoticed.
- Calling the recommendation hook *inside* the transaction, where a slow or
  failing refresh takes the meal down with it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import ForbiddenError, NotFoundError
from app.db.models import MealItem, MealLog, MealType, User
from app.integrations.usda_client import UsdaClient
from app.services import nutrition_service, recommendation_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MealItemInput:
    """One confirmed item from `03 §6`'s request body.

    Everything here is the *user's* assertion about their meal. Note what is
    absent: no calories, no macros, no food name. Those are derived, and derived
    values are the server's (`02 §6`).
    """

    food101_class: str
    cooking_method: str
    portion_confirmed_g: float
    portion_min_g: float | None = None
    portion_max_g: float | None = None
    confidence_score: float | None = None


def _now() -> datetime:
    """Naive UTC, matching the `TIMESTAMP` columns in `02`."""
    return datetime.now(UTC).replace(tzinfo=None)


def count_confirmed_meals(user_id: int, db: Session) -> int:
    """All confirmed meals ever logged (`03 §7`'s `meals_logged`).

    Counts the user's whole history, not the recommendation window: unlocking is a
    milestone, so someone who logs ten meals and then takes a week off must not
    find the feature re-locked.
    """
    return (
        db.execute(
            select(func.count())
            .select_from(MealLog)
            .where(MealLog.user_id == user_id, MealLog.user_confirmed.is_(True))
        ).scalar_one()
        or 0
    )


def _warn_if_portion_outside_range(item: MealItemInput) -> None:
    """`02 §6`: log, do not reject. FR06 makes the user the authority."""
    low, high, confirmed = item.portion_min_g, item.portion_max_g, item.portion_confirmed_g
    if (low is not None and confirmed < low) or (high is not None and confirmed > high):
        logger.info(
            "Confirmed portion %.1f g for %s is outside the estimated range [%s, %s]",
            confirmed,
            item.food101_class,
            low,
            high,
        )


def create_meal(
    user: User,
    meal_type: MealType,
    items: list[MealItemInput],
    db: Session,
    usda: UsdaClient,
    *,
    logged_at: datetime | None = None,
) -> MealLog:
    """FR10 — persist a confirmed meal, with every derived value computed here.

    One transaction: either the whole meal lands or none of it does. Raises the
    documented USDA errors (`03 §5`) if a food cannot be resolved and is not
    cached, in which case nothing is written.

    `logged_at` is for trusted callers only (tests, the demo seed); a future value
    is clamped to now — see the module docstring.
    """
    stamp = _now()
    if logged_at is not None:
        stamp = min(logged_at, stamp)

    meal = MealLog(
        user_id=user.user_id,
        logged_at=stamp,
        meal_type=meal_type,
        # 03 §6: this endpoint exists only for confirmed meals; NutriLens never
        # writes a draft (02 §6).
        user_confirmed=True,
        total_calories=0,
    )

    # The rollback is explicit rather than left to the session's teardown. Closing
    # a session does discard an uncommitted transaction, so in a live request the
    # outcome is the same either way — but that makes "the meal is written whole or
    # not at all" a property of the request lifecycle instead of a property of this
    # function. Any other caller reusing the session (the demo seed script, a test,
    # a future batch import) would otherwise inherit a half-built meal still
    # pending in the identity map.
    try:
        db.add(meal)
        db.flush()  # assign log_id before the items reference it

        total_calories = 0.0
        for item in items:
            _warn_if_portion_outside_range(item)

            # Cache-first, exactly as POST /nutrition — the same function, so the
            # two endpoints cannot report different macros for the same food.
            food, _source = nutrition_service.resolve_food_item(
                db, item.food101_class, item.cooking_method, usda
            )
            _, scaled = nutrition_service.scale_to_portion(food, item.portion_confirmed_g)

            db.add(
                MealItem(
                    log_id=meal.log_id,
                    food_id=food.food_id,
                    portion_min_g=item.portion_min_g,
                    portion_max_g=item.portion_max_g,
                    portion_confirmed_g=item.portion_confirmed_g,
                    confidence_score=item.confidence_score,
                    calories_est=scaled.calories,
                    protein_est_g=scaled.protein_g,
                    carbs_est_g=scaled.carbs_g,
                    fat_est_g=scaled.fat_g,
                )
            )
            total_calories += scaled.calories

        # 02 §6: always recomputed as the sum of the items, never from the client.
        meal.total_calories = total_calories

        db.commit()
    except Exception:
        # Includes the documented USDA failures, which the caller re-raises to the
        # shared error handler. Nothing of this meal survives.
        db.rollback()
        raise

    db.refresh(meal)
    return meal


def list_meals(
    user_id: int,
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> tuple[list[MealLog], int]:
    """FR11 — the user's meal history, newest first (`03 §6`).

    Returns `(page, total)` where `total` is the count matching the *filters*, not
    the length of the page — the client needs it to know whether more pages exist,
    so counting the page would make pagination undisplayable.

    Confirmed meals only (`02 §6`): an unconfirmed row is a draft and must not
    appear in history. NutriLens never writes one today, but the filter is the
    schema's stated contract and history is exactly where a stray draft would
    surface.
    """
    filters = [MealLog.user_id == user_id, MealLog.user_confirmed.is_(True)]
    if date_from is not None:
        filters.append(MealLog.logged_at >= date_from)
    if date_to is not None:
        filters.append(MealLog.logged_at <= date_to)

    total = db.execute(select(func.count()).select_from(MealLog).where(*filters)).scalar_one() or 0

    rows = (
        db.execute(
            select(MealLog)
            # `log_id` breaks ties: two meals logged in the same clock tick would
            # otherwise come back in an unstable order, which makes pagination drop
            # or repeat rows between pages.
            .where(*filters)
            .order_by(MealLog.logged_at.desc(), MealLog.log_id.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return list(rows), total


def get_meal(user_id: int, log_id: int, db: Session) -> MealLog:
    """One meal, ownership-checked (`03 §6`).

    Raises `NotFoundError` when no such meal exists and `ForbiddenError` when it
    exists but belongs to someone else — the split `03 §6` and `03 §1` specify.

    Worth naming the trade-off, because it is a real one: answering 403 rather
    than 404 for another user's meal confirms that `log_id` exists, so an
    authenticated user can probe the id space to learn how many meals the system
    holds. That is a small disclosure (ids are sequential, so the count is
    inferable anyway) and the contract asks for the distinction, which is more
    useful to a client deciding whether to retry or to refresh its list. A system
    where that leak mattered would return 404 for both.
    """
    meal = db.get(MealLog, log_id)
    if meal is None:
        raise NotFoundError("No meal with that id.")
    if meal.user_id != user_id:
        raise ForbiddenError("This meal belongs to another user.")
    return meal


def delete_meal(user_id: int, log_id: int, db: Session) -> None:
    """Delete one meal and its items (FR11, `03 §6`).

    Ownership is enforced by reusing `get_meal`, so the two endpoints cannot
    disagree about who may touch a row.

    The `meal_item` children go with it: `02 §1` puts `ON DELETE CASCADE` on the
    owned chain, and the ORM relationship carries `delete-orphan` to match. The
    `food_item` rows they referenced are deliberately untouched — that is shared
    reference data behind an `ON DELETE RESTRICT` foreign key, and another user's
    history may point at the same row.
    """
    meal = get_meal(user_id, log_id, db)
    db.delete(meal)
    db.commit()


def log_meal(
    user: User,
    meal_type: MealType,
    items: list[MealItemInput],
    db: Session,
    usda: UsdaClient,
    *,
    logged_at: datetime | None = None,
) -> tuple[MealLog, int, bool]:
    """`POST /meals` end to end: persist, then refresh insights.

    Returns `(meal, meals_logged_total, recommendations_unlocked)` — the three
    things `03 §6`'s response needs.

    The recommendation refresh happens **after** the commit and cannot fail the
    meal: `on_meal_logged` swallows its own errors by contract (`09` M7), and this
    call site is the integration point that milestone deferred to M5.
    """
    meal = create_meal(user, meal_type, items, db, usda, logged_at=logged_at)

    total = count_confirmed_meals(user.user_id, db)
    unlocked = total >= settings.min_meals_for_recommendation

    recommendation_service.on_meal_logged(user, db)

    return meal, total, unlocked
