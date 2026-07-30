"""
app/services/recommendation_service.py
=========================================

WHY THIS FILE EXISTS
---------------------
`ml/recommender.py` is deliberately pure — it knows how to turn a list of meals
into insights and nothing else. This file is everything that makes that usable
against a real user: reading their confirmed meals out of Postgres, enforcing the
FR10 unlock gate, deciding whether stored insights are still fresh enough to
reuse, writing new ones, and marking them read. It is the only module in M7 that
touches a `Session`.

WHAT PROBLEM IT SOLVES
------------------------
Three concerns that look like one:

1. **Am I allowed to say anything yet?** FR10 gates insights behind >= 7 confirmed
   meals. That is a policy decision, not arithmetic, and it belongs at the
   boundary where the user's history is visible.
2. **What are the numbers?** One aggregate query builds the `MealMacros` list the
   engine consumes. Doing this in SQL rather than loading every `meal_item` row
   keeps the request cheap as history grows.
3. **Should I regenerate?** `06 §3` requires idempotency: insights generated less
   than `recommendation_cache_hours` ago, with no meal logged since, must be
   returned as-is rather than recomputed.

Keeping these here means `recommender.py` stays testable without a database and
the API layer stays a thin translation to JSON.

HOW IT WORKS INTERNALLY
--------------------------
`get_recommendations()` is the single entry point, and runs in this order:

    count confirmed meals  ->  locked?  ->  cache still valid?  ->  aggregate
    ->  generate  ->  persist  ->  return

The window query sums each meal's `meal_item` nutrition into one row per meal
(`GROUP BY log_id`), filtered to `user_confirmed = TRUE` and the rolling window.
The engine re-applies its own filters on the result, which is intentional
belt-and-braces: the SQL narrows for speed, the pure function guarantees
correctness.

Persistence is **append-only**. Each generation writes its rows with one shared
`generated_at` timestamp, and "the user's current insights" means the rows whose
`generated_at` equals the newest value. Batching the timestamp is what makes that
query exact — without it, a batch of 2 insights following a batch of 5 would leave
3 stale rows in any "5 newest rows" query, and the user would see a mix of old and
new advice.

FOUR DESIGN DECISIONS WORTH DEFENDING
----------------------------------------
1. **The unlock gate counts ALL confirmed meals, not just those in the window.**
   Unlocking is a milestone, not a rolling status. If the gate counted only the
   last 7 days, a user who logged 10 meals and then took a week off would find the
   Insights screen *re-locked*, with a progress bar telling them to start over.
   Losing a feature you already earned reads as a bug and punishes exactly the
   behaviour the app wants to encourage.

2. **Unlocked with an empty window is its own state, not a locked one.** The cost
   of decision 1 is that "unlocked" no longer implies "has recent data". Rather
   than fake a profile or reuse a stale one, the service returns a single
   `no_recent_meals` insight. It is honest about having nothing to say, and it
   does not persist — there is no nutritional finding to record.

3. **The cache checks "any meal since", not a row count.** The condition in
   `06 §3` is "no new meals". Comparing the newest confirmed `logged_at` against
   the last `generated_at` catches edits and deletions too, where a count
   comparison would miss a delete-then-add that leaves the total unchanged.

4. **`on_meal_logged()` exists before its caller does.** `09` puts the
   auto-trigger from `POST /meals` in M7's scope, but `/meals` is M5. Exposing the
   hook now means the wiring is a single call site later, and its behaviour is
   already tested, rather than M5 having to rediscover the contract.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why compute the macro sums in SQL instead of in Python?"
A: A user with a year of history has thousands of `meal_item` rows. Summing in the
   database transfers one row per meal instead of one per item, and the
   aggregation runs where the data already is. The window filter uses the
   `idx_meal_log_user_time` index from `02`.

Q: "What stops two simultaneous requests generating two batches of insights?"
A: Nothing at this scope, and the consequence is benign: the second batch gets a
   later `generated_at` and simply becomes the current one, so the user sees one
   coherent set either way. A unique constraint or advisory lock would be the fix
   if it ever mattered; for a single-user mobile client it does not.

Q: "Why not delete the previous insights on refresh?"
A: Append-only keeps a trail the report can point at ("here is how this user's
   insights evolved"), and it avoids destroying `was_read` state. The cost is one
   extra subquery to find the current batch, which the user index already covers.

COMMON STUDENT MISTAKES
--------------------------
- Filtering `user_confirmed` in Python after loading everything. FR10 counts
  *confirmed* meals; an unconfirmed draft must never reach the engine, and the
  cheapest place to guarantee that is the WHERE clause.
- Letting `generated_at` default per row (`server_default=now()`). Rows in one
  batch then differ by microseconds and "the current batch" stops being a single
  equality test.
- Calling `db.commit()` inside a helper that the route also commits. Commit once,
  at the end of the operation that owns the transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.constants import RECOMMENDATION_DISCLAIMER
from app.core.errors import ForbiddenError, NotFoundError
from app.db.models import FoodItem, MealItem, MealLog, Recommendation, User
from app.ml import reco_targets as rt
from app.ml import recommender as rec


@dataclass
class RecommendationResult:
    """
    What the API layer needs to build either `03 §7` response shape.

    `unlocked=False` carries the progress fields and an empty `insights`; the
    unlocked case carries the rows and their shared `generated_at`. The disclaimer
    is attached unconditionally (`06 §5`) rather than being the route's job to
    remember.
    """

    unlocked: bool
    meals_logged: int = 0
    meals_required: int = 0
    message: str | None = None
    insights: list[Recommendation] = field(default_factory=list)
    generated_at: datetime | None = None
    disclaimer: str = RECOMMENDATION_DISCLAIMER
    # True when the rows were reused rather than regenerated (06 §3). Not part of
    # the API contract; useful in tests and logs to prove the cache is working.
    from_cache: bool = False


def _now() -> datetime:
    """Naive UTC, matching the `TIMESTAMP` columns in `02`."""
    return datetime.now(UTC).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def count_confirmed_meals(user_id: int, db: Session) -> int:
    """
    All confirmed meals ever logged by this user — the FR10 unlock counter.

    Deliberately not windowed: see decision 1 in the module docstring. A user does
    not lose access to insights by taking a week off.
    """
    return (
        db.execute(
            select(func.count())
            .select_from(MealLog)
            .where(MealLog.user_id == user_id, MealLog.user_confirmed.is_(True))
        ).scalar_one()
        or 0
    )


def load_window_meals(
    user_id: int, db: Session, *, days: int, now: datetime
) -> list[rec.MealMacros]:
    """
    One `MealMacros` per confirmed meal in the rolling window, with each meal's
    nutrition summed from its `meal_item` rows in SQL.

    `COALESCE` is deliberately absent on the macro sums: a meal whose items carry
    no nutrition yields NULL, which `MealMacros` treats as "unknown" and the
    engine's filter drops. Coalescing to 0.0 would instead assert the meal
    contained no protein at all, which is a different and false claim.
    """
    cutoff = now - timedelta(days=days)
    rows = db.execute(
        select(
            MealLog.logged_at,
            MealLog.total_calories,
            func.sum(MealItem.protein_est_g).label("protein_g"),
            func.sum(MealItem.carbs_est_g).label("carbs_g"),
            func.sum(MealItem.fat_est_g).label("fat_g"),
        )
        .join(MealItem, MealItem.log_id == MealLog.log_id)
        .where(
            MealLog.user_id == user_id,
            MealLog.user_confirmed.is_(True),
            MealLog.logged_at >= cutoff,
            MealLog.logged_at <= now,
        )
        .group_by(MealLog.log_id, MealLog.logged_at, MealLog.total_calories)
    ).all()

    return [
        rec.MealMacros(
            logged_at=row.logged_at,
            calories=float(row.total_calories) if row.total_calories is not None else None,
            protein_g=float(row.protein_g) if row.protein_g is not None else None,
            carbs_g=float(row.carbs_g) if row.carbs_g is not None else None,
            fat_g=float(row.fat_g) if row.fat_g is not None else None,
        )
        for row in rows
    ]


def load_food_candidates(user_id: int, db: Session) -> list[rec.FoodCandidate]:
    """
    The `food_item` rows this user's meals have touched, with how often — the
    candidate pool for Stage 3's content-based scoring.

    Scoped to foods the user has actually eaten rather than the whole cache: the
    scoring function's `already_frequent` term needs a per-user count anyway, and
    suggesting a food from someone else's history would be a recommendation drawn
    from data this user never generated.
    """
    rows = db.execute(
        select(
            FoodItem.food_name,
            FoodItem.calories_per_100g,
            FoodItem.protein_per_100g,
            FoodItem.carbs_per_100g,
            FoodItem.fat_per_100g,
            func.count(MealItem.item_id).label("times_logged"),
        )
        .join(MealItem, MealItem.food_id == FoodItem.food_id)
        .join(MealLog, MealLog.log_id == MealItem.log_id)
        .where(MealLog.user_id == user_id, MealLog.user_confirmed.is_(True))
        .group_by(
            FoodItem.food_id,
            FoodItem.food_name,
            FoodItem.calories_per_100g,
            FoodItem.protein_per_100g,
            FoodItem.carbs_per_100g,
            FoodItem.fat_per_100g,
        )
    ).all()

    return [
        rec.FoodCandidate(
            food_name=row.food_name,
            calories_per_100g=(
                float(row.calories_per_100g) if row.calories_per_100g is not None else None
            ),
            protein_per_100g=(
                float(row.protein_per_100g) if row.protein_per_100g is not None else None
            ),
            carbs_per_100g=float(row.carbs_per_100g) if row.carbs_per_100g is not None else None,
            fat_per_100g=float(row.fat_per_100g) if row.fat_per_100g is not None else None,
            times_logged=int(row.times_logged or 0),
        )
        for row in rows
    ]


def latest_generated_at(user_id: int, db: Session) -> datetime | None:
    """When this user's most recent batch of insights was written, if ever."""
    return db.execute(
        select(func.max(Recommendation.generated_at)).where(Recommendation.user_id == user_id)
    ).scalar_one_or_none()


def current_batch(user_id: int, db: Session) -> list[Recommendation]:
    """
    The rows of the newest generation, oldest-first within the batch so the order
    the engine chose (most actionable first) survives the round trip.
    """
    generated_at = latest_generated_at(user_id, db)
    if generated_at is None:
        return []
    return list(
        db.execute(
            select(Recommendation)
            .where(
                Recommendation.user_id == user_id,
                Recommendation.generated_at == generated_at,
            )
            .order_by(Recommendation.rec_id)
        )
        .scalars()
        .all()
    )


def _newest_confirmed_meal_at(user_id: int, db: Session) -> datetime | None:
    return db.execute(
        select(func.max(MealLog.logged_at)).where(
            MealLog.user_id == user_id, MealLog.user_confirmed.is_(True)
        )
    ).scalar_one_or_none()


def _cache_is_fresh(user_id: int, db: Session, *, now: datetime) -> bool:
    """
    `06 §3`: reuse stored insights when they are younger than
    `recommendation_cache_hours` AND no confirmed meal has been logged since.

    The second condition is what makes this correct rather than merely cheap — a
    user who logs a meal and immediately opens the Insights screen must see advice
    that accounts for it, not a six-hour-old snapshot.
    """
    generated_at = latest_generated_at(user_id, db)
    if generated_at is None:
        return False
    if now - generated_at >= timedelta(hours=settings.recommendation_cache_hours):
        return False
    newest_meal = _newest_confirmed_meal_at(user_id, db)
    return newest_meal is None or newest_meal <= generated_at


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _persist(
    user_id: int, insights: list[rec.Insight], db: Session, *, generated_at: datetime
) -> list[Recommendation]:
    """
    Write one batch, sharing a single `generated_at` so `current_batch` can
    identify it with an equality test (see the module docstring).
    """
    rows = [
        Recommendation(
            user_id=user_id,
            generated_at=generated_at,
            insight_text=insight.insight_text,
            nutrient_focus=insight.nutrient_focus,
            was_read=False,
        )
        for insight in insights
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def mark_read(user_id: int, rec_id: int, db: Session) -> Recommendation:
    """
    PATCH `/recommendations/{rec_id}/read`.

    Raises `NotFoundError` when no such row exists and `ForbiddenError` when it
    belongs to another user — the two distinct codes `03 §1` specifies. Idempotent:
    marking an already-read insight read again succeeds rather than erroring, since
    the client may retry after a dropped response.
    """
    row = db.get(Recommendation, rec_id)
    if row is None:
        raise NotFoundError("No insight with that id exists.")
    if row.user_id != user_id:
        raise ForbiddenError("This insight belongs to another user.")
    if not row.was_read:
        row.was_read = True
        db.commit()
        db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# Orchestration — the single entry point
# ---------------------------------------------------------------------------


def _locked_result(count: int) -> RecommendationResult:
    required = settings.min_meals_for_recommendation
    remaining = max(required - count, 0)
    return RecommendationResult(
        unlocked=False,
        meals_logged=count,
        meals_required=required,
        message=(
            f"Log {remaining} more meal{'s' if remaining != 1 else ''} to unlock "
            "personalised insights."
        ),
    )


def get_recommendations(
    user: User, db: Session, *, force: bool = False, now: datetime | None = None
) -> RecommendationResult:
    """
    The `06 §3` entry point, used by both `GET /recommendations` (force=False) and
    `POST /recommendations/refresh` (force=True).

    Takes the `User` object rather than an id because generation needs their health
    goal and body metrics, and the route already has the authenticated user loaded.
    """
    now = now or _now()
    count = count_confirmed_meals(user.user_id, db)
    if count < settings.min_meals_for_recommendation:
        return _locked_result(count)

    if not force and _cache_is_fresh(user.user_id, db, now=now):
        rows = current_batch(user.user_id, db)
        if rows:
            return RecommendationResult(
                unlocked=True,
                meals_logged=count,
                meals_required=settings.min_meals_for_recommendation,
                insights=rows,
                generated_at=rows[0].generated_at,
                from_cache=True,
            )

    meals = load_window_meals(user.user_id, db, days=settings.recommendation_window_days, now=now)
    profile = rec.aggregate_profile(meals, window_days=settings.recommendation_window_days, now=now)

    if profile is None:
        # Unlocked, but nothing logged inside the window. Say so plainly and write
        # nothing — there is no nutritional finding to record (decision 2).
        return RecommendationResult(
            unlocked=True,
            meals_logged=count,
            meals_required=settings.min_meals_for_recommendation,
            message=rt.INSIGHT_TEMPLATES["no_recent_meals"].format(
                days=f"{settings.recommendation_window_days} days"
            ),
        )

    tdee = rt.estimate_tdee(
        weight_kg=float(user.weight_kg) if user.weight_kg is not None else None,
        height_cm=float(user.height_cm) if user.height_cm is not None else None,
        birth_year=user.birth_year,
        sex=user.sex,
        activity_level=user.activity_level,
    )
    insights = rec.generate_insights(
        profile,
        goal=user.health_goal,
        calorie_target=rt.goal_target_calories(tdee, user.health_goal),
        candidates=load_food_candidates(user.user_id, db),
        missing_metrics=rt.missing_metric_labels(
            height_cm=user.height_cm,
            weight_kg=user.weight_kg,
            birth_year=user.birth_year,
            sex=user.sex,
            activity_level=user.activity_level,
        ),
    )

    generated_at = now
    rows = _persist(user.user_id, insights, db, generated_at=generated_at)
    return RecommendationResult(
        unlocked=True,
        meals_logged=count,
        meals_required=settings.min_meals_for_recommendation,
        insights=rows,
        generated_at=generated_at,
    )


def on_meal_logged(user: User, db: Session) -> RecommendationResult | None:
    """
    Hook for `POST /meals` (`06 §3`, `09` M7): regenerate after a meal is logged,
    including the one that crosses the 7-meal threshold.

    Returns `None` while the user is still locked, so the caller can cheaply tell
    whether anything was produced. **Never raises** — a failure to refresh
    insights must not fail the meal log that triggered it, because the meal is the
    user's data and the insights are derived from it. M5 calls this after its own
    commit.
    """
    try:
        result = get_recommendations(user, db, force=True)
    except Exception:
        db.rollback()
        return None
    return result if result.unlocked else None
