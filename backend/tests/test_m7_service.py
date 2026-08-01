"""
tests/test_m7_service.py
===========================

WHY THIS FILE EXISTS
---------------------
Everything up to now was tested against hand-built `MealMacros` objects. This file
is the first to prove the engine works against **real rows in a real schema** —
that the SQL sums the right columns, that FR10's gate counts the right meals, that
the `06 §3` cache reuses rather than regenerates, and that one user can never see
another's insights.

The tests build meals through `seed_meals()`, which writes genuine `meal_log`,
`meal_item` and `food_item` rows. That matters: a fixture that stubbed the query
would pass while the real `GROUP BY` summed the wrong column, which is exactly the
class of bug this layer can have.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- The unlock gate (FR10): under 7 meals is `unlocked=False` with an accurate
  progress message and — critically — **no rows written**, so a locked user cannot
  accumulate a hidden history of insights. Unconfirmed drafts must not count
  toward the 7, or a user could unlock the feature without ever completing a log.

- The gate is not windowed: a user with 10 meals from a month ago stays unlocked.
  This is the design decision that costs the most to get wrong, because the
  failure mode is a user *losing* a feature they already earned.

- Empty window while unlocked: its own state, with a message and no fabricated
  profile. Tested because the natural bug here is to return a zeroed profile and
  generate confident advice from no data.

- The 6 h cache: a second call reuses the same rows (`from_cache`), a new meal
  invalidates it immediately, and `force=True` bypasses it. The "new meal
  invalidates" case is the one users would notice — logging a meal and seeing
  stale advice reads as the app ignoring them.

- Per-user isolation: the queries filter by `user_id`, and `mark_read` refuses
  another user's row. This is a security property, not a correctness one, so it is
  asserted on both the read and the write path.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.config import settings
from app.core.errors import ForbiddenError, NotFoundError
from app.db.models import (
    ActivityLevel,
    FoodItem,
    HealthGoal,
    MealItem,
    MealLog,
    MealType,
    Recommendation,
    Sex,
    User,
)
from app.services import recommendation_service as svc

NOW = datetime(2026, 7, 30, 12, 0, 0)


def make_user(
    db,
    *,
    email="reco@meu.edu.jo",
    username="reco1",
    goal=HealthGoal.weight_loss,
    with_metrics=False,
) -> User:
    user = User(
        username=username,
        email=email,
        password_hash="x" * 20,
        health_goal=goal,
    )
    if with_metrics:
        user.height_cm = 175.0
        user.weight_kg = 72.0
        user.birth_year = 2001
        user.sex = Sex.male
        user.activity_level = ActivityLevel.moderate
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def make_food(db, name="Grilled Chicken", protein=30.0, carbs=2.0, fat=5.0) -> FoodItem:
    food = FoodItem(
        food_name=name,
        food101_class=name.lower().replace(" ", "_"),
        calories_per_100g=protein * 4 + carbs * 4 + fat * 9,
        protein_per_100g=protein,
        carbs_per_100g=carbs,
        fat_per_100g=fat,
    )
    db.add(food)
    db.commit()
    db.refresh(food)
    return food


def seed_meals(
    db,
    user: User,
    *,
    count: int,
    food: FoodItem | None = None,
    protein=30.0,
    carbs=60.0,
    fat=20.0,
    confirmed=True,
    days_ago_start=0.0,
    day_step=1.0,
) -> list[MealLog]:
    """
    Write `count` real confirmed meals, one per day walking backwards from
    `days_ago_start`. Each gets one `meal_item` carrying the macros, and
    `total_calories` matching the macro-derived energy.
    """
    food = food or make_food(db)
    calories = protein * 4 + carbs * 4 + fat * 9
    logs = []
    for index in range(count):
        log = MealLog(
            user_id=user.user_id,
            logged_at=NOW - timedelta(days=days_ago_start + index * day_step),
            meal_type=MealType.lunch,
            total_calories=calories,
            user_confirmed=confirmed,
        )
        db.add(log)
        db.flush()
        db.add(
            MealItem(
                log_id=log.log_id,
                food_id=food.food_id,
                portion_confirmed_g=200.0,
                calories_est=calories,
                protein_est_g=protein,
                carbs_est_g=carbs,
                fat_est_g=fat,
            )
        )
        logs.append(log)
    db.commit()
    return logs


# ---------------------------------------------------------------------------
# The FR10 unlock gate
# ---------------------------------------------------------------------------


def test_locked_below_seven_meals_with_accurate_progress(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=4, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is False
    assert result.meals_logged == 4
    assert result.meals_required == 7
    assert "3 more meals" in result.message


def test_locked_state_writes_no_rows(db_session):
    """A locked user must not accumulate a hidden history of insights."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=3, day_step=0.5)

    svc.get_recommendations(user, db_session, now=NOW)

    assert db_session.query(Recommendation).count() == 0


def test_progress_message_is_singular_for_one_remaining_meal(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=6, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)
    assert "1 more meal to unlock" in result.message


def test_unconfirmed_meals_do_not_count_toward_the_gate(db_session):
    """
    FR10 counts *confirmed* meals. If drafts counted, a user could unlock insights
    without ever completing a log, and the engine would then be asked to describe
    meals the user never confirmed.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=4, day_step=0.5)
    seed_meals(db_session, user, count=5, confirmed=False, days_ago_start=0.1, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is False
    assert result.meals_logged == 4


def test_seven_meals_unlocks(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is True
    assert result.insights
    assert len(result.insights) <= 5


def test_seven_meals_on_the_same_day_still_unlocks(db_session):
    """06 §4: the threshold is 7 *meals*, the window is 7 *days* — both satisfied."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.0)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is True
    assert result.insights


def test_the_gate_is_not_windowed_so_users_never_re_lock(db_session):
    """
    The design decision with the worst failure mode: if the gate counted only the
    window, a user who logged 10 meals then took a week off would find the feature
    they earned taken away.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=10, days_ago_start=30.0, day_step=1.0)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is True
    assert result.meals_logged == 10


def test_unlocked_but_empty_window_says_so_without_inventing_a_profile(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=10, days_ago_start=30.0, day_step=1.0)

    result = svc.get_recommendations(user, db_session, now=NOW)

    assert result.unlocked is True
    assert result.insights == []
    assert "not logged any meals in the last 7 days" in result.message
    # Nothing to record: there is no nutritional finding here.
    assert db_session.query(Recommendation).count() == 0


# ---------------------------------------------------------------------------
# Aggregation against real rows
# ---------------------------------------------------------------------------


def test_window_query_sums_meal_item_macros(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, protein=40.0, carbs=50.0, fat=10.0, day_step=0.5)

    meals = svc.load_window_meals(user.user_id, db_session, days=7, now=NOW)

    assert len(meals) == 7
    assert all(meal.protein_g == pytest.approx(40.0) for meal in meals)
    assert all(meal.carbs_g == pytest.approx(50.0) for meal in meals)
    assert all(meal.fat_g == pytest.approx(10.0) for meal in meals)


def test_window_query_excludes_unconfirmed_and_out_of_window_meals(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=3, day_step=0.5)
    seed_meals(db_session, user, count=2, confirmed=False, days_ago_start=0.2, day_step=0.5)
    seed_meals(db_session, user, count=4, days_ago_start=20.0, day_step=1.0)

    meals = svc.load_window_meals(user.user_id, db_session, days=7, now=NOW)
    assert len(meals) == 3


def test_window_query_is_scoped_to_one_user(db_session):
    mine = make_user(db_session, email="a@x.com", username="a")
    theirs = make_user(db_session, email="b@x.com", username="b")
    seed_meals(db_session, mine, count=3, day_step=0.5)
    seed_meals(db_session, theirs, count=5, days_ago_start=0.1, day_step=0.5)

    assert len(svc.load_window_meals(mine.user_id, db_session, days=7, now=NOW)) == 3


def test_meal_with_no_nutrition_yields_none_not_zero(db_session):
    """
    `COALESCE` is deliberately absent from the sums: NULL means "unknown" and the
    engine drops the meal. Coalescing to 0.0 would assert the meal contained no
    protein, which is a different and false claim.
    """
    user = make_user(db_session)
    food = make_food(db_session)
    log = MealLog(
        user_id=user.user_id,
        logged_at=NOW,
        meal_type=MealType.snack,
        total_calories=300.0,
        user_confirmed=True,
    )
    db_session.add(log)
    db_session.flush()
    db_session.add(MealItem(log_id=log.log_id, food_id=food.food_id))
    db_session.commit()

    meals = svc.load_window_meals(user.user_id, db_session, days=7, now=NOW)

    assert len(meals) == 1
    assert meals[0].protein_g is None


def test_food_candidates_carry_a_per_user_log_count(db_session):
    user = make_user(db_session)
    food = make_food(db_session, name="Lentil Soup", protein=9.0, carbs=20.0, fat=1.0)
    seed_meals(db_session, user, count=4, food=food, day_step=0.5)

    candidates = svc.load_food_candidates(user.user_id, db_session)

    assert len(candidates) == 1
    assert candidates[0].food_name == "Lentil Soup"
    assert candidates[0].times_logged == 4
    assert candidates[0].is_scorable


def test_food_candidates_exclude_other_users_history(db_session):
    mine = make_user(db_session, email="a@x.com", username="a")
    theirs = make_user(db_session, email="b@x.com", username="b")
    seed_meals(db_session, mine, count=2, food=make_food(db_session, name="Mine"), day_step=0.5)
    seed_meals(db_session, theirs, count=2, food=make_food(db_session, name="Theirs"), day_step=0.5)

    names = {c.food_name for c in svc.load_food_candidates(mine.user_id, db_session)}
    assert names == {"Mine"}


# ---------------------------------------------------------------------------
# Persistence and the current batch
# ---------------------------------------------------------------------------


def test_insights_are_persisted_with_a_shared_generated_at(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)
    stamps = {row.generated_at for row in result.insights}

    assert len(stamps) == 1, "one batch must share one timestamp"
    assert result.generated_at == NOW


def test_persisted_rows_start_unread_and_carry_a_focus(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    result = svc.get_recommendations(user, db_session, now=NOW)

    for row in result.insights:
        assert row.was_read is False
        assert row.nutrient_focus
        assert row.insight_text


def test_current_batch_returns_only_the_newest_generation(db_session):
    """
    The reason batches share a timestamp: a 2-insight batch after a 5-insight one
    must not leave 3 stale rows visible, or the user reads old and new advice mixed.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    first = svc.get_recommendations(user, db_session, now=NOW)
    later = NOW + timedelta(hours=10)
    second = svc.get_recommendations(user, db_session, force=True, now=later)

    batch = svc.current_batch(user.user_id, db_session)
    assert {row.rec_id for row in batch} == {row.rec_id for row in second.insights}
    assert all(row.rec_id not in {r.rec_id for r in first.insights} for row in batch)


def test_history_is_kept_append_only(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    first = svc.get_recommendations(user, db_session, now=NOW)
    svc.get_recommendations(user, db_session, force=True, now=NOW + timedelta(hours=10))

    total = db_session.query(Recommendation).count()
    assert total > len(first.insights)


# ---------------------------------------------------------------------------
# The 06 §3 idempotency cache
# ---------------------------------------------------------------------------


def test_second_call_within_the_window_reuses_the_stored_rows(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    first = svc.get_recommendations(user, db_session, now=NOW)
    second = svc.get_recommendations(user, db_session, now=NOW + timedelta(hours=1))

    assert second.from_cache is True
    assert [r.rec_id for r in second.insights] == [r.rec_id for r in first.insights]
    assert db_session.query(Recommendation).count() == len(first.insights)


def test_cache_expires_after_the_configured_hours(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    first = svc.get_recommendations(user, db_session, now=NOW)
    stale = NOW + timedelta(hours=settings.recommendation_cache_hours, minutes=1)
    second = svc.get_recommendations(user, db_session, now=stale)

    assert second.from_cache is False
    assert {r.rec_id for r in second.insights}.isdisjoint({r.rec_id for r in first.insights})


def test_a_new_meal_invalidates_the_cache_immediately(db_session):
    """
    The case a user would notice: log a meal, open Insights, and see advice that
    accounts for it rather than a six-hour-old snapshot.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)
    svc.get_recommendations(user, db_session, now=NOW)

    later = NOW + timedelta(hours=1)
    seed_meals(db_session, user, count=1, days_ago_start=-1.0 / 24, day_step=0.0)

    second = svc.get_recommendations(user, db_session, now=later)
    assert second.from_cache is False


def test_force_bypasses_a_fresh_cache(db_session):
    """POST /recommendations/refresh must always regenerate."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    svc.get_recommendations(user, db_session, now=NOW)
    forced = svc.get_recommendations(user, db_session, force=True, now=NOW + timedelta(minutes=5))

    assert forced.from_cache is False


def test_cache_is_per_user(db_session):
    mine = make_user(db_session, email="a@x.com", username="a")
    theirs = make_user(db_session, email="b@x.com", username="b")
    seed_meals(db_session, mine, count=7, day_step=0.5)
    seed_meals(db_session, theirs, count=7, days_ago_start=0.1, day_step=0.5)

    svc.get_recommendations(mine, db_session, now=NOW)
    theirs_result = svc.get_recommendations(theirs, db_session, now=NOW)

    assert theirs_result.from_cache is False
    assert all(row.user_id == theirs.user_id for row in theirs_result.insights)


# ---------------------------------------------------------------------------
# Goal and body metrics flow through
# ---------------------------------------------------------------------------


def test_body_metrics_produce_a_calorie_verdict(db_session):
    user = make_user(db_session, with_metrics=True)
    seed_meals(db_session, user, count=7, day_step=1.0)

    result = svc.get_recommendations(user, db_session, now=NOW)
    calorie = [r for r in result.insights if r.nutrient_focus == "calories"]

    assert len(calorie) == 1
    assert "below" in calorie[0].insight_text or "above" in calorie[0].insight_text


def test_missing_metrics_are_named_in_the_calorie_note(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=1.0)

    result = svc.get_recommendations(user, db_session, now=NOW)
    calorie = next(r for r in result.insights if r.nutrient_focus == "calories")

    assert "height" in calorie.insight_text
    assert "activity level" in calorie.insight_text


def test_health_goal_changes_the_insights(db_session):
    """FR13: changing the goal is reflected on the next refresh."""
    user = make_user(db_session, goal=HealthGoal.weight_loss)
    seed_meals(db_session, user, count=7, protein=20.0, carbs=90.0, fat=25.0, day_step=1.0)

    loss = svc.get_recommendations(user, db_session, now=NOW)
    loss_text = " ".join(r.insight_text for r in loss.insights)

    user.health_goal = HealthGoal.weight_gain
    db_session.commit()
    gain = svc.get_recommendations(user, db_session, force=True, now=NOW + timedelta(hours=7))
    gain_text = " ".join(r.insight_text for r in gain.insights)

    assert "weight loss" in loss_text
    assert "weight gain" in gain_text


def test_every_result_carries_the_disclaimer(db_session):
    user = make_user(db_session)
    locked = svc.get_recommendations(user, db_session, now=NOW)
    seed_meals(db_session, user, count=7, day_step=0.5)
    unlocked = svc.get_recommendations(user, db_session, now=NOW)

    for result in (locked, unlocked):
        assert "informational purposes only" in result.disclaimer
        assert "qualified healthcare professional" in result.disclaimer


# ---------------------------------------------------------------------------
# mark_read
# ---------------------------------------------------------------------------


def test_mark_read_sets_the_flag(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)
    result = svc.get_recommendations(user, db_session, now=NOW)
    target = result.insights[0]

    updated = svc.mark_read(user.user_id, target.rec_id, db_session)
    assert updated.was_read is True


def test_mark_read_is_idempotent(db_session):
    """The client may retry after a dropped response; a second call must not error."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)
    target = svc.get_recommendations(user, db_session, now=NOW).insights[0]

    svc.mark_read(user.user_id, target.rec_id, db_session)
    assert svc.mark_read(user.user_id, target.rec_id, db_session).was_read is True


def test_mark_read_rejects_a_missing_id(db_session):
    user = make_user(db_session)
    with pytest.raises(NotFoundError):
        svc.mark_read(user.user_id, 9999, db_session)


def test_mark_read_rejects_another_users_insight(db_session):
    """Per-user isolation on the write path — a security property, not a nicety."""
    mine = make_user(db_session, email="a@x.com", username="a")
    theirs = make_user(db_session, email="b@x.com", username="b")
    seed_meals(db_session, theirs, count=7, day_step=0.5)
    their_insight = svc.get_recommendations(theirs, db_session, now=NOW).insights[0]

    with pytest.raises(ForbiddenError):
        svc.mark_read(mine.user_id, their_insight.rec_id, db_session)

    db_session.refresh(their_insight)
    assert their_insight.was_read is False


# ---------------------------------------------------------------------------
# The M5 hook
# ---------------------------------------------------------------------------


def test_on_meal_logged_returns_none_while_locked(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=3, day_step=0.5)

    assert svc.on_meal_logged(user, db_session) is None
    assert db_session.query(Recommendation).count() == 0


def test_on_meal_logged_generates_when_the_threshold_is_crossed(db_session):
    """The meal that takes the user from 6 to 7 must trigger the first batch."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=6, day_step=0.5)
    assert svc.on_meal_logged(user, db_session) is None

    seed_meals(db_session, user, count=1, days_ago_start=0.1, day_step=0.0)
    result = svc.on_meal_logged(user, db_session)

    assert result is not None
    assert result.unlocked is True
    assert result.insights


def test_on_meal_logged_refreshes_an_already_unlocked_user(db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)
    first = svc.get_recommendations(user, db_session, now=NOW)

    result = svc.on_meal_logged(user, db_session)

    assert result is not None
    assert result.from_cache is False
    assert {r.rec_id for r in result.insights}.isdisjoint({r.rec_id for r in first.insights})


def test_on_meal_logged_swallows_a_generation_failure(db_session, monkeypatch):
    """
    A failure to refresh insights must not fail the meal log that triggered it —
    the meal is the user's data, the insights are merely derived from it. Forced by
    making generation raise, since no realistic input can.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    def explode(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(svc, "get_recommendations", explode)

    assert svc.on_meal_logged(user, db_session) is None


def test_on_meal_logged_leaves_the_session_usable_after_a_failure(db_session, monkeypatch):
    """
    The rollback in the handler matters: M5 commits the meal *before* calling the
    hook, so a poisoned session here would break whatever the route does next.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, day_step=0.5)

    monkeypatch.setattr(
        svc, "get_recommendations", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    svc.on_meal_logged(user, db_session)
    monkeypatch.undo()

    # The session still works, and the meals committed before the failure survive.
    assert svc.count_confirmed_meals(user.user_id, db_session) == 7


def test_now_defaults_to_the_real_clock(db_session):
    """The `now` parameter is a test seam; production passes nothing."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=7, days_ago_start=0.0, day_step=0.2)

    result = svc.get_recommendations(user, db_session)

    assert result.unlocked is True
    assert result.generated_at is not None
    assert abs((datetime.now(UTC).replace(tzinfo=None) - result.generated_at).total_seconds()) < 60
