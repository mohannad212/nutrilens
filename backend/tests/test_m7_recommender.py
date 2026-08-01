"""
tests/test_m7_recommender.py
===============================

WHY THIS FILE EXISTS
---------------------
This is the evaluation `06 §6` prescribes. A recommender with no ground-truth
"correct answer" cannot be scored with accuracy or MAPE, so the standard for this
class of system is *sanity and consistency testing*: construct synthetic profiles
whose dominant gap is known by construction, and assert the engine identifies it.
These tests are the evidence the results chapter cites, which is why they are
written against known-by-construction inputs rather than realistic-looking ones.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- Usability filter (`06 §4`): meals with null or zero calories must be dropped.
  If they are not, a user with three real meals and four placeholder rows gets a
  daily average roughly half of what they ate, and the engine tells them to eat
  more. The "calories but no macros" case is tested separately because it is the
  subtler bug: such a meal would add to the energy total while contributing zero
  grams, deflating every macro percentage at once.

- Window boundary: a meal one hour inside the window counts and one an hour
  outside does not. Tested with an injected `now` because "the last 7 days" is a
  classic off-by-one, and a silent boundary error changes insights for anyone who
  logs sporadically.

- Percentages summing to 100: the design decision documented in
  `recommender.py`. A profile built from macros whose implied energy disagrees
  with the reported calorie figure must still produce percentages summing to
  exactly 100, because the targets do. Without this, every macro reads as low
  simultaneously — a phantom deficit for a user eating perfectly well.

- Denominator invariance: the same eating pattern logged over 2 days or 6 days
  must produce identical macro percentages, and only the absolute calorie average
  may differ. This proves partial logging cannot bias the macro insights, which
  is what makes the active-days denominator safe to use.

- Stage 2 focus selection: the `06 §6` requirement proper. A protein-poor profile
  must yield `protein`; a balanced profile must yield `balance` and no actionable
  gaps. Determinism is tested too: the same profile must produce the same focus
  every time, because an insight that changes on refresh with no new meals reads
  as a bug.

- Goal sensitivity (FR13): one identical profile, three health goals, three
  different verdicts. This is the whole claim of "personalised to your goal" —
  25% protein is below target for weight loss and above it for maintenance, and
  the engine has to say so.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.db.models import HealthGoal
from app.ml import reco_targets as rt
from app.ml import recommender as rec

NOW = datetime(2026, 7, 30, 12, 0, 0)
WINDOW = 7


def meal(
    *,
    days_ago: float = 0,
    calories: float | None = 600.0,
    protein_g: float | None = 30.0,
    carbs_g: float | None = 60.0,
    fat_g: float | None = 20.0,
) -> rec.MealMacros:
    return rec.MealMacros(
        logged_at=NOW - timedelta(days=days_ago),
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
    )


def macro_meal(
    protein_g: float, carbs_g: float, fat_g: float, *, days_ago: float = 0
) -> rec.MealMacros:
    """A meal whose reported calories exactly match its macro-derived energy."""
    calories = protein_g * 4 + carbs_g * 4 + fat_g * 9
    return rec.MealMacros(
        logged_at=NOW - timedelta(days=days_ago),
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
    )


def profile_from(meals: list[rec.MealMacros]) -> rec.MacroProfile:
    result = rec.aggregate_profile(meals, window_days=WINDOW, now=NOW)
    assert result is not None, "fixture built an unusable profile"
    return result


# ---------------------------------------------------------------------------
# Stage 1 — the usability filter (06 §4)
# ---------------------------------------------------------------------------


def test_zero_calorie_meals_are_skipped():
    profile = profile_from([meal(), meal(calories=0.0)])
    assert profile.meals_counted == 1


def test_null_calorie_meals_are_skipped():
    profile = profile_from([meal(), meal(calories=None)])
    assert profile.meals_counted == 1


def test_meal_with_calories_but_no_macros_is_skipped():
    """
    The subtle one: counting it would add energy while adding no grams, so every
    macro percentage would be deflated at once.
    """
    profile = profile_from([meal(), meal(calories=500.0, protein_g=None, carbs_g=None, fat_g=None)])
    assert profile.meals_counted == 1


def test_meal_with_partial_macros_is_kept():
    """Some macro data is still signal; only a total absence is not."""
    profile = profile_from([meal(protein_g=40.0, carbs_g=None, fat_g=None)])
    assert profile.meals_counted == 1
    assert profile.macro_pct["protein"] == pytest.approx(100.0)


def test_negative_macros_are_rejected_not_averaged():
    """
    A corrupt row with a negative macro does not merely skew the average — it makes
    the percentage negative, and Stage 3 renders that into user-facing text
    ("below your target (-40% of calories vs 20%)"). Dropping it is the only
    outcome that cannot ship nonsense to a user.
    """
    profile = profile_from([meal(), meal(protein_g=-30.0)])
    assert profile.meals_counted == 1
    assert all(pct >= 0 for pct in profile.macro_pct.values())


def test_a_wholly_negative_meal_yields_no_profile():
    assert (
        rec.aggregate_profile(
            [meal(protein_g=-30.0, carbs_g=-60.0, fat_g=-20.0)], window_days=WINDOW, now=NOW
        )
        is None
    )


def test_future_timestamps_are_excluded():
    """
    Clock skew or a mistyped date on a client-supplied `logged_at` must not add a
    meal to a window it does not belong in, nor invent an active day.
    """
    assert rec.aggregate_profile([meal(days_ago=-3)], window_days=WINDOW, now=NOW) is None


def test_a_future_meal_does_not_inflate_active_days():
    profile = profile_from([meal(days_ago=0), meal(days_ago=-5)])
    assert profile.meals_counted == 1
    assert profile.active_days == 1


def test_profile_is_none_when_no_meal_is_usable():
    """A zero-filled profile would produce maximal gaps in every macro."""
    assert (
        rec.aggregate_profile(
            [meal(calories=0.0), meal(calories=None)], window_days=WINDOW, now=NOW
        )
        is None
    )


def test_profile_is_none_for_an_empty_list():
    assert rec.aggregate_profile([], window_days=WINDOW, now=NOW) is None


# ---------------------------------------------------------------------------
# Stage 1 — the rolling window
# ---------------------------------------------------------------------------


def test_meal_just_inside_the_window_is_counted():
    profile = profile_from([meal(days_ago=WINDOW - 0.05)])
    assert profile.meals_counted == 1


def test_meal_just_outside_the_window_is_excluded():
    assert (
        rec.aggregate_profile([meal(days_ago=WINDOW + 0.05)], window_days=WINDOW, now=NOW) is None
    )


def test_timezone_aware_timestamps_are_converted_not_truncated():
    """
    A +03:00 timestamp truncated rather than converted would land 3 h later and
    could cross the window boundary. Same instant, two representations, one result.
    """
    naive = rec.MealMacros(
        logged_at=datetime(2026, 7, 28, 9, 0, 0),
        calories=600.0,
        protein_g=30.0,
        carbs_g=60.0,
        fat_g=20.0,
    )
    aware = rec.MealMacros(
        logged_at=datetime(2026, 7, 28, 9, 0, 0, tzinfo=UTC),
        calories=600.0,
        protein_g=30.0,
        carbs_g=60.0,
        fat_g=20.0,
    )
    left = profile_from([naive])
    right = profile_from([aware])
    assert left.macro_pct == right.macro_pct
    assert left.active_days == right.active_days


# ---------------------------------------------------------------------------
# Stage 1 — the two documented design decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "protein_g,carbs_g,fat_g",
    [(30.0, 60.0, 20.0), (10.0, 100.0, 5.0), (80.0, 20.0, 40.0), (40.0, 0.0, 0.0)],
)
def test_macro_percentages_always_sum_to_100(protein_g, carbs_g, fat_g):
    profile = profile_from([macro_meal(protein_g, carbs_g, fat_g)])
    assert sum(profile.macro_pct.values()) == pytest.approx(100.0)


def test_percentages_sum_to_100_even_when_reported_calories_disagree():
    """
    The phantom-deficit bug. Reported calories 20% higher than the macros imply
    (rounding, fibre, alcohol) must not make all three macros read as low.
    """
    honest = macro_meal(30.0, 60.0, 20.0)
    inflated = rec.MealMacros(
        logged_at=honest.logged_at,
        calories=float(honest.calories) * 1.20,
        protein_g=honest.protein_g,
        carbs_g=honest.carbs_g,
        fat_g=honest.fat_g,
    )
    profile = profile_from([inflated])
    assert sum(profile.macro_pct.values()) == pytest.approx(100.0)
    assert profile.macro_pct == pytest.approx(profile_from([honest]).macro_pct)


def test_macro_percentages_are_independent_of_logging_coverage():
    """
    Denominator invariance: percentages are ratios, so spreading the same food
    over more days cannot change them. This is what makes the active-days
    denominator safe for the macro insights.
    """
    pattern = (30.0, 60.0, 20.0)
    dense = [macro_meal(*pattern, days_ago=0), macro_meal(*pattern, days_ago=0.2)]
    sparse = [macro_meal(*pattern, days_ago=1), macro_meal(*pattern, days_ago=5)]

    assert profile_from(dense).macro_pct == pytest.approx(profile_from(sparse).macro_pct)


def test_calorie_average_divides_by_active_days_not_window_length():
    """
    Two meals on one day is a ~1200 kcal day, not a 170 kcal one. Dividing by the
    7-day window would understate it by 7x and prompt advice to eat more.
    """
    profile = profile_from([macro_meal(30.0, 60.0, 20.0, days_ago=0)] * 2)
    per_meal_kcal = 30.0 * 4 + 60.0 * 4 + 20.0 * 9

    assert profile.active_days == 1
    assert profile.avg_daily_calories == pytest.approx(2 * per_meal_kcal)


def test_active_days_counts_distinct_calendar_dates():
    meals = [
        macro_meal(30.0, 60.0, 20.0, days_ago=0),
        macro_meal(30.0, 60.0, 20.0, days_ago=0.1),  # same date
        macro_meal(30.0, 60.0, 20.0, days_ago=2),
    ]
    assert profile_from(meals).active_days == 2


def test_grams_are_averaged_per_active_day():
    profile = profile_from(
        [macro_meal(30.0, 60.0, 20.0, days_ago=0), macro_meal(30.0, 60.0, 20.0, days_ago=3)]
    )
    assert profile.active_days == 2
    assert profile.avg_daily_grams["protein"] == pytest.approx(30.0)


def test_coverage_guard_blocks_a_calorie_verdict_on_thin_logging():
    one_day = profile_from([macro_meal(30.0, 60.0, 20.0, days_ago=0)])
    assert not one_day.has_enough_coverage_for_calorie_verdict

    spread = profile_from([macro_meal(30.0, 60.0, 20.0, days_ago=day) for day in (0, 1, 2)])
    assert spread.active_days == rt.MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT
    assert spread.has_enough_coverage_for_calorie_verdict


def test_larger_meals_carry_more_weight_than_small_ones():
    """
    Percentages come from summed grams, not from averaging per-meal ratios — a
    50 kcal snack must not count as much as a 900 kcal dinner.
    """
    big_protein_meal = macro_meal(90.0, 10.0, 5.0)
    tiny_carb_snack = macro_meal(0.5, 12.0, 0.2)
    profile = profile_from([big_protein_meal, tiny_carb_snack])

    solo = profile_from([big_protein_meal])
    assert abs(profile.macro_pct["protein"] - solo.macro_pct["protein"]) < 12.0


# ---------------------------------------------------------------------------
# Stage 2 — gaps, focus and the 06 §6 sanity requirement
# ---------------------------------------------------------------------------

# Built by construction so that PROTEIN is the single largest gap: ~2% of energy
# from protein against a 30% weight-loss target (-28 pp), with the displaced
# energy split between carbs and fat (~+14 pp each) so neither outranks it.
# Because the three percentages sum to 100 and the targets do too, the gaps
# always sum to zero — so making one macro dominant means deliberately spreading
# the counterpart surplus, not just driving one number to an extreme.
PROTEIN_POOR = [macro_meal(5.0, 135.0, 49.0, days_ago=day) for day in range(5)]
# 20/50/30 by energy — exactly the maintenance target.
BALANCED_MAINTENANCE = [macro_meal(50.0, 125.0, 33.3, days_ago=day) for day in range(5)]


def test_gaps_are_signed_percentage_points_against_the_goal_target():
    profile = profile_from(BALANCED_MAINTENANCE)
    gaps = rec.compute_gaps(profile, HealthGoal.maintenance)

    by_macro = {gap.macro: gap for gap in gaps}
    assert by_macro["protein"].target_pct == 20.0
    assert by_macro["protein"].gap_pp == pytest.approx(by_macro["protein"].actual_pct - 20.0)


def test_gaps_are_ranked_by_absolute_magnitude():
    profile = profile_from(PROTEIN_POOR)
    gaps = rec.compute_gaps(profile, HealthGoal.weight_loss)
    magnitudes = [abs(gap.gap_pp) for gap in gaps]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_protein_poor_profile_yields_protein_focus():
    """The headline 06 §6 sanity check."""
    profile = profile_from(PROTEIN_POOR)
    gaps = rec.compute_gaps(profile, HealthGoal.weight_loss)

    assert rec.primary_focus(gaps) == "protein"
    assert gaps[0].direction == "low"
    assert gaps[0].template_key == "protein_low"


def test_balanced_profile_yields_balance_focus_and_no_actionable_gaps():
    """The second 06 §6 walkthrough: a balanced user gets affirmation, not advice."""
    profile = profile_from(BALANCED_MAINTENANCE)
    gaps = rec.compute_gaps(profile, HealthGoal.maintenance)

    assert rec.primary_focus(gaps) == rt.FOCUS_BALANCE
    assert rec.actionable_gaps(gaps) == []
    assert len(rec.aligned_gaps(gaps)) == 3


def test_fat_heavy_profile_yields_fat_focus():
    profile = profile_from([macro_meal(20.0, 20.0, 80.0, days_ago=day) for day in range(4)])
    gaps = rec.compute_gaps(profile, HealthGoal.maintenance)

    assert rec.primary_focus(gaps) == "fat"
    assert gaps[0].direction == "high"


def test_focus_is_deterministic_across_repeated_calls():
    """An insight that changes on refresh with no new meals reads as a bug."""
    profile = profile_from(PROTEIN_POOR)
    focuses = {
        rec.primary_focus(rec.compute_gaps(profile, HealthGoal.weight_loss)) for _ in range(10)
    }
    assert focuses == {"protein"}


def test_same_profile_gives_different_verdicts_per_goal():
    """
    FR13's payoff. ~25% protein by energy is below the weight-loss target (30%)
    and above the maintenance target (20%) — the engine must say both.
    """
    profile = profile_from([macro_meal(50.0, 100.0, 30.0, days_ago=day) for day in range(4)])

    loss = {g.macro: g for g in rec.compute_gaps(profile, HealthGoal.weight_loss)}
    maintain = {g.macro: g for g in rec.compute_gaps(profile, HealthGoal.maintenance)}

    assert loss["protein"].gap_pp < maintain["protein"].gap_pp
    assert loss["protein"].target_pct == 30.0
    assert maintain["protein"].target_pct == 20.0


def test_actionable_gaps_are_capped_at_three():
    profile = profile_from(PROTEIN_POOR)
    gaps = rec.compute_gaps(profile, HealthGoal.weight_loss)
    assert len(rec.actionable_gaps(gaps)) <= 3


def test_actionable_gaps_exclude_aligned_macros():
    profile = profile_from(BALANCED_MAINTENANCE)
    gaps = rec.compute_gaps(profile, HealthGoal.maintenance)
    assert all(not gap.is_aligned for gap in rec.actionable_gaps(gaps))


def test_tolerance_boundary_is_inclusive():
    """A macro exactly at the tolerance edge counts as aligned, not as a problem."""
    gap = rec.MacroGap(
        macro="protein",
        actual_pct=20.0 + rt.MACRO_ALIGNED_TOLERANCE_PP,
        target_pct=20.0,
        gap_pp=rt.MACRO_ALIGNED_TOLERANCE_PP,
    )
    assert gap.is_aligned
    assert gap.direction == "aligned"


def test_a_gap_just_past_the_tolerance_is_still_actionable():
    """
    The epsilon in `is_aligned` must absorb float noise only — not real distance.
    A gap 0.04 pp past the tolerance is a genuine difference and must stay
    actionable, otherwise the guard would have quietly widened the threshold.
    """
    gap = rec.MacroGap(
        macro="carbs",
        actual_pct=45.04,
        target_pct=40.0,
        gap_pp=5.04,
    )
    assert not gap.is_aligned
    assert gap.direction == "high"


def test_every_direction_maps_to_an_existing_template():
    """
    Guards the Stage 3 contract: `template_key` must resolve for every macro and
    every non-aligned direction, or generation raises KeyError on a live request.
    """
    for macro in rt.MACROS:
        for direction, gap_pp in (("low", -20.0), ("high", 20.0)):
            gap = rec.MacroGap(macro=macro, actual_pct=0.0, target_pct=0.0, gap_pp=gap_pp)
            assert gap.direction == direction
            assert gap.template_key in rt.INSIGHT_TEMPLATES
