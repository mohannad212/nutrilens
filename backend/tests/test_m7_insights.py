"""
tests/test_m7_insights.py
============================

WHY THIS FILE EXISTS
---------------------
Stage 3 is where the engine stops doing arithmetic and starts talking to a user
about their eating. Two very different classes of thing can go wrong here, and
this file covers both:

1. **Contract failures** — more than 5 insights, a missing positive insight, an
   unrendered `{placeholder}` leaking into user-facing text, a KeyError on a
   template that doesn't exist. These break FR12 or crash the request.
2. **Wellbeing failures** — a shaming phrase, a calorie verdict invented from no
   data, an "eat less" nudge aimed at someone who logged one day out of seven.
   These are the ones that matter more, because a user reads them and believes
   them, and `06 §5` calls them non-negotiable.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- Composition and the cap of 5: asserted across every goal and several profile
  shapes rather than one, because the cap holds by *construction* (<=3 corrective
  + <=1 calorie + 1 positive) and a change to any of those three parts could
  break it without touching the truncation.

- The mandatory positive insight (`06 §2`): tested on a deliberately bad profile,
  not a good one. Any implementation gets this right for a balanced user; the
  requirement exists precisely so that someone whose week went badly still gets
  one affirming line instead of a wall of corrections.

- The calorie insight's three-way behaviour: the interesting cases are the two
  where the engine must *decline* to judge — no body metrics, and too few logged
  days. Both produce a descriptive note instead. A recommender that guesses a
  calorie target and tells a user they are overeating is the single most harmful
  thing this module could do, so both paths are pinned.

- Food scoring: that alignment is measured *relative to what the user currently
  eats* (an 18%-protein food helps nobody already at 20%), that conflicting foods
  are penalised, that frequently-logged foods are demoted, and that unscorable
  rows — the normal state of `food_item` before M5 fills the USDA cache — cleanly
  fall through to the curated examples.

- Tone: no rendered insight may contain a blocked phrase, checked over a matrix
  of profiles x goals rather than on the templates alone, because interpolation
  is what could introduce one.
"""

from datetime import datetime, timedelta

import pytest

from app.db.models import HealthGoal
from app.ml import reco_targets as rt
from app.ml import recommender as rec

NOW = datetime(2026, 7, 30, 12, 0, 0)
WINDOW = 7


def macro_meal(
    protein_g: float, carbs_g: float, fat_g: float, *, days_ago: float = 0
) -> rec.MealMacros:
    return rec.MealMacros(
        logged_at=NOW - timedelta(days=days_ago),
        calories=protein_g * 4 + carbs_g * 4 + fat_g * 9,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
    )


def profile_of(
    protein_g: float, carbs_g: float, fat_g: float, *, days: int = 5
) -> rec.MacroProfile:
    meals = [macro_meal(protein_g, carbs_g, fat_g, days_ago=day) for day in range(days)]
    result = rec.aggregate_profile(meals, window_days=WINDOW, now=NOW)
    assert result is not None
    return result


# Protein ~2% of energy against a 30% weight-loss target, surplus split between
# carbs and fat so protein is the dominant gap (see test_m7_recommender.py).
PROTEIN_POOR = profile_of(5.0, 135.0, 49.0)
# 20/50/30 by energy — exactly the maintenance target, so every macro is aligned.
BALANCED = profile_of(50.0, 125.0, 33.3)
FAT_HEAVY = profile_of(20.0, 20.0, 80.0)

ALL_PROFILES = [PROTEIN_POOR, BALANCED, FAT_HEAVY]


def texts(insights: list[rec.Insight]) -> list[str]:
    return [insight.insight_text for insight in insights]


# ---------------------------------------------------------------------------
# Composition, ordering and the cap of 5 (FR12 / 06 §1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_never_more_than_five_insights(profile, goal):
    insights = rec.generate_insights(profile, goal=goal, calorie_target=2200.0)
    assert len(insights) <= rt.MAX_INSIGHTS


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_always_at_least_two_insights(profile, goal):
    """A single lonely card is a bad screen; 06 §4's behavioural fallback fills it."""
    insights = rec.generate_insights(profile, goal=goal)
    assert len(insights) >= rt.MIN_INSIGHTS_BEFORE_PADDING


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_every_insight_has_text_and_a_focus(profile, goal):
    for insight in rec.generate_insights(profile, goal=goal, calorie_target=2200.0):
        assert insight.insight_text.strip()
        assert insight.nutrient_focus
        # 02: recommendation.nutrient_focus is VARCHAR(30).
        assert len(insight.nutrient_focus) <= 30


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_no_unrendered_placeholders_leak_into_user_text(profile, goal):
    """A stray {examples} in a shipped insight is both a bug and embarrassing."""
    for text in texts(rec.generate_insights(profile, goal=goal, calorie_target=2200.0)):
        assert "{" not in text and "}" not in text


def test_corrective_insights_come_before_the_positive_one():
    """Most actionable first; the affirming line closes rather than buries."""
    insights = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss)
    assert insights[0].nutrient_focus == "protein"
    assert "Keep it up" in insights[-1].insight_text or "consistency" in insights[-1].insight_text


def test_at_most_three_corrective_macro_insights():
    macro_focuses = [
        insight.nutrient_focus
        for insight in rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss)
        if insight.nutrient_focus in rt.MACROS
    ]
    # <=3 corrective + possibly 1 positive naming an aligned macro.
    assert len(macro_focuses) <= 4


def test_generation_is_deterministic():
    """Same input, same output — a refresh with no new meals must not reshuffle."""
    first = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss, calorie_target=2000.0)
    second = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss, calorie_target=2000.0)
    assert first == second


# ---------------------------------------------------------------------------
# The mandatory positive insight (06 §2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_a_positive_insight_is_always_present(profile, goal):
    """
    Tested on bad profiles as well as good ones: the requirement exists so a user
    whose week went badly still gets one affirming line, not a wall of criticism.
    """
    rendered = " ".join(texts(rec.generate_insights(profile, goal=goal)))
    affirming = ("Keep it up", "consistency", "well aligned", "hardest")
    assert any(phrase in rendered for phrase in affirming)


def test_balanced_user_is_affirmed_by_name_not_corrected():
    insights = rec.generate_insights(BALANCED, goal=HealthGoal.maintenance)
    joined = " ".join(texts(insights))

    assert "well aligned" in joined
    assert not any(
        insight.insight_text.startswith("Your protein has been below") for insight in insights
    )


def test_positive_insight_falls_back_to_consistency_when_nothing_is_aligned():
    """
    FAT_HEAVY has no macro within tolerance, so there is no macro to praise. The
    engine credits the logging habit instead — always something true to say.
    """
    joined = " ".join(texts(rec.generate_insights(FAT_HEAVY, goal=HealthGoal.maintenance)))
    assert "hardest" in joined or "consistency" in joined


# ---------------------------------------------------------------------------
# The calorie insight — and the two cases where it must decline to judge
# ---------------------------------------------------------------------------


def test_no_body_metrics_gives_a_descriptive_note_not_a_verdict():
    """
    The core honesty property. Without a TDEE there is no target, so the engine
    reports the average and invites the user to complete their profile — it never
    says "above" or "below".
    """
    insights = rec.generate_insights(BALANCED, goal=HealthGoal.weight_loss, calorie_target=None)
    calorie = [i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES]

    assert len(calorie) == 1
    assert "profile" in calorie[0].insight_text
    assert "above" not in calorie[0].insight_text
    assert "below" not in calorie[0].insight_text


def test_calorie_note_names_only_the_metrics_actually_missing():
    """
    The dead-end loop this guards against: a user told to add "height, weight and
    activity level" does exactly that, TDEE still returns None because age and sex
    are absent, and the identical note reappears. Naming the real gaps means
    following the suggestion always makes progress.
    """
    insights = rec.generate_insights(
        BALANCED,
        goal=HealthGoal.maintenance,
        calorie_target=None,
        missing_metrics=["age", "sex"],
    )
    calorie = next(i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES)

    # "and", not "or" — both are required, so "or" would imply either will do.
    assert "age and sex" in calorie.insight_text
    assert "height" not in calorie.insight_text
    assert "weight" not in calorie.insight_text


def test_calorie_note_lists_all_five_metrics_when_none_are_known():
    insights = rec.generate_insights(BALANCED, goal=HealthGoal.maintenance, calorie_target=None)
    calorie = next(i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES)

    for label in rt.METRIC_LABELS.values():
        assert label in calorie.insight_text


def test_complete_profile_is_never_asked_to_complete_the_profile():
    """
    The second copy bug: thin coverage and a missing profile are different
    situations. A user whose profile is already full must not be told to fill it.
    """
    one_day = profile_of(50.0, 125.0, 33.3, days=1)
    insights = rec.generate_insights(
        one_day,
        goal=HealthGoal.maintenance,
        calorie_target=2200.0,
        missing_metrics=[],
    )
    calorie = next(i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES)

    assert "profile" not in calorie.insight_text
    assert "Logging on a few more days" in calorie.insight_text
    # Pluralisation, asserted as the whole phrase: a loose "1 day" substring check
    # also matches the broken "1 day days" a leftover literal would produce.
    assert "across the 1 day you recorded" in calorie.insight_text


def test_thin_logging_coverage_blocks_a_calorie_verdict():
    """
    One logged day is not a daily average. Telling this user they are eating too
    little would be advice drawn from a single meal.
    """
    one_day = profile_of(5.0, 135.0, 49.0, days=1)
    assert one_day.active_days == 1

    insights = rec.generate_insights(one_day, goal=HealthGoal.weight_loss, calorie_target=2400.0)
    calorie = [i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES]

    assert len(calorie) == 1
    assert "above" not in calorie[0].insight_text
    assert "below" not in calorie[0].insight_text


def test_calorie_verdict_is_given_when_target_and_coverage_both_exist():
    profile = profile_of(50.0, 125.0, 33.3, days=5)
    low_target = profile.avg_daily_calories / 2  # well outside tolerance

    insights = rec.generate_insights(
        profile, goal=HealthGoal.weight_loss, calorie_target=low_target
    )
    calorie = [i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES]

    assert len(calorie) == 1
    assert "above" in calorie[0].insight_text


def test_calorie_insight_reports_below_when_intake_is_under_target():
    profile = profile_of(50.0, 125.0, 33.3, days=5)
    high_target = profile.avg_daily_calories * 2

    insights = rec.generate_insights(
        profile, goal=HealthGoal.weight_gain, calorie_target=high_target
    )
    calorie = [i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES]
    assert "below" in calorie[0].insight_text


def test_no_calorie_insight_when_intake_is_within_tolerance():
    """
    Inside the tolerance the difference is smaller than the estimate's own error.
    Saying nothing leaves room for an insight that is actually informative.
    """
    profile = profile_of(50.0, 125.0, 33.3, days=5)
    insights = rec.generate_insights(
        profile, goal=HealthGoal.maintenance, calorie_target=profile.avg_daily_calories
    )
    assert [i for i in insights if i.nutrient_focus == rt.FOCUS_CALORIES] == []


# ---------------------------------------------------------------------------
# Content-based food scoring (06 §1 Stage 3)
# ---------------------------------------------------------------------------


def food(
    name: str,
    protein: float,
    carbs: float,
    fat: float,
    *,
    times_logged: int = 0,
    calories: float | None = 200.0,
) -> rec.FoodCandidate:
    return rec.FoodCandidate(
        food_name=name,
        calories_per_100g=calories,
        protein_per_100g=protein,
        carbs_per_100g=carbs,
        fat_per_100g=fat,
        times_logged=times_logged,
    )


PROTEIN_GAP = rec.MacroGap(macro="protein", actual_pct=2.0, target_pct=30.0, gap_pp=-28.0)


def test_seeded_rows_with_null_macros_are_unscorable():
    """
    The normal state of `food_item` today: 101 class names, no nutrition. These
    must fall through to the curated examples rather than scoring as zeros.
    """
    seeded = rec.FoodCandidate(
        food_name="Chicken Curry",
        calories_per_100g=None,
        protein_per_100g=None,
        carbs_per_100g=None,
        fat_per_100g=None,
    )
    assert not seeded.is_scorable
    assert (
        rec.select_foods(
            [seeded], gap=PROTEIN_GAP, profile=PROTEIN_POOR, goal=HealthGoal.weight_loss
        )
        == []
    )


def test_high_protein_food_scores_above_low_protein_food_for_a_protein_gap():
    lean = food("grilled chicken", protein=30.0, carbs=0.0, fat=3.0)
    bread = food("white bread", protein=8.0, carbs=50.0, fat=3.0)

    kwargs = {"gap": PROTEIN_GAP, "profile": PROTEIN_POOR, "goal": HealthGoal.weight_loss}
    assert rec.score_food(lean, **kwargs) > rec.score_food(bread, **kwargs)


def test_alignment_is_relative_to_what_the_user_already_eats():
    """
    An 18%-protein food does nothing for someone already averaging 20% — absolute
    protein density is not the criterion, movement toward the target is.
    """
    already_at_20 = profile_of(50.0, 125.0, 33.3)
    gap = rec.MacroGap(macro="protein", actual_pct=20.0, target_pct=30.0, gap_pp=-10.0)
    modest = food("modest protein", protein=9.0, carbs=45.0, fat=8.0)

    assert rec.score_food(modest, gap=gap, profile=already_at_20, goal=HealthGoal.weight_loss) <= 0


def test_conflicting_macros_are_penalised():
    """
    Both foods help the protein gap; the fried one worsens the existing fat
    surplus, so the leaner one must win.
    """
    lean = food("grilled chicken", protein=30.0, carbs=0.0, fat=3.0)
    fried = food("fried chicken", protein=30.0, carbs=8.0, fat=25.0)

    kwargs = {"gap": PROTEIN_GAP, "profile": PROTEIN_POOR, "goal": HealthGoal.weight_loss}
    assert rec.score_food(lean, **kwargs) > rec.score_food(fried, **kwargs)


def test_frequently_logged_foods_are_demoted():
    """Suggesting more of what someone eats daily is not a recommendation."""
    novel = food("lentils", protein=25.0, carbs=20.0, fat=2.0, times_logged=0)
    routine = food("eggs", protein=25.0, carbs=20.0, fat=2.0, times_logged=9)

    kwargs = {"gap": PROTEIN_GAP, "profile": PROTEIN_POOR, "goal": HealthGoal.weight_loss}
    assert rec.score_food(novel, **kwargs) > rec.score_food(routine, **kwargs)


def test_frequency_penalty_saturates():
    heavy = food("eggs", protein=25.0, carbs=20.0, fat=2.0, times_logged=100)
    at_cap = food(
        "eggs", protein=25.0, carbs=20.0, fat=2.0, times_logged=rt.FREQUENCY_SATURATION_LOGS
    )

    kwargs = {"gap": PROTEIN_GAP, "profile": PROTEIN_POOR, "goal": HealthGoal.weight_loss}
    assert rec.score_food(heavy, **kwargs) == pytest.approx(rec.score_food(at_cap, **kwargs))


def test_unhelpful_foods_are_never_suggested():
    """Zero or negative score means it moves the user the wrong way — name nothing."""
    sugary = food("candy", protein=0.0, carbs=95.0, fat=0.0)
    assert (
        rec.select_foods(
            [sugary], gap=PROTEIN_GAP, profile=PROTEIN_POOR, goal=HealthGoal.weight_loss
        )
        == []
    )


def test_selection_is_capped_and_deterministic():
    candidates = [
        food(name, protein=30.0, carbs=5.0, fat=2.0)
        for name in ("tuna", "chicken", "lentils", "tofu", "eggs")
    ]
    chosen = rec.select_foods(
        candidates, gap=PROTEIN_GAP, profile=PROTEIN_POOR, goal=HealthGoal.weight_loss
    )

    assert len(chosen) == rt.MAX_EXAMPLES_PER_INSIGHT
    # Equal scores must break on name, not on query order.
    assert chosen == sorted(chosen)


def test_scored_foods_appear_in_the_rendered_insight():
    candidates = [food("grilled halloumi", protein=25.0, carbs=2.0, fat=8.0)]
    insights = rec.generate_insights(
        PROTEIN_POOR, goal=HealthGoal.weight_loss, candidates=candidates
    )
    assert "grilled halloumi" in " ".join(texts(insights))


def test_curated_examples_are_used_when_the_cache_is_empty():
    """The normal path until M5 populates the USDA cache."""
    insights = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss, candidates=[])
    joined = " ".join(texts(insights))
    assert any(example in joined for example in rt.FOOD_EXAMPLES["protein"])


def test_examples_are_joined_readably():
    insights = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss)
    protein_text = next(i.insight_text for i in insights if i.nutrient_focus == "protein")
    assert " or " in protein_text


# ---------------------------------------------------------------------------
# Tone and wellbeing (06 §5) — the non-negotiable ones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal", list(HealthGoal))
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_no_generated_insight_contains_a_blocked_phrase(profile, goal):
    """
    Checked on *rendered* text, not just templates: interpolation of a goal name
    or a food name from the USDA cache is what could introduce one.
    """
    for insight in rec.generate_insights(profile, goal=goal, calorie_target=1800.0):
        hit = rec.contains_blocked_phrase(insight.insight_text)
        assert hit is None, f"blocked phrase {hit!r} in: {insight.insight_text}"


def test_an_unsafe_food_name_cannot_reach_the_user():
    """
    The runtime filter earning its place: a cache row named with blocked wording
    must not be laundered into an insight by interpolation.
    """
    nasty = food("junk food surprise", protein=40.0, carbs=5.0, fat=2.0)
    insights = rec.generate_insights(PROTEIN_POOR, goal=HealthGoal.weight_loss, candidates=[nasty])
    for insight in insights:
        assert "junk food" not in insight.insight_text.lower()


def test_blocked_phrase_detection_is_case_insensitive():
    # Returns the first match in BLOCKED_PHRASES order, where "guilty" precedes "guilt".
    assert rec.contains_blocked_phrase("You should feel GUILTY about this") == "guilty"
    assert rec.contains_blocked_phrase("A balanced plate helps") is None


@pytest.mark.parametrize("goal", list(HealthGoal))
def test_insights_use_tentative_not_imperative_language(goal):
    """
    06 §5's tone rule, spot-checked: the corrective templates suggest ("could",
    "may", "tends to") rather than command ("you must", "stop eating").
    """
    joined = " ".join(texts(rec.generate_insights(PROTEIN_POOR, goal=goal))).lower()
    assert "you must" not in joined
    assert "stop eating" not in joined
