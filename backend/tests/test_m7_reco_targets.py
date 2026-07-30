"""
tests/test_m7_reco_targets.py
================================

WHY THIS FILE EXISTS
---------------------
`ml/reco_targets.py` is the recommendation engine's entire "model": reference
numbers and sentence templates. It contains no logic to speak of, which is
exactly why it needs tests — a data file fails silently. A macro target that
sums to 95% instead of 100% produces a permanent phantom gap the engine will
report to every user forever, and no exception is ever raised.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- Target integrity: the three distributions must each sum to 100% of energy,
  because Stage 2 subtracts observed percentages from them. If a column of `06
  §1`'s table is mistyped here, every user of that goal gets a fabricated gap.

- TDEE estimation: pinned against a hand-computed Mifflin-St Jeor value so the
  equation itself is verified, not just "it returns a number". The `None`
  behaviour is tested per-field, because a partially filled profile is the
  common real case and must NOT quietly produce a wrong estimate from defaults.

- Goal posture and floor: the deficit/surplus must stay mild, and the floor must
  hold even when a low TDEE and a deficit posture multiply out below it. This is
  the mechanical enforcement of `06 §5`'s "never recommend extreme deficits" —
  the one rule in this module that is a safety property rather than an accuracy
  one.

- Template and tone hygiene: every template must be renderable (no stray
  placeholder that would raise KeyError at request time) and no template may
  contain a blocked phrase. Because the templates are the only source of
  user-facing text in the engine, checking them here proves no code path can emit
  a shaming sentence — the property `06 §5` requires.
"""

import re
from datetime import UTC, datetime

import pytest

from app.db.models import ActivityLevel, HealthGoal, Sex
from app.ml import reco_targets as rt

# ---------------------------------------------------------------------------
# Target distribution integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal", list(HealthGoal))
def test_every_goal_has_a_target_distribution(goal):
    assert goal in rt.MACRO_TARGETS


@pytest.mark.parametrize("goal", list(HealthGoal))
def test_targets_sum_to_100_percent(goal):
    """Stage 2 subtracts from these; a sum != 100 is a permanent phantom gap."""
    assert sum(rt.MACRO_TARGETS[goal].values()) == pytest.approx(100.0)


@pytest.mark.parametrize("goal", list(HealthGoal))
def test_targets_cover_exactly_the_three_macros(goal):
    assert set(rt.MACRO_TARGETS[goal]) == set(rt.MACROS)


def test_weight_loss_targets_more_protein_than_maintenance():
    """The documented rationale in 06 §1 — satiety/lean-mass retention."""
    assert (
        rt.MACRO_TARGETS[HealthGoal.weight_loss]["protein"]
        > rt.MACRO_TARGETS[HealthGoal.maintenance]["protein"]
    )


# ---------------------------------------------------------------------------
# Mifflin-St Jeor
# ---------------------------------------------------------------------------

# 25-year-old (for the reference year below), 75 kg, 180 cm, male:
#   10*75 + 6.25*180 - 5*25 + 5 = 750 + 1125 - 125 + 5 = 1755
REFERENCE_AGE = 25
REFERENCE_BMR_MALE = 1755.0


def _birth_year_for_age(age: int) -> int:
    return datetime.now(UTC).year - age


def test_bmr_matches_hand_computed_mifflin_st_jeor():
    bmr = rt.estimate_bmr(
        weight_kg=75.0,
        height_cm=180.0,
        birth_year=_birth_year_for_age(REFERENCE_AGE),
        sex=Sex.male,
    )
    assert bmr == pytest.approx(REFERENCE_BMR_MALE)


def test_female_constant_lowers_bmr_by_the_documented_offset():
    kwargs = {
        "weight_kg": 75.0,
        "height_cm": 180.0,
        "birth_year": _birth_year_for_age(REFERENCE_AGE),
    }
    male = rt.estimate_bmr(sex=Sex.male, **kwargs)
    female = rt.estimate_bmr(sex=Sex.female, **kwargs)
    assert male - female == pytest.approx(166.0)  # +5 vs -161


def test_sex_has_exactly_two_values():
    """
    Product decision: no third "unspecified" option. Declining is expressed by
    leaving the column NULL, not by a placeholder value — so there is no code path
    that invents a substitute constant on the user's behalf.
    """
    assert {member.value for member in Sex} == {"female", "male"}
    assert set(rt.SEX_CONSTANTS) == set(Sex)


def test_declining_to_state_sex_yields_no_estimate_rather_than_a_guess():
    """
    The privacy path, end to end: sex is the only missing field, and the result is
    `None` — not a midpoint, not a default body. The caller then degrades to a
    descriptive calorie note while every macro insight still works.
    """
    assert (
        rt.estimate_tdee(
            weight_kg=75.0,
            height_cm=180.0,
            birth_year=_birth_year_for_age(REFERENCE_AGE),
            sex=None,
            activity_level=ActivityLevel.moderate,
        )
        is None
    )


@pytest.mark.parametrize(
    "missing_field",
    ["weight_kg", "height_cm", "birth_year", "sex"],
)
def test_bmr_is_none_when_any_single_field_is_missing(missing_field):
    """A partial profile must not silently fall back to a default body."""
    kwargs = {
        "weight_kg": 75.0,
        "height_cm": 180.0,
        "birth_year": _birth_year_for_age(REFERENCE_AGE),
        "sex": Sex.male,
    }
    kwargs[missing_field] = None
    assert rt.estimate_bmr(**kwargs) is None


def test_tdee_applies_the_activity_multiplier():
    tdee = rt.estimate_tdee(
        weight_kg=75.0,
        height_cm=180.0,
        birth_year=_birth_year_for_age(REFERENCE_AGE),
        sex=Sex.male,
        activity_level=ActivityLevel.moderate,
    )
    assert tdee == pytest.approx(REFERENCE_BMR_MALE * 1.55)


def test_tdee_is_none_without_activity_level():
    assert (
        rt.estimate_tdee(
            weight_kg=75.0,
            height_cm=180.0,
            birth_year=_birth_year_for_age(REFERENCE_AGE),
            sex=Sex.male,
            activity_level=None,
        )
        is None
    )


def test_activity_multipliers_are_ordered_and_bounded():
    ordered = [
        ActivityLevel.sedentary,
        ActivityLevel.light,
        ActivityLevel.moderate,
        ActivityLevel.active,
        ActivityLevel.very_active,
    ]
    values = [rt.ACTIVITY_MULTIPLIERS[level] for level in ordered]
    assert values == sorted(values)
    assert 1.2 <= values[0] and values[-1] <= 1.9


# ---------------------------------------------------------------------------
# Goal posture and the safety floor (06 §5)
# ---------------------------------------------------------------------------


def test_goal_posture_is_mild_in_both_directions():
    """Turns "never extreme" into a number: within 20% of maintenance either way."""
    for goal, posture in rt.GOAL_CALORIE_POSTURE.items():
        assert 0.80 <= posture <= 1.20, goal


def test_weight_loss_is_a_deficit_and_gain_is_a_surplus():
    assert rt.GOAL_CALORIE_POSTURE[HealthGoal.weight_loss] < 1.0
    assert rt.GOAL_CALORIE_POSTURE[HealthGoal.maintenance] == 1.0
    assert rt.GOAL_CALORIE_POSTURE[HealthGoal.weight_gain] > 1.0


def test_target_calories_never_fall_below_the_floor():
    """
    A small/inactive body with a deficit posture multiplies out below the floor;
    the clamp is what stops the engine ever stating an unsafe-looking target.
    """
    tiny_tdee = 1000.0
    target = rt.goal_target_calories(tiny_tdee, HealthGoal.weight_loss)
    assert target == rt.CALORIE_FLOOR_KCAL
    assert target >= rt.CALORIE_FLOOR_KCAL


def test_target_calories_are_none_without_a_tdee():
    """No body metrics -> no target -> the caller must not make a verdict."""
    assert rt.goal_target_calories(None, HealthGoal.maintenance) is None


def test_normal_tdee_is_not_clamped():
    target = rt.goal_target_calories(2400.0, HealthGoal.weight_loss)
    assert target == pytest.approx(2400.0 * 0.85)


# ---------------------------------------------------------------------------
# Template and tone hygiene
# ---------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Everything the generator can supply. A template referencing anything outside
# this set would raise KeyError on a live request.
KNOWN_PLACEHOLDERS = {
    "goal",
    "actual",
    "target",
    "examples",
    "nutrient",
    "kcal",
    "target_kcal",
    "direction",
    "days",
    "meals",
    "missing",
}


@pytest.mark.parametrize("key", sorted(rt.INSIGHT_TEMPLATES))
def test_template_placeholders_are_all_supported(key):
    used = set(PLACEHOLDER_RE.findall(rt.INSIGHT_TEMPLATES[key]))
    assert used <= KNOWN_PLACEHOLDERS, f"{key} uses unknown placeholder(s)"


@pytest.mark.parametrize("key", sorted(rt.INSIGHT_TEMPLATES))
def test_no_template_contains_a_blocked_phrase(key):
    """
    The core wellbeing property (06 §5). Templates are the only text source in
    the engine, so this proves no code path can emit a shaming sentence.
    """
    text = rt.INSIGHT_TEMPLATES[key].lower()
    hits = [phrase for phrase in rt.BLOCKED_PHRASES if phrase in text]
    assert hits == [], f"{key} contains blocked phrasing: {hits}"


def test_fibre_template_was_removed():
    """
    Deliberate scope decision: the schema caches energy + three macros only, so
    a fibre claim would have no data behind it. Documented in 06.
    """
    assert "fibre_low" not in rt.INSIGHT_TEMPLATES


def test_blocked_phrases_do_not_ban_ordinary_nutrition_words():
    """
    The classic own-goal: banning "fat" would make the fat-macro insights
    unwritable. Blocked entries must be judgement phrases, not nutrients.
    """
    for nutrient in ("fat", "carbs", "protein", "calories"):
        assert nutrient not in rt.BLOCKED_PHRASES


def test_missing_metric_labels_covers_every_tdee_input():
    """
    The list must name all five inputs `estimate_tdee` requires. If it named
    fewer, a user could supply everything the calorie note asked for and still
    get no verdict — following the instruction would change nothing.
    """
    all_missing = rt.missing_metric_labels(
        height_cm=None, weight_kg=None, birth_year=None, sex=None, activity_level=None
    )
    assert all_missing == ["height", "weight", "age", "sex", "activity level"]


def test_missing_metric_labels_is_empty_for_a_complete_profile():
    assert (
        rt.missing_metric_labels(
            height_cm=180.0,
            weight_kg=75.0,
            birth_year=2000,
            sex=Sex.male,
            activity_level=ActivityLevel.moderate,
        )
        == []
    )


def test_missing_metric_labels_names_only_what_is_absent():
    """A user four-fifths done should be asked for the fifth field, not all five."""
    assert rt.missing_metric_labels(
        height_cm=180.0,
        weight_kg=75.0,
        birth_year=2000,
        sex=Sex.female,
        activity_level=None,
    ) == ["activity level"]


def test_an_omitted_sex_is_reported_as_missing():
    """
    With no `unspecified` value, a NULL sex is a genuine gap and the calorie note
    should say so — the user can then choose to fill it in or accept that the
    calorie insight stays descriptive.
    """
    assert "sex" in rt.missing_metric_labels(
        height_cm=180.0,
        weight_kg=75.0,
        birth_year=2000,
        sex=None,
        activity_level=ActivityLevel.light,
    )


def test_health_goal_is_not_a_missing_metric():
    """It is NOT NULL at registration, so it can never be one of the gaps."""
    assert "goal" not in rt.METRIC_LABELS.values()
    assert "health goal" not in rt.METRIC_LABELS.values()


def test_every_goal_has_a_display_name():
    for goal in HealthGoal:
        assert rt.GOAL_DISPLAY[goal] and "_" not in rt.GOAL_DISPLAY[goal]


def test_examples_exist_for_every_macro():
    for macro in rt.MACROS:
        assert len(rt.FOOD_EXAMPLES[macro]) >= rt.MAX_EXAMPLES_PER_INSIGHT


def test_insight_ceiling_matches_fr12():
    assert rt.MAX_INSIGHTS == 5
