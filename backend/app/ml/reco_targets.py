"""
app/ml/reco_targets.py
=========================

WHY THIS FILE EXISTS
---------------------
The recommendation engine in `06` is a *rule-based, content-based* recommender:
its entire "model" is a set of reference numbers and a set of sentence templates.
This file is that model. Everything the engine knows about nutrition — the target
macro distribution for each health goal, how many kcal a goal implies, which
foods to name as examples, and the exact words used to say it — lives here, and
`recommender.py` next door contains only the arithmetic that consumes it.

WHAT PROBLEM IT SOLVES
------------------------
Rules and reference values scattered through the code that applies them are
impossible to review, cite, or defend. A supervisor asking "where does 30%
protein come from, and what happens if it changes?" should get one file and one
answer, not a grep across a service layer. Separating the *values* from the
*algorithm* also means the wellbeing constraints of `06 §5` are auditable: every
sentence this app can ever say about a user's diet is a literal in this file, so
"prove nothing shaming can be emitted" is a review of one module, and the
blocked-phrase test can assert it mechanically.

HOW IT WORKS INTERNALLY
--------------------------
Four groups of data, all immutable module-level constants:

1. `MACRO_TARGETS` — protein/carbs/fat as **% of energy** per health goal
   (`06 §1` Stage 2). Percent-of-energy rather than grams is what makes the
   comparison body-size independent: it is a ratio, so it needs no anthropometrics
   and stays valid whether the user eats 1,600 or 3,000 kcal.
2. Energy estimation — `estimate_tdee()` implements Mifflin-St Jeor (BMR from
   weight/height/age/sex, scaled by an activity multiplier), and
   `goal_target_calories()` applies a deliberately *mild* posture per goal. This
   is the only part of the engine that needs the optional body metrics added in
   migration `0002`; it returns `None` when they are absent so the caller can
   degrade to a descriptive statement instead of guessing.
3. `FOOD_EXAMPLES` — a small curated list of everyday, non-prescriptive foods per
   nutrient, used to fill `{examples}` when the `food_item` cache has nothing
   better to offer.
4. `INSIGHT_TEMPLATES` + `BLOCKED_PHRASES` — the sentences, and the phrases that
   must never appear in one.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Mifflin-St Jeor (chosen) for BMR: it is the equation most widely reported as
  the more accurate of the common predictive equations for non-obese adults, and
  it needs only the four inputs we are willing to ask for. Harris-Benedict is
  older and generally reported as less accurate; Katch-McArdle needs body-fat
  percentage, which we cannot obtain from a photo-based food logger.
- A *mild* goal posture (-15% / +10%) with an absolute floor: `06 §5` forbids
  recommending extreme deficits. Encoding the mildness as a constant here — and
  clamping the result at `CALORIE_FLOOR_KCAL` — makes it structurally impossible
  for the engine to suggest a starvation-level target, even if a user's TDEE
  estimate comes out implausibly low from bad input.
- Templates as format strings (chosen) over generated prose from an LLM: every
  possible output is enumerable and reviewable, the app works offline, there is
  no per-request cost or latency, and no chance of a model inventing a medical
  claim. For a rule-based recommender at this scope this is also what the report
  can honestly describe and evaluate (`06 §6`).
- Targets in a database table instead of code: they are not user data and change
  only when the algorithm changes, which is a code change with a code review.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Where do the macro percentages come from?"
A: They are the reference proportions given in `06 §1` — informational defaults
   drawn from widely published dietary-guideline ranges (a higher protein share
   for weight loss to support satiety and lean-mass retention, a moderate share
   for maintenance). They are stated in the report as *reference proportions, not
   medical prescriptions*, which is also why every response carries the
   disclaimer.

Q: "Why does `estimate_tdee` return None instead of a default?"
A: Because a default would be a fabricated personal target. `None` is honest and
   the caller has a defined behaviour for it: report the observed average as a
   neutral observation and say nothing about whether it is too high or too low.

Q: "What happens to a user who does not want to state their sex?"
A: They leave the field empty. `estimate_tdee` then returns `None`, the calorie
   insight becomes a descriptive note instead of a verdict, and every macro
   insight is unaffected — those are ratios and need no body data. So the app is
   fully usable without disclosing anything, and no substitute value is invented
   on the user's behalf.

Q: "How is the tone requirement enforced rather than just intended?"
A: `BLOCKED_PHRASES` is asserted against every rendered insight in the M7 tests.
   The templates are the only text source, so passing that test on the templates
   means no code path can emit a blocked phrase.

COMMON STUDENT MISTAKES
--------------------------
- Putting "fat" in the blocked-phrase list. Fat is a macronutrient this app must
  be able to name; the blocked entries are deliberately multi-word judgement
  phrases ("too fat", "bad food") so they cannot collide with normal nutrition
  vocabulary.
- Letting the macro percentages drift out of sync with `06 §1`'s table. The
  `test_targets_sum_to_100` test catches arithmetic slips, not documentation
  drift — if the table in `06` changes, this file changes in the same commit.
- Mixing units: `MACRO_TARGETS` is percent-of-energy, NOT grams. Comparing a gram
  figure against these numbers produces nonsense that looks plausible.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.db.models import ActivityLevel, HealthGoal, Sex

# ---------------------------------------------------------------------------
# 1. Target macro distribution per health goal — % of total energy (06 §1)
# ---------------------------------------------------------------------------
# Reference proportions from widely accepted dietary guidelines; informational
# defaults, not medical prescriptions (cited as such in the report).
MACRO_TARGETS: dict[HealthGoal, dict[str, float]] = {
    HealthGoal.weight_loss: {"protein": 30.0, "carbs": 40.0, "fat": 30.0},
    HealthGoal.maintenance: {"protein": 20.0, "carbs": 50.0, "fat": 30.0},
    HealthGoal.weight_gain: {"protein": 25.0, "carbs": 50.0, "fat": 25.0},
}

MACROS: tuple[str, ...] = ("protein", "carbs", "fat")

# kcal per gram, used to convert grams <-> % of energy (Atwater factors).
KCAL_PER_G: dict[str, float] = {"protein": 4.0, "carbs": 4.0, "fat": 9.0}

# A macro within this many percentage points of its target counts as "aligned":
# it produces no corrective insight and becomes a candidate for the mandatory
# positive insight instead. Wide enough that normal day-to-day variation in a
# 7-day window is not reported as a problem.
MACRO_ALIGNED_TOLERANCE_PP = 5.0

# Slack for float comparisons against the tolerance above. The macro percentages
# are the result of several divisions, so a profile that is mathematically exactly
# on the boundary arrives as 5.000000000000004 and would fall on the wrong side of
# a bare `<=`. Far smaller than any difference a user could perceive.
FLOAT_TOLERANCE_EPSILON = 1e-9

# ---------------------------------------------------------------------------
# 2. Energy needs — Mifflin-St Jeor (only used when body metrics are present)
# ---------------------------------------------------------------------------
ACTIVITY_MULTIPLIERS: dict[ActivityLevel, float] = {
    ActivityLevel.sedentary: 1.200,
    ActivityLevel.light: 1.375,
    ActivityLevel.moderate: 1.550,
    ActivityLevel.active: 1.725,
    ActivityLevel.very_active: 1.900,
}

# Sex-specific constant in the Mifflin-St Jeor equation. A user who prefers not
# to state sex leaves the column NULL, which yields no TDEE and therefore no
# calorie verdict — the macro insights are unaffected either way.
SEX_CONSTANTS: dict[Sex, float] = {
    Sex.male: 5.0,
    Sex.female: -161.0,
}

# Mild by design (06 §5): a modest deficit for loss, a modest surplus for gain.
GOAL_CALORIE_POSTURE: dict[HealthGoal, float] = {
    HealthGoal.weight_loss: 0.85,
    HealthGoal.maintenance: 1.00,
    HealthGoal.weight_gain: 1.10,
}

# Hard floor on any calorie target this engine will ever state, regardless of
# what the TDEE estimate and posture multiply out to. A guard against bad input
# producing an unsafe-looking number, not a nutritional recommendation.
CALORIE_FLOOR_KCAL = 1200.0

# The observed average must differ from the goal target by more than this
# fraction before the engine comments on it — below that, the difference is well
# inside the estimate's own error and not worth reporting.
CALORIE_GAP_TOLERANCE_FRACTION = 0.12

# A daily calorie average needs to rest on at least this many separate logged
# days before the engine will compare it to an energy target. Macro percentages
# need no equivalent guard — they are ratios, so partial logging does not bias
# them — but an absolute daily figure drawn from a single day is not a daily
# average at all, and telling someone to eat more on that basis would be wrong.
MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT = 3


def estimate_bmr(
    *,
    weight_kg: float | None,
    height_cm: float | None,
    birth_year: int | None,
    sex: Sex | None,
) -> float | None:
    """
    Mifflin-St Jeor basal metabolic rate, or None if any input is missing.

        BMR = 10*weight_kg + 6.25*height_cm - 5*age_years + sex_constant
    """
    if weight_kg is None or height_cm is None or birth_year is None or sex is None:
        return None
    age_years = datetime.now(UTC).year - birth_year
    return 10.0 * float(weight_kg) + 6.25 * float(height_cm) - 5.0 * age_years + SEX_CONSTANTS[sex]


def estimate_tdee(
    *,
    weight_kg: float | None,
    height_cm: float | None,
    birth_year: int | None,
    sex: Sex | None,
    activity_level: ActivityLevel | None,
) -> float | None:
    """
    Total daily energy expenditure = BMR x activity multiplier.

    Returns None when the user has not supplied the body metrics — the caller
    must then avoid making any above/below claim about their intake.
    """
    bmr = estimate_bmr(weight_kg=weight_kg, height_cm=height_cm, birth_year=birth_year, sex=sex)
    if bmr is None or activity_level is None:
        return None
    return bmr * ACTIVITY_MULTIPLIERS[activity_level]


# User-facing names for the five body metrics, in the order they read naturally
# in a sentence. `health_goal` is deliberately absent: it is NOT NULL at
# registration, so it can never be one of the missing fields.
METRIC_LABELS: dict[str, str] = {
    "height_cm": "height",
    "weight_kg": "weight",
    "birth_year": "age",
    "sex": "sex",
    "activity_level": "activity level",
}


def missing_metric_labels(
    *,
    height_cm: float | None,
    weight_kg: float | None,
    birth_year: int | None,
    sex: Sex | None,
    activity_level: ActivityLevel | None,
) -> list[str]:
    """
    The user-facing labels of the body metrics still needed for a TDEE estimate.

    Empty means the profile is complete. This exists so the calorie note can name
    exactly what is missing: a user who has filled in four of five fields should
    be asked for the fifth, not handed the whole list again.
    """
    supplied = {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "birth_year": birth_year,
        "sex": sex,
        "activity_level": activity_level,
    }
    return [label for field, label in METRIC_LABELS.items() if supplied[field] is None]


def goal_target_calories(tdee: float | None, goal: HealthGoal) -> float | None:
    """
    Apply the goal's mild calorie posture to a TDEE estimate, clamped at
    CALORIE_FLOOR_KCAL so no goal can ever imply an extreme deficit (06 §5).
    """
    if tdee is None:
        return None
    return max(tdee * GOAL_CALORIE_POSTURE[goal], CALORIE_FLOOR_KCAL)


# ---------------------------------------------------------------------------
# 3. Curated examples per nutrient (06 §2 `{examples}`)
# ---------------------------------------------------------------------------
# Everyday, widely available, non-prescriptive. Used when the `food_item`
# nutrition cache has no better-scoring suggestion to offer.
FOOD_EXAMPLES: dict[str, tuple[str, ...]] = {
    "protein": ("chicken breast", "Greek yogurt", "lentils", "eggs", "tuna"),
    "carbs": ("oats", "brown rice", "whole-grain bread", "sweet potato"),
    "fat": ("olive oil", "avocado", "walnuts", "tahini"),
    "vegetables": ("leafy greens", "tomatoes", "cucumber", "peppers"),
}

# How many example foods to name in a single insight. Two or three reads as a
# helpful nudge; a longer list reads like a prescription.
MAX_EXAMPLES_PER_INSIGHT = 3

# --- Content-based scoring weights (06 §1 Stage 3) --------------------------
#   score(food) = w1*aligns_with_gap - w2*conflicts_other_macros - w3*already_frequent
# Every term is measured in percentage points of energy, so the weights are
# dimensionless and directly comparable to each other.
#
# The ratio is what matters, not the absolute values. Alignment leads at 1.0.
# Conflict is deliberately *half* of that rather than equal: a food that helps
# the primary gap should still win when it drags a secondary macro by a similar
# amount, because the primary gap is by definition the larger error — weighting
# them equally would produce a recommender that can only ever suggest perfectly
# balanced foods, which in practice means suggesting nothing.
W_ALIGNS_WITH_GAP = 1.0
W_CONFLICTS_OTHER_MACROS = 0.5
# Sized to be meaningful but never decisive: a strongly aligned food (20+ pp)
# still beats a weakly aligned novel one, while two similar foods break in favour
# of the one the user eats less often.
W_ALREADY_FREQUENT = 6.0

# The `already_frequent` penalty saturates here: beyond this many prior logs a
# food is simply "a regular", and scaling the penalty further would push the
# engine toward suggesting obscure foods purely for being unfamiliar.
FREQUENCY_SATURATION_LOGS = 5

# If generation produces fewer than this many insights, one behavioural insight
# is added (06 §4). Two exists so a balanced user with no body metrics does not
# land on an Insights screen holding a single lonely card.
MIN_INSIGHTS_BEFORE_PADDING = 2

# ---------------------------------------------------------------------------
# 4. Insight templates (06 §2) — the complete set of things this app can say
# ---------------------------------------------------------------------------
# Tone rules (06 §5): supportive, tentative ("could", "may"), never imperative,
# never evaluative about the user. Note there is no `fibre_low` template: the
# schema caches energy plus the three macros only, so a fibre claim would have
# no data behind it (documented in 06).
INSIGHT_TEMPLATES: dict[str, str] = {
    "protein_low": (
        "Your protein has been below your {goal} target ({actual}% of calories "
        "vs {target}%). Adding lean protein like {examples} to a couple of meals "
        "could help."
    ),
    "protein_high": (
        "Protein made up {actual}% of your calories, a little above your {goal} "
        "target of {target}%. There is room for more carbohydrates or healthy "
        "fats if you would like a bit more variety."
    ),
    "carbs_low": (
        "Carbohydrates were {actual}% of your calories, below your {goal} target "
        "of {target}%. Wholegrain options like {examples} are an easy way to add "
        "some back."
    ),
    "carbs_high": (
        "Carbohydrates made up {actual}% of your intake, a bit above your {goal} "
        "target of {target}%. Swapping some refined carbs for vegetables or whole "
        "grains may help balance this."
    ),
    "fat_low": (
        "Fat was {actual}% of your calories, below your {goal} target of "
        "{target}%. Sources like {examples} can help round this out."
    ),
    "fat_high": (
        "Fat was {actual}% of your intake, above your {goal} target of {target}%. "
        "Choosing grilled or baked options over fried a few times this week could "
        "bring this closer to your target."
    ),
    # Used when a TDEE estimate exists (body metrics supplied).
    # `{days}` receives an already-pluralised phrase ("7 days"), not a bare number,
    # so the copy never reads "1 days" if the window is ever reconfigured.
    "calorie_gap": (
        "Your average daily intake over the last {days} was about {kcal} kcal, "
        "which is {direction} the {target_kcal} kcal your {goal} goal suggests. A "
        "gradual adjustment tends to be easier to sustain than a large one."
    ),
    # Used when body metrics are missing: descriptive, no verdict. `{missing}` names
    # exactly the fields still needed, so following the suggestion actually unlocks
    # the verdict — naming a fixed subset would leave the user in a loop where they
    # add what was asked for and nothing changes.
    "calorie_note_profile": (
        "Over the last {days} your logged meals averaged about {kcal} kcal a day. "
        "Adding your {missing} in your profile lets these insights compare that to "
        "your own estimated needs."
    ),
    # Used when the profile IS complete but too few days were logged to call the
    # average a daily average. Must not mention the profile — there is nothing
    # left to fill in, and asking again would be confusing.
    "calorie_note_coverage": (
        "Your logged meals averaged about {kcal} kcal a day across the {days} you "
        "recorded. Logging on a few more days will give a clearer picture to "
        "compare against your estimated needs."
    ),
    "positive": (
        "Nice consistency — your {nutrient} intake is well aligned with your "
        "{goal} goal at {actual}% of calories. Keep it up."
    ),
    "positive_generic": (
        "You have logged {meals} meals recently — that consistency is the part "
        "most people find hardest, and it is what makes these insights possible."
    ),
    # 06 §4: emitted instead of a food list when no suitable foods can be named.
    "behavioural_balance": (
        "Your recent meals lean heavily on one part of the plate. Aiming for a "
        "mix of a protein, a wholegrain and some vegetables at most meals is a "
        "simple way to even things out."
    ),
    "behavioural_cooking": (
        "Cooking method makes a noticeable difference to the numbers. Grilling, "
        "baking or steaming a few of your usual meals keeps them closer to your "
        "{goal} target without changing what you eat."
    ),
    # The user has unlocked insights but logged nothing inside the window, so
    # there is no current pattern to describe. Not a locked state — they earned
    # the feature — and not a correction either, since there is no data to correct.
    "no_recent_meals": (
        "You have not logged any meals in the last {days}. Log a few and these "
        "insights will refresh with your latest eating pattern."
    ),
    "behavioural_variety": (
        "Your log shows a lot of the same foods. Rotating in a few alternatives "
        "makes it easier to hit your {goal} target across the week."
    ),
}

# Human-readable goal names for template interpolation.
GOAL_DISPLAY: dict[HealthGoal, str] = {
    HealthGoal.weight_loss: "weight loss",
    HealthGoal.maintenance: "maintenance",
    HealthGoal.weight_gain: "weight gain",
}

# `nutrient_focus` values persisted on the `recommendation` row (02: VARCHAR(30)).
FOCUS_BALANCE = "balance"
FOCUS_CALORIES = "calories"

# ---------------------------------------------------------------------------
# 5. Wellbeing guard rails (06 §5) — asserted by the M7 tone tests
# ---------------------------------------------------------------------------
# Deliberately multi-word judgement phrases, so they cannot collide with the
# ordinary nutrition vocabulary the app must be able to use ("fat", "calories").
BLOCKED_PHRASES: tuple[str, ...] = (
    "bad food",
    "bad foods",
    "junk food",
    "cheat meal",
    "cheat day",
    "guilty",
    "guilt",
    "shame",
    "shameful",
    "ashamed",
    "starve",
    "starving",
    "burn off",
    "work it off",
    "punish",
    "detox",
    "too fat",
    "overweight",
    "obese",
    "you failed",
    "you should stop",
    "never eat",
    "avoid completely",
    "cut out",
    "forbidden",
    "must lose",
    "lose weight fast",
)

# Insight count ceiling from FR12 / 06 §1.
MAX_INSIGHTS = 5
