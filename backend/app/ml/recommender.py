"""
app/ml/recommender.py
========================

WHY THIS FILE EXISTS
---------------------
This is the recommendation engine's arithmetic: the three stages of `06 §1`
turned into functions. Stage 1 collapses a user's recent meals into a single
macro profile, Stage 2 diffs that profile against the target distribution for
their health goal, and Stage 3 turns the resulting gaps into plain-language
insights — selecting example foods from the user's own nutrition cache when it
has usable data, and from a curated list when it does not.

Everything here is a **pure function over plain values** — no `Session`, no
`settings`, no clock reads except through an injected `now`. The database work
lives in `services/recommendation_service.py`, which loads rows, converts them
to the small dataclasses below, and calls into here.

WHAT PROBLEM IT SOLVES
------------------------
`06 §6` asks for the engine to be evaluated by "synthetic profiles with a known
dominant gap must yield the matching nutrient_focus". That evaluation is only
cheap if the maths is reachable without a database — otherwise every test case
needs a user, seven meal rows, seven meal_items and a food_item per item just to
assert one percentage. Keeping I/O out of this module turns the report's
evaluation section into a parametrised unit test.

HOW IT WORKS INTERNALLY
--------------------------
Stage 1 (`aggregate_profile`):
  1. Filter to meals inside the window and drop the ones `06 §4` excludes
     (null/zero calories — a meal with no nutrition attached carries no signal
     and would drag the average toward zero).
  2. Sum grams and calories across the surviving meals.
  3. Divide by the number of **active days** (distinct calendar dates with a
     logged meal), not by the window length. See the design note below.
  4. Convert grams to % of energy using the Atwater factors in `reco_targets`.

Stage 2 (`compute_gaps`): subtract each target percentage from the observed one,
producing a signed gap in **percentage points**, then sort by absolute magnitude.
The largest is the primary `nutrient_focus`. Gaps within
`MACRO_ALIGNED_TOLERANCE_PP` are labelled `aligned` and become candidates for the
mandatory positive insight rather than a corrective one.

TWO DESIGN DECISIONS THAT DEVIATE FROM A LITERAL READING OF `06 §1`
----------------------------------------------------------------------
1. **Percentages are computed against macro-derived energy, not reported
   calories.** `06 §1` writes `p_pct = protein_g x 4 / calories`. Using the
   *reported* calorie figure as the denominator is subtly broken: reported
   calories rarely equal 4p + 4c + 9f exactly (rounding in the USDA source,
   fibre and alcohol contributing differently), so the three percentages sum to
   something like 94% or 106% rather than 100%. Since the targets sum to exactly
   100%, that mismatch shows up as a *systematic phantom gap in every macro at
   once* — a user eating a perfectly balanced diet would be told all three macros
   are low. Dividing by `4p + 4c + 9f` forces the percentages to sum to 100 and
   makes the gaps mean what Stage 2 assumes they mean. The reported calorie total
   is still kept, separately, for the calorie insight, which is the only consumer
   that needs an absolute number.

2. **The daily average divides by active days, not by the 7-day window.**
   Dividing a week's logged food by 7 assumes the user logged every day; someone
   who logs Monday and Tuesday then stops would appear to eat 2/7 of what they
   actually ate, and the calorie insight would tell them to eat more. Dividing by
   active days answers the honest question — "on the days you logged, what did a
   day look like?" Note this choice cannot bias the macro percentages at all:
   they are ratios, so the denominator cancels. It only affects the absolute
   calorie figure, which is why `active_days` is carried on the profile so Stage
   3 can decline to make a calorie verdict when coverage is too thin
   (`MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Frozen dataclasses (chosen) over dicts for the profile and gaps: the field
  names are checked by mypy, so a typo in `actual_pct` is a build error rather
  than a `KeyError` on a live request. They are also immutable, so Stage 3 cannot
  quietly mutate the profile it was handed.
- Pydantic models for these internal shapes: unnecessary validation cost on data
  that never crosses a trust boundary — it comes from our own database.
- Returning `None` from `aggregate_profile` when nothing usable survives the
  filter, rather than a zero-filled profile: a zeroed profile would flow through
  Stage 2 and produce maximal gaps in every macro, generating confident advice
  from no data at all.
- An injected `now` rather than calling `datetime.now()` inside: makes the window
  boundary testable, which matters because "the last 7 days" is exactly the kind
  of off-by-one that only shows up at month ends.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why percentage points rather than a ratio for the gap?"
A: The comparison is between two percentages, so their difference is in
   percentage points, and the tolerance in `06` is naturally expressed the same
   way ("within 5 points of target"). A ratio would make a 2%-vs-4% difference
   look as severe as 25%-vs-50%.

Q: "What happens if a user logs one enormous meal and nothing else?"
A: It passes the filter (it has calories), `active_days` is 1, and the macro
   percentages describe that meal honestly. The 7-meal unlock gate in the service
   layer is what stops a single meal from ever reaching this function in the
   first place — this module is deliberately not the place that enforces it.

Q: "Why is the window filter here rather than in the SQL query?"
A: It is in both, and intentionally. The service narrows the query for
   efficiency; this function re-filters because it must be correct when called
   directly from a test with a hand-built list, and because a pure function that
   trusts its caller to have already filtered is a trap.

COMMON STUDENT MISTAKES
--------------------------
- Averaging the per-meal percentages instead of computing percentages from the
  summed grams. Those are different numbers: the mean of ratios is not the ratio
  of means, and the former lets one tiny 50-kcal snack count as much as a 900-kcal
  dinner.
- Treating a missing macro as "unknown" but still counting the meal's calories,
  or vice versa — the two must be filtered together or the percentages are drawn
  from a different set of meals than the calorie average.
- Comparing timezone-aware and naive datetimes. Postgres `TIMESTAMP` columns come
  back naive here, so `_as_naive_utc` normalises everything before any comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.db.models import HealthGoal
from app.ml import reco_targets as rt

# ---------------------------------------------------------------------------
# Inputs and outputs — the small value types the three stages pass around
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MealMacros:
    """
    One logged meal, reduced to the only five things Stage 1 needs.

    The service layer builds these from `meal_log` + summed `meal_item` rows;
    tests build them by hand. `None` in any nutrition field means "not known",
    not "zero" — the filter in `aggregate_profile` drops such meals rather than
    treating absent data as an absence of nutrients.
    """

    logged_at: datetime
    calories: float | None
    protein_g: float | None
    carbs_g: float | None
    fat_g: float | None


@dataclass(frozen=True)
class MacroProfile:
    """The user's recent eating pattern — the output of Stage 1 (`06 §1`)."""

    # Reported energy, averaged over active days. The only absolute figure here,
    # and the only one whose value depends on the denominator choice.
    avg_daily_calories: float
    # Grams per active day, per macro.
    avg_daily_grams: dict[str, float]
    # Share of macro-derived energy, per macro. Sums to 100.0 by construction.
    macro_pct: dict[str, float]
    # Provenance, so Stage 3 can judge how much weight the numbers deserve.
    meals_counted: int
    active_days: int
    window_days: int

    @property
    def has_enough_coverage_for_calorie_verdict(self) -> bool:
        """
        Whether the calorie average rests on enough separate days to compare
        against an energy target. Percentages need no such guard (they are
        ratios); an absolute daily figure drawn from one day does.
        """
        return self.active_days >= rt.MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT


@dataclass(frozen=True)
class MacroGap:
    """One macro's signed distance from its target — the output of Stage 2."""

    macro: str
    actual_pct: float
    target_pct: float
    # Signed, in percentage points: actual - target. Negative means below target.
    gap_pp: float

    @property
    def is_aligned(self) -> bool:
        """
        Whether this macro is close enough to target to count as on track.

        The epsilon makes the inclusive boundary documented on
        `MACRO_ALIGNED_TOLERANCE_PP` robust: `actual_pct` is the result of a chain
        of float divisions in Stage 1, so a profile that is mathematically exactly
        at the tolerance can land a fraction above it and be classified off-target
        on rounding error alone. The slack is nine orders of magnitude below any
        difference that could change an insight, so it cannot widen the threshold
        in any way a user would notice.
        """
        return abs(self.gap_pp) <= rt.MACRO_ALIGNED_TOLERANCE_PP + rt.FLOAT_TOLERANCE_EPSILON

    @property
    def direction(self) -> str:
        """`aligned`, `low` or `high` — selects the template in Stage 3."""
        if self.is_aligned:
            return "aligned"
        return "low" if self.gap_pp < 0 else "high"

    @property
    def template_key(self) -> str:
        """e.g. `protein_low` — the key into `rt.INSIGHT_TEMPLATES`."""
        return f"{self.macro}_{self.direction}"


# ---------------------------------------------------------------------------
# Stage 1 — aggregate the profile vector (06 §1)
# ---------------------------------------------------------------------------


def _as_naive_utc(value: datetime) -> datetime:
    """
    Normalise to naive UTC so DB-sourced naive timestamps and caller-supplied
    aware ones can be compared. Aware values are *converted* to UTC before the
    tzinfo is dropped — truncating instead would shift a +03:00 timestamp three
    hours into the future and could pull a meal across the window boundary.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def is_usable(meal: MealMacros) -> bool:
    """
    `06 §4`: skip meals whose `total_calories` is null or 0 — a meal with no
    nutrition attached carries no dietary signal, and counting it would pull
    every average toward zero.

    A meal that has calories but no macro breakdown at all is dropped for the
    same reason: its calories would enter the energy average while contributing
    nothing to the gram totals, understating every macro percentage.

    Negative values are also rejected. Nothing in the normal pipeline can produce
    one (USDA nutrition is non-negative, portion grams are positive), so a
    negative macro means a corrupt row — but if one survives to Stage 1 it does
    not merely skew the average, it makes the percentages *negative*, and the
    engine renders that straight into user-facing text: "your protein has been
    below your maintenance target (-40% of calories vs 20%)". Dropping the row is
    the only outcome that cannot put nonsense in front of a user.
    """
    if meal.calories is None or float(meal.calories) <= 0:
        return False
    macros = (meal.protein_g, meal.carbs_g, meal.fat_g)
    if any(m is not None and float(m) < 0 for m in macros):
        return False
    return any(m is not None and float(m) > 0 for m in macros)


def select_window_meals(
    meals: list[MealMacros], *, window_days: int, now: datetime
) -> list[MealMacros]:
    """
    Meals inside the rolling window that survive the `06 §4` usability filter.

    The window is bounded at *both* ends. The upper bound looks redundant — how can
    a logged meal be in the future? — but `logged_at` is not necessarily the
    server's own clock: `POST /meals` may accept a client-supplied timestamp, and a
    device with a skewed clock or a mistyped date would otherwise contribute a meal
    to a window it does not belong in, and inflate `active_days` with a day that
    has not happened yet.
    """
    reference = _as_naive_utc(now)
    cutoff = reference - timedelta(days=window_days)
    return [
        meal
        for meal in meals
        if is_usable(meal) and cutoff <= _as_naive_utc(meal.logged_at) <= reference
    ]


def aggregate_profile(
    meals: list[MealMacros], *, window_days: int, now: datetime
) -> MacroProfile | None:
    """
    Stage 1. Collapse recent meals into one profile, or `None` if no meal in the
    window carries usable nutrition (the caller must not invent a profile).
    """
    counted = select_window_meals(meals, window_days=window_days, now=now)
    if not counted:
        return None

    total_calories = sum(float(meal.calories or 0.0) for meal in counted)
    total_grams = {
        "protein": sum(float(meal.protein_g or 0.0) for meal in counted),
        "carbs": sum(float(meal.carbs_g or 0.0) for meal in counted),
        "fat": sum(float(meal.fat_g or 0.0) for meal in counted),
    }

    # Energy implied by the macros themselves. This, not the reported calorie
    # total, is the denominator for the percentages — see the module docstring.
    macro_energy = sum(total_grams[macro] * rt.KCAL_PER_G[macro] for macro in rt.MACROS)
    if macro_energy <= 0:
        return None

    active_days = len({_as_naive_utc(meal.logged_at).date() for meal in counted})

    return MacroProfile(
        avg_daily_calories=total_calories / active_days,
        avg_daily_grams={macro: total_grams[macro] / active_days for macro in rt.MACROS},
        macro_pct={
            macro: 100.0 * total_grams[macro] * rt.KCAL_PER_G[macro] / macro_energy
            for macro in rt.MACROS
        },
        meals_counted=len(counted),
        active_days=active_days,
        window_days=window_days,
    )


# ---------------------------------------------------------------------------
# Stage 2 — compare against the goal's target distribution (06 §1)
# ---------------------------------------------------------------------------


def compute_gaps(profile: MacroProfile, goal: HealthGoal) -> list[MacroGap]:
    """
    Stage 2. Signed gap per macro in percentage points, sorted by absolute
    magnitude so `gaps[0]` is the primary nutritional focus.

    Ties break on the fixed `rt.MACROS` order rather than on dict iteration
    order, so the same profile always yields the same focus (an insight that
    changes on refresh with no new meals reads as a bug to the user).
    """
    targets = rt.MACRO_TARGETS[goal]
    gaps = [
        MacroGap(
            macro=macro,
            actual_pct=profile.macro_pct[macro],
            target_pct=targets[macro],
            gap_pp=profile.macro_pct[macro] - targets[macro],
        )
        for macro in rt.MACROS
    ]
    return sorted(gaps, key=lambda gap: (-abs(gap.gap_pp), rt.MACROS.index(gap.macro)))


def primary_focus(gaps: list[MacroGap]) -> str:
    """
    The `nutrient_focus` persisted on the `recommendation` row (`02`, `03 §7`).

    The macro with the largest absolute gap, or `balance` when every macro is
    within tolerance — in that case there is no single nutrient to focus on and
    labelling one would misrepresent a well-balanced week.
    """
    if not gaps:
        return rt.FOCUS_BALANCE
    if all(gap.is_aligned for gap in gaps):
        return rt.FOCUS_BALANCE
    return gaps[0].macro


def actionable_gaps(gaps: list[MacroGap], limit: int = 3) -> list[MacroGap]:
    """
    The top 1-3 gaps worth an insight (`06 §1` Stage 3), already ranked. Aligned
    macros are excluded — they are the positive insight's material, not a
    problem to solve.
    """
    return [gap for gap in gaps if not gap.is_aligned][:limit]


def aligned_gaps(gaps: list[MacroGap]) -> list[MacroGap]:
    """
    Macros on target, best-aligned first. `06 §2` requires at least one
    positive/affirming insight whenever the user is on track for a macro.
    """
    return sorted((gap for gap in gaps if gap.is_aligned), key=lambda gap: abs(gap.gap_pp))


# ---------------------------------------------------------------------------
# Stage 3 — generate the insights (06 §1, §2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoodCandidate:
    """
    One row of the `food_item` nutrition cache, as Stage 3 needs it.

    `times_logged` is how often this user has already logged the food — the
    `already_frequent` term of the `06 §1` scoring function. Suggesting someone
    eat more of the thing they eat every day is not a recommendation.
    """

    food_name: str
    calories_per_100g: float | None
    protein_per_100g: float | None
    carbs_per_100g: float | None
    fat_per_100g: float | None
    times_logged: int = 0

    @property
    def is_scorable(self) -> bool:
        """
        Whether this row has enough nutrition data to score. The `food_item`
        table is seeded with 101 Food-101 class names whose macro columns are all
        NULL until the USDA client (M5) fills them, so in practice most rows are
        unscorable and Stage 3 falls back to the curated examples.
        """
        macros = (self.protein_per_100g, self.carbs_per_100g, self.fat_per_100g)
        return any(macro is not None and float(macro) > 0 for macro in macros)

    @property
    def macro_pct(self) -> dict[str, float]:
        """
        This food's own macro split as % of its macro-derived energy — the same
        basis as `MacroProfile.macro_pct`, so the two are directly comparable.
        """
        grams = {
            "protein": float(self.protein_per_100g or 0.0),
            "carbs": float(self.carbs_per_100g or 0.0),
            "fat": float(self.fat_per_100g or 0.0),
        }
        energy = sum(grams[macro] * rt.KCAL_PER_G[macro] for macro in rt.MACROS)
        if energy <= 0:
            return dict.fromkeys(rt.MACROS, 0.0)
        return {macro: 100.0 * grams[macro] * rt.KCAL_PER_G[macro] / energy for macro in rt.MACROS}


@dataclass(frozen=True)
class Insight:
    """One generated insight, ready to persist as a `recommendation` row."""

    insight_text: str
    nutrient_focus: str


def contains_blocked_phrase(text: str) -> str | None:
    """
    The first `06 §5` blocked phrase in `text`, or None. Used both by the tone
    tests and as a runtime filter in `generate_insights` — see the note there for
    why a static template set still gets a runtime check.
    """
    lowered = text.lower()
    for phrase in rt.BLOCKED_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def score_food(
    candidate: FoodCandidate,
    *,
    gap: MacroGap,
    profile: MacroProfile,
    goal: HealthGoal,
) -> float:
    """
    The content-based score from `06 §1` Stage 3:

        score = w1*aligns_with_gap - w2*conflicts_other_macros - w3*already_frequent

    All three terms are in percentage points of energy, so the weights are
    dimensionless and comparable:

    - `aligns_with_gap`: how far this food's share of the gap macro sits *in the
      corrective direction* relative to what the user is currently eating. A food
      only helps a protein deficit if it is more protein-dense than the user's
      current average — an 18%-protein food does nothing for someone already at
      20%, even though 18% sounds respectable in isolation.
    - `conflicts_other_macros`: penalises pushing an already-wrong macro further
      wrong. Without it, a deep-fried protein would score well on a protein
      deficit while worsening an existing fat surplus.
    - `already_frequent`: saturating penalty on foods the user logs often.
    """
    food_pct = candidate.macro_pct

    # A negative gap means "below target", so more of this macro helps and a food
    # denser than the user's current average scores positively. A positive gap
    # inverts that: the helpful food is the one *less* dense than they eat now.
    corrective_sign = 1.0 if gap.gap_pp < 0 else -1.0
    aligns = corrective_sign * (food_pct[gap.macro] - profile.macro_pct[gap.macro])

    targets = rt.MACRO_TARGETS[goal]
    conflicts = 0.0
    for macro in rt.MACROS:
        if macro == gap.macro:
            continue
        # Only movement that makes another macro's existing error *worse* counts.
        # Same sign on both terms means the food drifts further in the direction
        # the user is already wrong in: over on fat, and more fat-dense still.
        other_error = profile.macro_pct[macro] - targets[macro]
        drift = food_pct[macro] - profile.macro_pct[macro]
        if other_error * drift > 0:
            conflicts += abs(drift)

    frequency = min(candidate.times_logged, rt.FREQUENCY_SATURATION_LOGS) / (
        rt.FREQUENCY_SATURATION_LOGS or 1
    )

    return (
        rt.W_ALIGNS_WITH_GAP * aligns
        - rt.W_CONFLICTS_OTHER_MACROS * conflicts
        - rt.W_ALREADY_FREQUENT * frequency
    )


def select_foods(
    candidates: list[FoodCandidate],
    *,
    gap: MacroGap,
    profile: MacroProfile,
    goal: HealthGoal,
    limit: int = rt.MAX_EXAMPLES_PER_INSIGHT,
) -> list[str]:
    """
    The best-scoring foods to name for this gap, or `[]` when the cache offers
    nothing that would actually help — which is `06 §4`'s "no suitable foods"
    case and sends the caller to the curated examples or a behavioural insight.

    Only strictly positive scores qualify: a food that scores zero or negative
    would move the user the wrong way, and naming it would be worse than naming
    nothing.
    """
    scored = [
        (score_food(candidate, gap=gap, profile=profile, goal=goal), candidate)
        for candidate in candidates
        if candidate.is_scorable
    ]
    helpful = [(score, candidate) for score, candidate in scored if score > 0]
    # Sort by score, then name, so equal scores order deterministically rather
    # than by whatever order the query returned.
    helpful.sort(key=lambda pair: (-pair[0], pair[1].food_name))
    return [candidate.food_name for _, candidate in helpful[:limit]]


def _humanise_list(items: list[str], *, conjunction: str = "or") -> str:
    """
    `[a, b, c]` -> "a, b or c".

    The conjunction carries real meaning here. Food examples take "or" because
    any one of them helps — they are alternatives, and "and" would read as an
    instruction to eat all three. Missing profile fields take "and" because every
    one is required for a TDEE estimate, and "or" would imply supplying any single
    field is enough.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _pluralise(count: int, noun: str) -> str:
    """`1 day` / `3 days` — user-facing copy should not read "1 days"."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _examples_for(
    gap: MacroGap,
    profile: MacroProfile,
    candidates: list[FoodCandidate],
    goal: HealthGoal,
) -> str:
    """
    Fill `{examples}` for a gap: prefer foods scored from the user's own
    nutrition cache, fall back to the curated list in `reco_targets`.

    The fallback is the normal path today, not an error case — `food_item` holds
    the 101 seeded class names with NULL macros until M5's USDA client populates
    them, so there is usually nothing scorable to choose from.
    """
    chosen = select_foods(candidates, gap=gap, profile=profile, goal=goal)
    if not chosen:
        chosen = list(rt.FOOD_EXAMPLES.get(gap.macro, ()))[: rt.MAX_EXAMPLES_PER_INSIGHT]
    return _humanise_list(chosen)


def _macro_insight(
    gap: MacroGap,
    *,
    profile: MacroProfile,
    goal: HealthGoal,
    candidates: list[FoodCandidate],
) -> Insight | None:
    """Render one corrective insight for a macro that is off target."""
    template = rt.INSIGHT_TEMPLATES.get(gap.template_key)
    if template is None:
        return None
    text = template.format(
        goal=rt.GOAL_DISPLAY[goal],
        actual=round(gap.actual_pct),
        target=round(gap.target_pct),
        examples=_examples_for(gap, profile, candidates, goal),
    )
    return Insight(insight_text=text, nutrient_focus=gap.macro)


def _calorie_insight(
    profile: MacroProfile,
    *,
    goal: HealthGoal,
    calorie_target: float | None,
    missing_metrics: list[str] | None = None,
) -> Insight | None:
    """
    The calorie line. Four outcomes, in order of how much the data supports:

    1. No `calorie_target` because body metrics are missing -> a descriptive note
       that names *exactly which* metrics are still needed. No verdict: there is
       nothing to compare against, and inventing a comparison is what `06 §5`
       forbids. Naming the specific gaps matters — a note listing a fixed subset
       would let a user add everything it asked for and still see the same note.
    2. No `calorie_target` but the profile looks complete (an unexpected
       combination, e.g. a metric outside the plausible range) -> the coverage
       wording, which asks for nothing the user cannot give.
    3. A target exists but coverage is too thin
       (`MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT`) -> the coverage note. Deliberately
       does NOT mention the profile: it is already complete, and asking again
       would be confusing. An "average" over one logged day is not an average.
    4. A target exists, coverage is adequate, and the difference exceeds
       `CALORIE_GAP_TOLERANCE_FRACTION` -> the actual above/below insight. Inside
       that tolerance the difference is smaller than the estimate's own error, so
       the engine says nothing and leaves room for a more useful insight.
    """
    kcal = round(profile.avg_daily_calories)

    if calorie_target is None:
        # `None` means "caller did not tell us", which for a missing target means
        # no metrics were supplied at all — so name all of them.
        if missing_metrics is None:
            missing_metrics = list(rt.METRIC_LABELS.values())
        if missing_metrics:
            return Insight(
                insight_text=rt.INSIGHT_TEMPLATES["calorie_note_profile"].format(
                    days=_pluralise(profile.window_days, "day"),
                    kcal=kcal,
                    # "and", not "or": every field is required for an estimate.
                    missing=_humanise_list(missing_metrics, conjunction="and"),
                ),
                nutrient_focus=rt.FOCUS_CALORIES,
            )

    if calorie_target is None or not profile.has_enough_coverage_for_calorie_verdict:
        return Insight(
            insight_text=rt.INSIGHT_TEMPLATES["calorie_note_coverage"].format(
                days=_pluralise(profile.active_days, "day"), kcal=kcal
            ),
            nutrient_focus=rt.FOCUS_CALORIES,
        )

    deviation = (profile.avg_daily_calories - calorie_target) / calorie_target
    if abs(deviation) <= rt.CALORIE_GAP_TOLERANCE_FRACTION:
        return None

    return Insight(
        insight_text=rt.INSIGHT_TEMPLATES["calorie_gap"].format(
            days=_pluralise(profile.window_days, "day"),
            kcal=kcal,
            target_kcal=round(calorie_target),
            direction="above" if deviation > 0 else "below",
            goal=rt.GOAL_DISPLAY[goal],
        ),
        nutrient_focus=rt.FOCUS_CALORIES,
    )


def _positive_insight(gaps: list[MacroGap], *, profile: MacroProfile, goal: HealthGoal) -> Insight:
    """
    The mandatory affirming insight (`06 §2`). Prefers naming a macro the user is
    actually on target for; when nothing is aligned it credits the logging
    consistency instead, because there is always something true and supportive to
    say and a purely deficit-framed list is what `06 §5` rules out.
    """
    aligned = aligned_gaps(gaps)
    if aligned:
        best = aligned[0]
        return Insight(
            insight_text=rt.INSIGHT_TEMPLATES["positive"].format(
                nutrient=best.macro,
                goal=rt.GOAL_DISPLAY[goal],
                actual=round(best.actual_pct),
            ),
            nutrient_focus=best.macro,
        )
    return Insight(
        insight_text=rt.INSIGHT_TEMPLATES["positive_generic"].format(meals=profile.meals_counted),
        nutrient_focus=rt.FOCUS_BALANCE,
    )


def _behavioural_insight(gaps: list[MacroGap], *, goal: HealthGoal) -> Insight:
    """
    `06 §4`'s fallback: advice about *how* to eat rather than *what* to eat, for
    when there is no useful food list to give. Chosen by what the gaps show, so
    it is still responsive to the user's data rather than filler.
    """
    high_fat = any(gap.macro == "fat" and gap.direction == "high" for gap in gaps)
    key = "behavioural_cooking" if high_fat else "behavioural_balance"
    text = rt.INSIGHT_TEMPLATES[key]
    if "{goal}" in text:
        text = text.format(goal=rt.GOAL_DISPLAY[goal])
    return Insight(insight_text=text, nutrient_focus=rt.FOCUS_BALANCE)


def generate_insights(
    profile: MacroProfile,
    *,
    goal: HealthGoal,
    calorie_target: float | None = None,
    candidates: list[FoodCandidate] | None = None,
    missing_metrics: list[str] | None = None,
) -> list[Insight]:
    """
    Stage 3. Turn a profile and its gaps into at most `rt.MAX_INSIGHTS` (5)
    plain-language insights, ordered most to least actionable.

    Composition, which is what keeps the total at or under 5 by construction
    rather than by truncation:

        <= 3 corrective (one per actionable gap, ranked)
         + <= 1 calorie
         + exactly 1 positive          = <= 5

    A behavioural insight is added only when the result would otherwise be
    almost empty (a user with a balanced week and no body metrics), so the
    Insights screen never shows a single lonely card.

    The blocked-phrase filter at the end is redundant against today's static
    templates — the tone tests already prove none of them can produce one. It
    stays because it is the only place that is still true if someone later adds a
    template, or interpolates a food name from the USDA cache, that contains
    something it shouldn't. Failing closed (dropping the insight) is correct here:
    a missing insight is a cosmetic loss, a shaming one is a wellbeing failure.
    """
    candidates = candidates or []
    gaps = compute_gaps(profile, goal)

    insights: list[Insight] = []
    for gap in actionable_gaps(gaps):
        insight = _macro_insight(gap, profile=profile, goal=goal, candidates=candidates)
        if insight is not None:
            insights.append(insight)

    calorie = _calorie_insight(
        profile,
        goal=goal,
        calorie_target=calorie_target,
        missing_metrics=missing_metrics,
    )
    if calorie is not None:
        insights.append(calorie)

    insights.append(_positive_insight(gaps, profile=profile, goal=goal))

    if len(insights) < rt.MIN_INSIGHTS_BEFORE_PADDING:
        insights.append(_behavioural_insight(gaps, goal=goal))

    safe = [
        insight for insight in insights if contains_blocked_phrase(insight.insight_text) is None
    ]
    return safe[: rt.MAX_INSIGHTS]
