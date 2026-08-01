# 06 — Recommendation Engine (Content-Based Filtering)

Implements **FR12**. Generates ≤ 5 personalised, plain-language dietary insights from the user's own
accumulated meal history, aligned to their health goal. Activates only after **≥ 7 confirmed meals**
(FR10). Runs **asynchronously** off stored data — never on the `/recognise` request path.

**Every output carries the medical disclaimer** (`03 §7`, NFR).

Module: `app/ml/recommender.py`, called by `app/services/recommendation_service.py`.

---

## 1. Algorithm — three stages (matches report §5.3.6)

```
Stage 1  AGGREGATE     → build the user's daily macro profile vector (7-day rolling window)
Stage 2  COMPARE       → diff profile vs. target distribution for the user's health goal → find gap
Stage 3  GENERATE      → pick ≤5 foods/behaviours that close the largest gap → plain-language insights
```

### Stage 1 — Aggregate the profile vector
- Window: confirmed meals in the last **`RECOMMENDATION_WINDOW_DAYS` = 7** days (config), bounded
  at **both** ends — a `logged_at` in the future (clock skew, mistyped date) is excluded.
- Guard: if `count(confirmed meals) < MIN_MEALS_FOR_RECOMMENDATION (7)` → return **locked** (not error).
  The count is over **all** confirmed meals, not only the window (see §3).
- Skip unusable meals (§4): null/zero calories, no macro breakdown at all, or any negative value.
- Compute the **daily average** macro vector over the window:
  `profile = mean_per_active_day(calories, protein_g, carbs_g, fat_g)`.
- Convert protein/carbs/fat to **% of energy** for comparability:
  `p_pct = protein_g×4 / E`, `c_pct = carbs_g×4 / E`, `f_pct = fat_g×9 / E`.

> **Two implementation refinements to the formulas above.** Both were found while building
> `app/ml/recommender.py` and both change results, so they are recorded here rather than left in
> code comments.
>
> **(a) The denominator `E` is macro-derived energy (`4p + 4c + 9f`), not the reported calorie
> total.** Reported calories rarely equal `4p + 4c + 9f` exactly — USDA rounding, and fibre and
> alcohol contributing differently — so dividing by the reported figure makes the three
> percentages sum to something like 94% or 106%. Since the *targets* sum to exactly 100%, that
> mismatch appears as a **systematic phantom gap in all three macros at once**: a user eating a
> perfectly balanced diet would be told protein, carbs *and* fat are all low. Using `4p + 4c + 9f`
> forces the sum to 100 so the Stage 2 subtraction means what it claims. The reported calorie
> total is kept separately for the calorie insight, the only consumer needing an absolute number.
>
> **(b) "Per day" means per *active* day — a distinct calendar date with a logged meal — not per
> window day.** Dividing a week's food by 7 assumes daily logging: someone who logs Monday and
> Tuesday then stops would appear to eat 2/7 of what they actually ate, and the engine would tell
> them to eat more. Note this cannot bias the macro percentages, which are ratios where the
> denominator cancels (unit-tested); it affects only the absolute kcal figure. The profile
> therefore carries `active_days`, and Stage 3 declines to give a calorie verdict below
> `MIN_ACTIVE_DAYS_FOR_CALORIE_VERDICT` (3) — an "average" over one day is not an average.

### Stage 2 — Compare to target distribution
Target macro distributions (`% of energy`) per goal, stored in `app/ml/reco_targets.py`. These are
**reference proportions from widely accepted dietary guidelines** (state the source in the report;
they are informational defaults, not medical prescriptions):

| Health goal | Protein % | Carbs % | Fat % | Calorie posture |
|-------------|-----------|---------|-------|-----------------|
| weight_loss | 30 | 40 | 30 | ×0.85 of TDEE (mild deficit) |
| maintenance | 20 | 50 | 30 | ×1.00 of TDEE |
| weight_gain | 25 | 50 | 25 | ×1.10 of TDEE (mild surplus) |

Compute the **signed gap** per macro in **percentage points**: `gap_x = profile_pct_x −
target_pct_x`. The macro with the **largest absolute gap** is the **primary nutritional focus**
(`nutrient_focus` in `02`/`03`). Gaps within `MACRO_ALIGNED_TOLERANCE_PP` (5 pp) count as
**aligned** — no corrective insight, and eligible for the positive insight instead.

> Because the profile percentages and the targets both sum to 100, **the three gaps always sum to
> zero.** A macro cannot be made the dominant gap in isolation; the counterpart surplus is always
> distributed across the other two. This is worth stating explicitly because it shapes how the
> sanity tests in §6 must be constructed.

**The calorie posture requires a per-user energy estimate**, which the M1 schema could not
provide. `user` therefore gained five nullable body metrics in migration `0002`
(`02 §2.1`), and `estimate_tdee()` computes **Mifflin-St Jeor** BMR × an activity multiplier:

```
BMR  = 10·weight_kg + 6.25·height_cm − 5·age_years + s      (s = +5 male, −161 female)
TDEE = BMR × activity_multiplier                            (1.2 … 1.9)
target_kcal = max(TDEE × posture, CALORIE_FLOOR_KCAL)       (floor = 1200)
```

Two safety properties, both unit-tested:
- **The floor is absolute.** No combination of a low TDEE and a deficit posture can make the engine
  state a target below `CALORIE_FLOOR_KCAL`, so §5's "never recommend extreme deficits" is enforced
  by arithmetic rather than by intent.
- **Missing metrics yield `None`, never a default.** A substituted "average body" would be a
  fabricated personal target. `None` routes Stage 3 to a descriptive note instead of a verdict.

### Stage 3 — Generate insights
- Rank gaps by magnitude; take the top 1–3 gaps.
- For each, select foods from the `food_item` table whose macro profile would **move the user toward
  target** (e.g. protein below target → suggest high-protein, moderate-calorie foods the user hasn't
  logged much). Selection = simple content-based score:
  `score(food) = w1·aligns_with_gap(food) − w2·conflicts_other_macros(food) − w3·already_frequent(food)`.
- Emit **plain-language** insight strings from templates (see §2), max 5 total.
- Persist each as a `recommendation` row (`insight_text`, `nutrient_focus`, `generated_at`).

## 2. Insight templates (tone: supportive, never alarming — see wellbeing note §5)

Every template lives in `app/ml/reco_targets.py` — the complete set of sentences this app can ever
say about a user's diet, in one reviewable file. There is a `_low` and a `_high` variant for each
of the three macros (six corrective templates), plus:

```
calorie_gap:            "Your average daily intake over the last {days} was about {kcal} kcal,
                         which is {direction} the {target_kcal} kcal your {goal} goal suggests.
                         A gradual adjustment tends to be easier to sustain than a large one."
calorie_note_profile:   "Over the last {days} your logged meals averaged about {kcal} kcal a day.
                         Adding your {missing} in your profile lets these insights compare that
                         to your own estimated needs."
calorie_note_coverage:  "Your logged meals averaged about {kcal} kcal a day across the {days} you
                         recorded. Logging on a few more days will give a clearer picture..."
positive:               "Nice consistency — your {nutrient} intake is well aligned with your
                         {goal} goal at {actual}% of calories. Keep it up."
positive_generic:       "You have logged {meals} meals recently — that consistency is the part
                         most people find hardest..."
no_recent_meals:        "You have not logged any meals in the last {days}..."
behavioural_*:          balance / cooking / variety — the §4 fallback
```

Always include at least **one positive/affirming** insight — from `positive` when a macro is
aligned, and from `positive_generic` when none is. The requirement matters most for the user whose
week went badly: any implementation gets this right for a balanced profile, but a purely
deficit-framed wall of corrections is exactly what §5 rules out.

**`fibre_low` was dropped.** `food_item` caches energy plus the three macros only, so a fibre claim
would have no data behind it. There is no fibre column anywhere in `02`, and inventing the signal
from a hand-tagged food list would be a guess presented as a measurement. A test asserts the
template stays absent so it cannot be reintroduced without a data source.

**Three calorie outcomes, not one.** The insight distinguishes cases the original single template
conflated: a *verdict* (target exists, ≥3 active days, deviation beyond
`CALORIE_GAP_TOLERANCE_FRACTION` = 12%); a *profile prompt* naming **exactly which** metrics are
still missing; and a *coverage note* for a complete profile with too few logged days. The
distinction is not cosmetic — a fixed list ("add your height, weight and activity level") would
let a user supply all three, still get no verdict because age and sex are absent, and see the
identical message again. And the coverage note deliberately never mentions the profile, because
that user's profile is already complete.

`{examples}` come from the content-based score below when the user's own nutrition cache has usable
rows, and from a small curated map per nutrient otherwise (protein → "chicken breast, Greek yogurt,
lentils…"). The curated fallback is the **normal** path today, not an error case: `food_item` holds
the 101 seeded Food-101 class names with NULL macros until M5's USDA client populates them.

Examples are joined with "**or**" (alternatives — "and" would read as an instruction to eat all
three), while the missing-metrics list is joined with "**and**" (all are required — "or" would
imply any one suffices).

## 3. Service interface

```python
# app/services/recommendation_service.py — the only module in M7 that touches a Session
def get_recommendations(user: User, db, *, force: bool = False) -> RecommendationResult:
    count = count_confirmed_meals(user.user_id, db)          # ALL confirmed meals
    if count < settings.min_meals_for_recommendation:
        return locked_result(count)                          # no rows written

    if not force and cache_is_fresh(user.user_id, db):       # <6 h AND no meal since
        return cached_batch(...)

    meals   = load_window_meals(user.user_id, db, days=...)  # macros summed in SQL
    profile = aggregate_profile(meals, ...)
    if profile is None:
        return unlocked_but_no_recent_meals(count)           # third state; nothing persisted

    tdee     = estimate_tdee(**user_metrics)                 # None if metrics absent
    insights = generate_insights(profile, goal=user.health_goal,
                                 calorie_target=goal_target_calories(tdee, user.health_goal),
                                 candidates=load_food_candidates(user.user_id, db),
                                 missing_metrics=missing_metric_labels(**user_metrics))
    return persist_batch(user.user_id, insights, db)         # ≤5, one shared generated_at
```

- Triggered: (a) automatically via `on_meal_logged()` after a `POST /meals` that crosses the 7-meal
  threshold or adds to an already-unlocked user, and (b) on demand via `POST
  /recommendations/refresh` (`03 §7`). `on_meal_logged` **never raises** — a failure to refresh
  insights must not fail the meal log that triggered it, since the meal is the user's own data and
  the insights are merely derived from it.
- Idempotency: if insights were generated < `RECOMMENDATION_CACHE_HOURS` (6) ago **and** no
  confirmed meal has been logged since, return the stored rows rather than regenerating. The second
  condition compares the newest confirmed `logged_at` against the last `generated_at` rather than
  comparing meal counts, so it also catches a delete-then-add that leaves the total unchanged.

**The unlock gate counts all confirmed meals, not only those in the window.** Unlocking is a
milestone, not a rolling status: if it were windowed, a user who logged 10 meals and then took a
week off would find the Insights screen re-locked with a progress bar telling them to start over.
Losing a feature you already earned reads as a bug and punishes the behaviour the app exists to
encourage. The cost is that "unlocked" no longer implies "has recent data", which is why there is a
**third response state** (`03 §7`): unlocked, no insights, and a `no_recent_meals` notice. Nothing
is persisted in that state — there is no nutritional finding to record.

**Persistence is append-only, and each batch shares one `generated_at`.** "The user's current
insights" is then the rows whose `generated_at` equals the maximum. Without the shared timestamp,
per-row defaults would differ by microseconds and a 2-insight batch following a 5-insight one would
leave 3 stale rows visible in any "5 newest rows" query — the user would read old and new advice
mixed on one screen.

## 4. Edge cases

| Case | Behaviour |
|------|-----------|
| < 7 confirmed meals | `unlocked:false` + progress message; no rows written |
| All 7 meals same day | Still valid; window is 7 *days* but threshold is 7 *meals* — both satisfied |
| Unconfirmed drafts | Excluded from the gate and the window; a draft must never reach the engine |
| Unlocked, empty window | `unlocked:true`, no insights, `no_recent_meals` message; nothing persisted |
| Goal changed (FR13) | Next refresh recomputes against the new target distribution |
| Sparse/zero-calorie meals | Skip meals with `total_calories` null/0 from the average |
| Calories but no macros | Skipped — its energy would inflate the average while adding no grams, deflating every macro percentage at once |
| Negative macro (corrupt row) | Skipped — it makes the percentages *negative*, which Stage 3 would render verbatim ("below your target (-40% of calories vs 20%)") |
| Future `logged_at` | Excluded — clock skew must not add a meal to a window it does not belong to, nor invent an active day |
| Fewer than 3 active days | Macro insights normal; calorie insight downgraded to the coverage note |
| No suitable foods to suggest | Emit behavioural insight (cooking method / balance) instead of a food list |
| Another user's `rec_id` on PATCH | `403` (`03 §1`); the target row is left untouched |

## 5. Wellbeing & safety (non-negotiable)

- Tone is **supportive and non-judgmental**. Never shame, never use language that could encourage
  disordered eating (no "bad foods", no guilt framing, no aggressive calorie cutting).
- Never recommend extreme deficits or surpluses; targets are mild by design.
- **Every response includes the disclaimer** and, where appropriate, a gentle pointer to consult a
  qualified professional for specific dietary needs.
- These rules are tested in `11` (assert disclaimer present; assert no blocked phrases).

**How each rule is enforced structurally rather than by intent** — this is the part to defend:

| Rule | Mechanism |
|------|-----------|
| No shaming language | `BLOCKED_PHRASES` in `reco_targets.py`, asserted against every template **and** every rendered insight across a profile × goal matrix. Templates are the only text source, so passing means no code path can emit one. A runtime filter drops any insight containing a blocked phrase — redundant today, but it is what stops a USDA cache row named "junk food surprise" being laundered into an insight by interpolation. |
| No extreme deficits | `CALORIE_FLOOR_KCAL` clamp inside `goal_target_calories()`, plus postures bounded to ±20% of TDEE. Arithmetic, not wording. |
| No invented targets | `estimate_tdee()` returns `None` on any missing metric; Stage 3 then cannot produce an above/below claim. |
| Disclaimer always present | A non-optional field with a constant default on the response schema, so a response object cannot be constructed without one. |
| Tentative phrasing | Templates use "could", "may", "tends to"; tests assert the absence of "you must" / "stop eating". |

The blocked list contains only **multi-word judgement phrases** ("bad food", "too fat", "cheat
day") — deliberately never bare nutrient names, since the app must be able to say "fat" and
"calories" in ordinary sentences.

## 6. Evaluation for the report

Because there is no ground-truth "correct" recommendation, evaluate via:
- **Sanity/consistency tests:** synthetic profiles with a known dominant gap must yield the matching
  `nutrient_focus` (unit-tested).
- **Face-validity walkthrough:** seed a user with protein-poor meals → confirm a protein insight; seed
  a balanced user → confirm a positive insight.
- Document the algorithm and these checks in the results chapter (this is standard for a rule/content
  based recommender at this scope).

**No model is trained for FR12, and that is a design position, not a gap.** A supervised recommender
needs labels — a ground-truth "correct insight" for a given eating pattern — which do not exist
because the correct answer is a clinical judgement. A collaborative recommender needs many users ×
rated items, which this app has neither of, plus it would derive one user's dietary advice from
strangers' eating habits: precisely the claim the disclaimer exists to avoid. The trained models in
this project sit where a learned function genuinely belongs — perception (`04` classifier, `05`
portion aux models). The advice layer is deliberately transparent so that every sentence it can
produce is auditable in one file.

**Test inventory (as built).** 256 M7 tests across six files:

| File | Covers |
|------|--------|
| `test_m7_anthropometrics.py` | The five body metrics stay optional (M2's contract unbroken), round-trip, and reject unit mix-ups (`height_cm: 1.78`) at the schema edge |
| `test_m7_reco_targets.py` | Target integrity (each distribution sums to 100), Mifflin-St Jeor pinned to a hand-computed value, the calorie floor, template placeholder/tone hygiene |
| `test_m7_recommender.py` | Stage 1 filters and window bounds, percentages summing to 100 under a calorie/macro mismatch, denominator invariance, Stage 2 focus selection and determinism, goal sensitivity |
| `test_m7_insights.py` | Composition and the cap of 5, the mandatory positive insight on *bad* profiles, the three calorie outcomes, content-based scoring, tone over a profile × goal matrix |
| `test_m7_service.py` | The gate against real rows, drafts excluded, the 6 h cache and its invalidation, batch persistence, per-user isolation, the `on_meal_logged` hook |
| `test_m7_api.py` | `03 §7` key sets field-for-field, 200-when-locked, auth on every route, `user_id` never exposed, 403/404 on another user's insight |

**Two bugs the tests caught that are worth reporting**, since "the tests found nothing" is a weaker
claim than a specific catch:
1. **An inverted sign in the content-based score.** The alignment term used `-1` where the
   corrective direction is `+1`, so for a protein deficit grilled chicken scored −79.6 and white
   bread −22.0 — the engine would have preferred the *least* helpful foods. And because
   `select_foods` only accepts positive scores, it would have silently suggested nothing and always
   fallen through to the curated list, hiding the fault behind a plausible-looking output.
2. **A dead-end in the calorie note.** The original wording named three of the five metrics TDEE
   requires, so a user could add exactly what was asked for and see the identical message again.

---

Prev: [`05-portion-estimation.md`](./05-portion-estimation.md) · Next: [`07-mobile-app.md`](./07-mobile-app.md)
