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
- Window: confirmed meals in the last **`RECOMMENDATION_WINDOW_DAYS` = 7** days (config).
- Guard: if `count(confirmed meals) < MIN_MEALS_FOR_RECOMMENDATION (7)` → return **locked** (not error).
- Compute the **daily average** macro vector over the window:
  `profile = mean_per_day(calories, protein_g, carbs_g, fat_g)`.
- Convert protein/carbs/fat to **% of energy** for comparability:
  `p_pct = protein_g×4 / calories`, `c_pct = carbs_g×4 / calories`, `f_pct = fat_g×9 / calories`.

### Stage 2 — Compare to target distribution
Target macro distributions (`% of energy`) per goal, stored in `app/ml/reco_targets.py`. These are
**reference proportions from widely accepted dietary guidelines** (state the source in the report;
they are informational defaults, not medical prescriptions):

| Health goal | Protein % | Carbs % | Fat % | Calorie posture |
|-------------|-----------|---------|-------|-----------------|
| weight_loss | 30 | 40 | 30 | mild deficit vs. maintenance estimate |
| maintenance | 20 | 50 | 30 | around estimated needs |
| weight_gain | 25 | 50 | 25 | mild surplus |

Compute the **signed gap** per macro: `gap_x = profile_pct_x − target_pct_x`. The macro with the
**largest absolute gap** is the **primary nutritional focus** (`nutrient_focus` in `02`/`03`).

### Stage 3 — Generate insights
- Rank gaps by magnitude; take the top 1–3 gaps.
- For each, select foods from the `food_item` table whose macro profile would **move the user toward
  target** (e.g. protein below target → suggest high-protein, moderate-calorie foods the user hasn't
  logged much). Selection = simple content-based score:
  `score(food) = w1·aligns_with_gap(food) − w2·conflicts_other_macros(food) − w3·already_frequent(food)`.
- Emit **plain-language** insight strings from templates (see §2), max 5 total.
- Persist each as a `recommendation` row (`insight_text`, `nutrient_focus`, `generated_at`).

## 2. Insight templates (tone: supportive, never alarming — see wellbeing note §5)

```
protein_low:  "Your protein has been below your {goal} target ({p_pct}% vs {target}%).
               Adding lean protein like {examples} to a couple of meals could help."
carbs_high:   "Carbohydrates made up {c_pct}% of your intake, a bit above your {goal} target.
               Swapping some refined carbs for vegetables or whole grains may help balance this."
fat_high:     "Fat was {f_pct}% of your intake. Choosing grilled or baked options over fried
               a few times this week could bring this closer to your target."
fibre_low:    "Your recent meals look low in fibre. Foods like {examples} are easy ways to add some."
calorie_gap:  "Your average daily intake ({kcal} kcal) is {above/below} what your {goal} goal suggests."
positive:     "Nice consistency — your protein intake is well aligned with your {goal} goal. Keep it up."
```
Always include at least **one positive/affirming** insight when the user is on track for a macro
(reinforces healthy behaviour, avoids a purely deficit-framed tone).

`{examples}` are pulled from a small curated map per nutrient (e.g. protein → "chicken breast,
Greek yogurt, lentils, eggs"). Keep examples generic and non-prescriptive.

## 3. Service interface

```python
# app/services/recommendation_service.py
def get_recommendations(user_id: int, db) -> RecommendationResult:
    count = count_confirmed_meals(user_id, db)
    if count < settings.MIN_MEALS_FOR_RECOMMENDATION:
        return RecommendationResult(unlocked=False, meals_logged=count,
                                    meals_required=settings.MIN_MEALS_FOR_RECOMMENDATION)
    profile = aggregate_profile(user_id, db, days=settings.RECOMMENDATION_WINDOW_DAYS)
    target  = TARGETS[user.health_goal]
    gaps    = compute_gaps(profile, target)
    insights = generate_insights(gaps, profile, user.health_goal, db)   # ≤5
    persist(user_id, insights, db)
    return RecommendationResult(unlocked=True, insights=insights, generated_at=now())
```

- Triggered: (a) automatically after a `POST /meals` that crosses the 7-meal threshold or adds to an
  already-unlocked user, and (b) on demand via `POST /recommendations/refresh` (`03 §7`).
- Idempotent per day: if insights were generated < 6 h ago and no new meals, return the cached rows
  rather than regenerating (avoid churn).

## 4. Edge cases

| Case | Behaviour |
|------|-----------|
| < 7 confirmed meals | `unlocked:false` + progress message; no rows written |
| All 7 meals same day | Still valid; window is 7 *days* but threshold is 7 *meals* — both satisfied |
| Goal changed (FR13) | Next refresh recomputes against the new target distribution |
| Sparse/￼zero-calorie meals | Skip meals with `total_calories` null/0 from the average |
| No suitable foods to suggest | Emit behavioural insight (cooking method / balance) instead of a food list |

## 5. Wellbeing & safety (non-negotiable)

- Tone is **supportive and non-judgmental**. Never shame, never use language that could encourage
  disordered eating (no "bad foods", no guilt framing, no aggressive calorie cutting).
- Never recommend extreme deficits or surpluses; targets are mild by design.
- **Every response includes the disclaimer** and, where appropriate, a gentle pointer to consult a
  qualified professional for specific dietary needs.
- These rules are tested in `11` (assert disclaimer present; assert no blocked phrases).

## 6. Evaluation for the report

Because there is no ground-truth "correct" recommendation, evaluate via:
- **Sanity/consistency tests:** synthetic profiles with a known dominant gap must yield the matching
  `nutrient_focus` (unit-tested).
- **Face-validity walkthrough:** seed a user with protein-poor meals → confirm a protein insight; seed
  a balanced user → confirm a positive insight.
- Document the algorithm and these checks in the results chapter (this is standard for a rule/content
  based recommender at this scope).

---

Prev: [`05-portion-estimation.md`](./05-portion-estimation.md) · Next: [`07-mobile-app.md`](./07-mobile-app.md)
