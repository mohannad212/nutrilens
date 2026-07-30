# M7_EXPLANATION_GUIDE.md — Recommendation Engine (FR12)

This guide explains **M7** (see `09-build-phases.md`): the milestone where NutriLens stops
describing individual meals and starts saying something about a user's *pattern* of eating — up to
five plain-language insights drawn from their own 7-day history and aligned to their health goal.

It is also the milestone with the least machinery and the most judgement. There is no model to
train and no accuracy metric to report, so almost every defensible claim in this chapter is about a
**decision** rather than a number. That is what this guide is for.

---

## 1. What we built

| File | Role |
|---|---|
| `app/core/constants.py` | The two contractual disclaimers (`03 §5`, `§7`), defined once |
| `app/ml/reco_targets.py` | The engine's entire "model": target distributions, Mifflin-St Jeor, every insight template, the blocked-phrase list, scoring weights |
| `app/ml/recommender.py` | The three stages of `06 §1` as **pure functions** — no `Session`, no clock, no I/O |
| `app/services/recommendation_service.py` | The only M7 module touching a `Session`: unlock gate, window query, 6 h cache, persistence, `on_meal_logged` hook |
| `app/schemas/recommendation.py` | `03 §7` response shapes; disclaimer as a required field |
| `app/api/recommendations.py` | `GET /recommendations`, `POST /recommendations/refresh`, `PATCH /recommendations/{id}/read` |
| `alembic/versions/0002_user_anthropometrics.py` | Five nullable body metrics on `user` + two enum types |
| `tests/test_m7_*.py` (6 files) | 256 tests: anthropometrics, targets, stages 1–2, generation, service, HTTP contract |

**Definition of Done, verified:** a user with ≥ 7 confirmed meals receives ≤ 5 insights aligned to
their goal, each response carrying the disclaimer; users below 7 get an accurate progress message
with `200`, not an error. Full suite: **320 passed**, ruff + black + mypy clean.

**Not done here, and deliberately:** the auto-trigger call site (M5 owns `POST /meals`; the hook is
built and tested, the one-line call is M5's) and the Insights screen (`07`, gated on the mobile
build-out). Everything else is reachable over HTTP today.

---

## 2. Why we built it this way

The single most important structural decision is the **split between `recommender.py` and
`recommendation_service.py`**. The engine is pure functions over plain values; the service does all
database work. Three things follow from that, and all three are worth saying out loud in a defence:

1. **`06 §6`'s evaluation becomes cheap.** "A synthetic profile with a known dominant gap must yield
   the matching `nutrient_focus`" is a one-line unit test. Had the aggregation read from a
   `Session`, every case would need a user, seven meal rows, seven meal items and a food item —
   and nobody writes forty of those, so the evaluation would have been thinner.
2. **The wellbeing rules become auditable.** Every sentence the app can say lives in one constants
   file, so "prove nothing shaming can be emitted" is a review of one module plus one test.
3. **Policy sits at the boundary.** The 7-meal gate is a product rule, not arithmetic, so it lives
   where the user's history is visible — in exactly one place.

---

## 3. The three stages (what actually happens)

```
Stage 1  AGGREGATE   meals → one daily-average macro profile (% of energy)
Stage 2  COMPARE     profile − goal target → signed gaps in percentage points → nutrient_focus
Stage 3  GENERATE    gaps → ≤5 plain-language insights (+ food examples, + the positive one)
```

Worked example — 8 falafel-wrap meals, `weight_loss` goal, body metrics supplied:

```
profile   12% protein / 47% carbs / 40% fat, 1707 kcal/day over 3 active days
targets   30% / 40% / 30%,  TDEE 2625 → target 2232 kcal (×0.85)
gaps      protein −18 pp | fat +10 pp | carbs +7 pp        (they always sum to 0)
focus     protein
insights  1 protein_low  2 fat_high  3 carbs_high  4 calorie_gap  5 positive_generic
```

Note the composition: **≤3 corrective + ≤1 calorie + exactly 1 positive = ≤5**. FR12's cap holds by
construction, not by truncating a longer list.

---

## 4. Decisions you should be able to defend

These are the ones a supervisor is most likely to probe, because each one had a plausible
alternative that is subtly wrong.

**Percentages divide by macro-derived energy (`4p + 4c + 9f`), not by reported calories.**
`06 §1` originally wrote `p_pct = protein_g×4 / calories`. Reported calories rarely equal
`4p + 4c + 9f` exactly, so that denominator makes the three percentages sum to ~94% or ~106%. The
targets sum to exactly 100%, so the mismatch shows up as a **phantom gap in all three macros at
once** — a user eating a perfectly balanced diet is told protein, carbs *and* fat are all low.

**"Per day" means per *active* day, not per window day.** Dividing a week's food by 7 assumes daily
logging. Someone who logs Monday and Tuesday then stops would appear to eat 2/7 of what they ate,
and the engine would tell them to eat *more*. The macro percentages are unaffected either way —
they are ratios, so the denominator cancels, and there is a test proving the same pattern logged
over 2 days or 6 days yields identical percentages. Only the absolute kcal figure moves, which is
why `active_days` rides on the profile and a calorie verdict requires ≥ 3 of them.

**The unlock gate counts all confirmed meals, not only those in the window.** Unlocking is a
milestone, not a rolling status. Windowing it would re-lock a user who logged 10 meals and then took
a week off — taking away a feature they had already earned, and punishing exactly the habit the app
wants to build. The cost is a third response state (unlocked, empty window, `no_recent_meals`
notice, nothing persisted), which is an honest "I have nothing to tell you yet" rather than a
fabricated profile.

**No model is trained, and that is a position rather than a shortcut.** A supervised recommender
needs a ground-truth "correct insight" per eating pattern, which is a clinical judgement and does
not exist. A collaborative one needs many users × rated items — absent here, and it would derive one
user's dietary advice from strangers' habits, the precise claim the disclaimer exists to avoid. The
trained models in this project sit where a learned function genuinely belongs: perception (`04`
classifier, `05` portion aux models). See `06 §6`.

**Five nullable body metrics rather than a generic calorie target.** The calorie insight needs
something to compare against. A self-reported target would let a user in a restrictive mindset
enter an unsafely low number and have the app validate it; a population average (`2000 kcal`) would
be a fabricated figure presented as personal advice. Mifflin-St Jeor from five optional fields is
the only option that is both personal and un-negotiable-with. Nullability is what keeps the app
usable for someone who declines: they lose the calorie *verdict* only.

---

## 5. Concepts you should be able to explain

**Content-based vs collaborative filtering.** Content-based scores items by their own attributes
against a profile built from the same user's history — no other users involved, so no cold-start
problem and no privacy question. Collaborative infers from similarity between users. FR12 is
content-based by specification, which is also why cold-start here is solved by a *gate* (7 meals)
rather than by borrowing other users' data.

**The scoring function** (`06 §1` Stage 3):
```
score(food) = 1.0·aligns_with_gap − 0.5·conflicts_other_macros − 6.0·already_frequent
```
All three terms are in percentage points of energy, so the weights are dimensionless. Conflict is
deliberately *half* of alignment: weighting them equally produces a recommender that can only
suggest perfectly balanced foods, which in practice means suggesting nothing. Alignment is measured
**relative to what the user currently eats** — an 18%-protein food helps nobody already averaging
20%, however respectable 18% sounds in isolation.

**Why percentage points, not ratios.** The comparison is between two percentages, so their
difference is in points. A ratio would make 2%-vs-4% look as severe as 25%-vs-50%.

**Idempotency (`06 §3`).** A GET within 6 hours with no confirmed meal logged since returns the
stored rows. The second condition compares the newest `logged_at` against the last `generated_at`
rather than comparing counts, so a delete-then-add that leaves the total unchanged still
invalidates. Without this the insight wording would churn on every screen open, which reads as
instability rather than freshness.

**Append-only persistence with a shared `generated_at`.** All rows in one batch take the same
timestamp, so "current insights" is `generated_at == max(generated_at)`. With per-row defaults they
would differ by microseconds, and a 2-insight batch after a 5-insight one would leave 3 stale rows
in any "5 newest" query — old and new advice mixed on one screen.

---

## 6. Likely graduation-discussion questions (with model answers)

**Q: "Where is the machine learning in your recommendation engine?"**
A: There isn't any, by design — `06` specifies a content-based rule engine. Training would require
either labelled "correct insights" (a clinical judgement that doesn't exist) or collaborative
signal (no rated items, and it would base one user's advice on strangers). The ML in this project is
in perception: the Food-101 classifier and the portion pipeline. The recommender is evaluated the
way this class of system is standardly evaluated — sanity/consistency tests plus a face-validity
walkthrough (`06 §6`).

**Q: "How do you know the recommendations are any good, with no accuracy metric?"**
A: Two ways. Sanity tests: profiles built so the dominant gap is known by construction must produce
the matching `nutrient_focus`, and the same profile must give different verdicts under different
goals. Face validity: a protein-poor user gets a protein insight, a balanced user gets an affirming
one. I can also name specific defects the tests caught — an inverted sign in the food score, and a
calorie prompt that asked for three of the five fields it needed.

**Q: "What stops the app giving harmful advice?"**
A: Four structural guards, not good intentions. A calorie floor clamp inside the target calculation,
so no combination of inputs can state an extreme deficit. `estimate_tdee` returning `None` rather
than a default, so no verdict can rest on invented data. A blocked-phrase list asserted against
every template *and* every rendered insight, with a runtime filter that also catches a food name
interpolated from the USDA cache. And the disclaimer as a required schema field, so a response
cannot be built without it. See the table in `06 §5`.

**Q: "Why does a user with 4 meals get a `200` instead of a `403`?"**
A: Because being new is not an error. `03 §7` specifies `unlocked:false` with progress information;
a 4xx would make the mobile client's error handler show "something went wrong" to someone whose only
problem is that they just signed up.

**Q: "What happens if the user logs nothing for two weeks?"**
A: They stay unlocked — the gate counts all confirmed meals — and get `unlocked:true` with no
insights and a notice that there is nothing recent to analyse. Nothing is persisted, because there
is no finding to record. Re-locking them would take away something they had earned.

**Q: "Your engine reads `meal_log`, but M5 isn't built. How is this tested?"**
A: The service tests write real `meal_log` / `meal_item` / `food_item` rows through the ORM, so the
SQL aggregation, the `user_confirmed` filter and the window bounds are all exercised against the
real schema. What M5 adds is the endpoint that creates those rows in production, plus one call to
the `on_meal_logged` hook that already exists and is tested.

**Q: "Why is `sex` nullable with only two values?"**
A: Mifflin-St Jeor takes a two-valued constant (+5 / −161); there is no meaningful third constant
to offer. Nullability is what protects a user who would rather not answer — they keep every macro
insight, which is a ratio and needs no body data, and lose only the calorie verdict. No substitute
value is invented on their behalf.

---

## 7. Common mistakes to watch for (and avoid saying you made)

- **Averaging per-meal percentages instead of computing percentages from summed grams.** The mean of
  ratios is not the ratio of means; the former lets a 50-kcal snack count as much as a 900-kcal
  dinner.
- **`COALESCE`-ing the SQL macro sums to `0.0`.** NULL means "unknown" and the meal is dropped; zero
  would *assert* the meal contained no protein and drag every percentage down.
- **Treating an unconfirmed draft as a logged meal.** FR10 counts confirmed meals; the filter belongs
  in the WHERE clause, not in Python after loading everything.
- **Putting a bare nutrient name in the blocked-phrase list.** Banning "fat" would make the
  fat-macro insights unwritable. Blocked entries are multi-word judgement phrases only.
- **Letting the locked path write rows.** A user below the threshold must not accumulate a hidden
  history of insights; there is a test asserting zero rows.
- **Taking `user_id` from the path or body.** The authenticated token is the only acceptable source,
  or one user ends up reading another's dietary insights.

---

## 8. How to demo M7

```bash
# 1. Register (body metrics optional — supply them to get a calorie verdict)
curl -X POST localhost:8000/api/v1/auth/register -H 'Content-Type: application/json' \
  -d '{"username":"demo","email":"demo@x.com","password":"Demo1234","health_goal":"weight_loss",
       "height_cm":175,"weight_kg":72,"birth_year":2001,"sex":"male","activity_level":"moderate"}'

# 2. Log in, keep the token
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"demo@x.com","password":"Demo1234"}' | jq -r .access_token)

# 3. Locked state (0 meals) -> 200, unlocked:false, progress message
curl -s localhost:8000/api/v1/recommendations -H "Authorization: Bearer $TOKEN" | jq

# 4. Seed 8 confirmed meals (scripts/seed_demo_user.py, or POST /meals once M5 lands)

# 5. Unlocked -> ≤5 insights, each with nutrient_focus, plus the disclaimer
curl -s localhost:8000/api/v1/recommendations -H "Authorization: Bearer $TOKEN" | jq

# 6. Force a regeneration, and mark the first insight read
curl -s -X POST localhost:8000/api/v1/recommendations/refresh -H "Authorization: Bearer $TOKEN" | jq
curl -s -X PATCH localhost:8000/api/v1/recommendations/1/read -H "Authorization: Bearer $TOKEN" | jq
```

Points to draw attention to while demoing: the locked response is a `200`; the disclaimer is present
in **both** states; calling GET twice returns the same `rec_id`s (the cache); refresh returns new
ones; and omitting the body metrics changes the calorie insight from a verdict into a note that
names the missing fields.

---

Prev: [`M2_EXPLANATION_GUIDE.md`](./M2_EXPLANATION_GUIDE.md) · Spec: [`06-recommendation-engine.md`](./06-recommendation-engine.md)
