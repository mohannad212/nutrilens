# 11 — Testing, QA & Acceptance

Tests are written **as each milestone is built** (`09`). A milestone is not done until its acceptance
tests are green. This doc gives the strategy, the per-FR acceptance criteria, and the demo script.

---

## 1. Test strategy (the pyramid)

| Layer | Tooling | What it covers |
|-------|---------|----------------|
| **Unit** | pytest (BE), Jest + React Native Testing Library (APP) | pure logic: scaling math, portion geometry, gap computation, reducers, components |
| **Integration** | pytest + httpx `TestClient` + test Postgres | endpoints against a real DB; auth flow; USDA client stubbed |
| **End-to-end** | Detox or Maestro (APP) against a running backend | the core loop on an emulator |
| **Load/perf** | Locust or k6 | 50 concurrent sessions; p95 latency |

Coverage targets: backend services/ml **≥ 80%** line coverage; critical paths (auth, nutrition scaling,
portion range, meal totals) **100%** of branches. CI fails under target.

## 2. Stubbing external dependencies

- **USDA:** never call the real API in tests. Provide `FakeUsdaClient` returning fixed per-100 g values;
  test cache-hit, cache-miss→upsert, rate-limit (429), unavailable (502→cache fallback), no-match (404).
- **Classifier:** inject a `FakeClassifier` returning deterministic predictions so recognition/endpoint
  tests don't need the real weights. Keep **one** slow "real-weights" smoke test behind a marker
  (`@pytest.mark.model`) run manually/nightly, asserting the exported model loads and predicts.
- **Portion:** unit-test geometry with synthetic masks + known scales (no ML needed).

## 3. Acceptance criteria per Functional Requirement

| FR | Acceptance test (pass = these hold) |
|----|-------------------------------------|
| FR01 | Register: valid→201 + row created; duplicate email→409; password <8 or no digit→422 |
| FR02 | Login: valid→200 + JWT; wrong pw→401 w/ `remaining_attempts`; 5th fail→`account_locked=TRUE`; locked→423 |
| FR03 | Upload: jpeg/png ≤10MB ≥640×480→accepted; >10MB→413; wrong type→400; <640×480→400; **no file written to disk** |
| FR04 | top-1 ≥ threshold→single item; top-1 < threshold→`low_confidence:true` + ≤3 `alternatives` |
| FR05 | `/recognise` returns `portion` with `min_g<point_g<max_g` and a valid `method`; never a bare number |
| FR06 | Slider emits confirmed grams within [min,max]; override outside range allowed but flagged |
| FR07 | `cooking_method` from the enum is appended to the USDA query; invalid method→422 |
| FR08 | Macros come from USDA/cache, correctly scaled: `scaled = per100g × portion_g/100` (±rounding); USDA-down + cached→still returns `source:"cache"` |
| FR09 | Nutrition response includes calorie + 3 macros + **disclaimer** string |
| FR10 | `POST /meals` persists log+items, `user_confirmed=TRUE`, `total_calories = Σ items`; returns `meals_logged_total`; **image already discarded** |
| FR11 | `GET /meals` returns confirmed-only, newest-first, paginated; other user's meal→403 |
| FR12 | <7 meals→`unlocked:false` + progress (not error); ≥7→≤5 insights + `nutrient_focus` + **disclaimer**; synthetic protein-poor profile→protein focus |
| FR13 | `PATCH /users/me/goal` persists; next refresh uses new target distribution |
| FR14 | `DELETE /users/me` removes user + meals + items + recommendations (cascade); token invalid after; `food_item` rows preserved |

## 4. Non-Functional acceptance

| NFR | Test |
|-----|------|
| Performance | Load test: p95 of `/recognise`+`/nutrition` ≤ **5 s** under normal load (warm model) |
| Accuracy | Classifier eval script reports test **top-1 ≥ 80%** (or documented fallback path used) |
| Usability | Manual: first-time user completes photo→log in **≤ 3 interactions** (scripted walkthrough) |
| Security | bcrypt hashes in DB (never plaintext); all endpoints require JWT except public ones; user A cannot read user B's meals (403); TLS in deploy |
| Scalability | 50 concurrent simulated users, no error-rate spike, latency within threshold |
| Reliability | Kill USDA (stub 503) with warm cache → nutrition still returned from cache; uptime ≥95% in eval window |
| Portability | App runs on Android 8.0+ and iOS 13+ from one codebase (smoke both) |
| Maintainability | Lint/format/type checks pass in CI; model rollback = redeploy prior Release artifact |

## 5. CI (`.github/workflows/ci.yml`)

On every PR to `develop`/`main`:
1. Backend: `ruff` + `black --check` + `mypy` + `pytest --cov` (fail under coverage target).
2. Mobile: `eslint` + `prettier --check` + `tsc --noEmit` + `jest`.
3. Spin an ephemeral Postgres service for backend integration tests.
4. (Nightly/manual) `@pytest.mark.model` smoke test loads the real weights.

## 6. Manual QA checklist (pre-demo)

- Register → login → capture → recognise → adjust portion → pick method → see nutrition + disclaimer →
  log → see it in history. All on a real device/emulator against the deployed backend.
- Trigger low-confidence path (ambiguous photo) → top-3 shown.
- Kill network → friendly offline states, no crashes.
- Log to 7 meals → insights unlock with disclaimer.
- Change goal → insights shift on refresh.
- Delete account → everything gone, logged out.

## 7. Demo script (for the committee)

1. **Login** as a seeded user (already has 6 confirmed meals).
2. **Photograph** a grilled chicken salad → recognised (top-1) + portion range shown.
3. **Adjust** the slider slightly; **pick** "grilled".
4. **Nutrition** screen: USDA macros scaled to the portion, disclaimer visible.
5. **Log** the meal → now 7 meals → **Insights unlock** live.
6. Open **Insights** → personalised, goal-aligned, disclaimer present.
7. Show **History**; change **goal**; show insights change on refresh.
8. Mention the **honest evaluation**: classifier top-1 %, portion MAPE, and the in-the-loop confirmation.

---

Prev: [`10-coding-standards.md`](./10-coding-standards.md) · Back to [`README.md`](./README.md)
