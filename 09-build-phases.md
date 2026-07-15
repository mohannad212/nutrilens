# 09 — Build Phases & Task Breakdown

**This is the execution plan. Build in order.** Each milestone is independently runnable and ends
with a **Definition of Done (DoD)** that must pass before moving on. The ordering guarantees the
**core loop works as early as possible** (by end of M5), then each milestone adds capability.

Legend: `[BE]` backend · `[ML]` model/training · `[APP]` mobile · `[INF]` infra/devops.

---

## M0 — Foundations & scaffolding
**Goal:** repos, tooling, CI, and a runnable empty app on every tier.

- `[INF]` Create monorepo (`01 §3`), `.gitignore`, `.env.example`, `docker-compose.yml` (Postgres + backend).
- `[INF]` GitHub repo, branch protection on `main`, `develop` branch, CI workflow skeleton (`10`, `11`).
- `[BE]` FastAPI app factory, `/health`, config loader, ruff/black/mypy configured.
- `[BE]` SQLAlchemy engine/session, Alembic init.
- `[APP]` Expo + TypeScript app scaffold, navigation shells (AuthStack/AppTabs), theme tokens (`07 §8`).
- `[ML]` `training/` folder, Colab notebook skeleton, download Food-101, verify official split loads.

**DoD:** `docker-compose up` serves `GET /health` → 200; app launches to a Login placeholder; CI runs
lint+empty tests green; Food-101 downloads and a batch renders in the notebook.

## M1 — Database & core plumbing
**Goal:** schema live, models wired, seed scripts.

- `[BE]` Implement all ORM models (`02 §3`); create the two ENUM types.
- `[BE]` First Alembic migration; `alembic upgrade head` builds all 5 tables + indexes.
- `[BE]` `seed_food101_classes.py`; dev `seed_demo_user.py` (guarded).
- `[BE]` `security.py` (bcrypt + JWT), `deps.py` (`get_db`, `get_current_user`).

**DoD:** migration creates the schema exactly as `02`; seed inserts 101 classes; a unit test round-trips
a `User` with a hashed password.

## M2 — Authentication (FR01, FR02)
**Goal:** register + login end-to-end.

- `[BE]` `auth_service` + `/auth/register`, `/auth/login`; lockout after 5 fails (423); error envelope.
- `[BE]` Pydantic schemas for auth (`03 §2`); validation rules.
- `[APP]` Login & Register screens (`07`), `AuthContext`, secure-store JWT, axios interceptors.
- `[BE][11]` Tests: duplicate email→409, weak password→422, 5 fails→lock→423, happy path issues JWT.

**DoD:** a new user registers and logs in from the app; token persists across relaunch; lockout works.

## M3 — Recognition (FR03, FR04) — needs a model
**Goal:** photo → predicted food class(es).

- `[ML]` Train EfficientNet-B4 per `04` (this runs in parallel starting at M0; must be ready here).
  Export `food101_efficientnet_b4.pth` + `labels_food101.json` to a GitHub Release.
- `[BE]` `preprocess.py`, `classifier.py`; load model at startup (lifespan).
- `[BE]` `recognition_service` + `POST /recognise` (classification part), image validation, in-memory
  only, top-1/top-3 threshold rule.
- `[APP]` Camera Capture screen (`07 §6`), upload, Recognition Results screen (predictions + confidence).

**DoD:** photographing a known food returns the correct top-1 (or a top-3 including it) in the app;
oversized/low-res images return the correct 4xx; **no image is written to disk** (verified in test).

## M4 — Portion estimation (FR05, FR06)
**Goal:** add calibrated portion ranges + user confirmation.

- `[ML]` Build the portion pipeline per `05` in the incremental order of `05 §10`
  (shape+container → reference → trained segmentation). Train/collect the aux models.
- `[BE]` `portion/` package; integrate into `recognition_service` so `/recognise` returns the `portion`
  object; unit tests with synthetic masks + known scales.
- `[APP]` Portion slider (`07`), live range display, `method`/`reference` shown subtly.

**DoD:** `/recognise` returns a sensible `{min_g,max_g,point_g,method}`; slider lets the user adjust;
portion pipeline unit tests pass; MAPE measured on the validation set (`05 §7`) and recorded.

## M5 — Nutrition + logging = **CORE LOOP COMPLETE** (FR07, FR08, FR09, FR10)
**Goal:** the whole loop works: photo → recognise → portion → nutrition → log.

- `[BE]` `usda_client` + `nutrition_cache` (`08 §5–6`); `nutrition_service` + `POST /nutrition`.
- `[BE]` `meal_service` + `POST /meals` (recompute totals, `user_confirmed=TRUE`, unlock flag).
- `[APP]` Cooking-method picker, Nutrition Summary screen (calorie ring + macro bars + **disclaimer**),
  "Log This Meal" → `POST /meals` → success → Home.
- `[BE][11]` Tests: cache hit vs USDA miss, scaling math, USDA-down→cache fallback, totals recomputed.

**DoD (headline):** on a device/emulator a user photographs a meal and completes photo→log in ≤ 3
interactions; nutrition comes from USDA/cache (never fabricated); meal appears with correct totals.

## M6 — History & profile (FR11, FR13)
**Goal:** review past meals; manage goal.

- `[BE]` `GET /meals` (paginated, confirmed-only), `GET /meals/{id}`, `DELETE /meals/{id}`,
  `PATCH /users/me/goal`, `GET /users/me`.
- `[APP]` Meal History screen (date-grouped, per-meal kcal, tap→detail, delete); Profile & Goals screen.

**DoD:** history shows logged meals newest-first with correct calories; changing the goal persists and
is reflected on next recommendation refresh.

## M7 — Recommendations (FR12)
**Goal:** personalised insights after ≥ 7 meals.

- `[BE]` `recommender.py` + `recommendation_service` (`06`); `GET /recommendations`,
  `POST /recommendations/refresh`, `PATCH /recommendations/{id}/read`; auto-trigger from `POST /meals`.
- `[APP]` Insights screen: locked-state progress (<7), unlocked insight cards + **disclaimer**.
- `[BE][11]` Tests: locked <7; synthetic protein-poor profile → protein focus; disclaimer always present;
  no blocked/harmful phrasing.

**DoD:** a user with ≥7 confirmed meals sees ≤5 relevant insights aligned to their goal, each with the
disclaimer; users with <7 see an accurate progress message.

## M8 — Account deletion, hardening & demo (FR14 + NFRs)
**Goal:** finish P2, meet NFRs, ship a demo.

- `[BE]` `DELETE /users/me` (irreversible cascade); confirm-flag guard.
- `[APP]` Delete-account flow (double confirm), empty/error/offline states polish (`07 §7`).
- `[INF]` Deploy to the eval VM with HTTPS/TLS (`08 §10`); run migrations; load model from Release.
- `[11]` Load test 50 concurrent sessions (Scalability); p95 latency ≤ 5 s (Performance); security pass
  (bcrypt, TLS, per-user isolation, no image persistence); portability check Android + iOS.
- `[DOC]` Record final metrics (classifier top-1/top-5, portion MAPE) for the GP2 report.

**DoD (project):** README §3 checklist passes end-to-end on a real device/emulator against the deployed
backend; all P0+P1 FRs green; FR14 works; NFR targets met; demo script (`11 §Demo`) runs clean.

---

## Dependency graph (what blocks what)

```
M0 → M1 → M2 ─────────────► (app auth)
        └─► M3(needs trained model from ML track) → M4 → M5(core loop) → M6 → M7 → M8
ML track: start Food-101 training at M0, must deliver weights by M3; portion aux models by M4.
```

**Parallelism:** the ML training track (`04`, `05` model training) runs from M0 alongside backend/app
work so weights are ready exactly when M3/M4 need them. Don't block M0–M2 on training.

## Per-milestone "you are done when"
Each milestone's DoD above is the gate. Do not start milestone *n+1* until milestone *n*'s DoD and its
tests (`11`) are green on `develop`.

---

Prev: [`08-backend-services.md`](./08-backend-services.md) · Next: [`10-coding-standards.md`](./10-coding-standards.md)
