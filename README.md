# NutriLens — Graduation Project 2 (Implementation)
## Product Requirements & Technical Build Specification

> **Purpose of this document set.** This is the complete, build-ready specification for the
> NutriLens application. It is written to be handed to **Claude Code** (or any competent
> engineer) to build a **fully functional application** end-to-end. Every design decision from
> Graduation Project 1 (Chapters 1–5) is turned here into concrete, technical, testable
> instructions — from the smallest detail (a column type, an HTTP status code, a hyperparameter)
> to the biggest (the full training pipeline and the end-to-end user loop).

---

## 0. What NutriLens is (one paragraph)

NutriLens is an AI-powered dietary tracking mobile app. A user photographs a meal; the system
recognises the food(s) with a fine-tuned **EfficientNet-B4** model trained on **Food-101**,
estimates the **portion size** from the photo using a hardware-free 3-stage computer-vision
pipeline, retrieves **lab-verified macronutrients** from the **USDA FoodData Central** API,
logs the meal, and — after enough history is accumulated — generates **personalised dietary
recommendations** with a content-based filtering engine. The client is **React Native**
(Android + iOS); the backend is **Python + FastAPI**; storage is **PostgreSQL**.

---

## 1. How to use these documents (read this first, Claude Code)

**Build in the order given in [`09-build-phases.md`](./09-build-phases.md).** Do not build
breadth-first. The milestones are ordered so the **core loop works as early as possible**, then
each subsequent milestone adds a capability. After each milestone there is a *Definition of Done*
you must satisfy (tests green, endpoint reachable, screen renders) before moving on.

When you need detail on a specific area, open the matching document:

| # | Document | What it covers |
|---|----------|----------------|
| 00 | [`00-product-overview.md`](./00-product-overview.md) | Vision, users, goals/non-goals, full feature list (FR01–FR14, NFRs) mapped to build priority, success criteria, glossary |
| 01 | [`01-architecture.md`](./01-architecture.md) | Three-tier architecture, exact tech-stack versions, monorepo folder layout, config & secrets, data-flow |
| 02 | [`02-database-schema.md`](./02-database-schema.md) | All 5 tables (3NF), every column/type/constraint/index, FK cascade rules, Alembic migrations, seed data |
| 03 | [`03-api-specification.md`](./03-api-specification.md) | Every REST endpoint: method, path, auth, request/response JSON, status codes, validation, errors, examples |
| 04 | [`04-ml-model.md`](./04-ml-model.md) | EfficientNet-B4 on Food-101: data prep, augmentation, transfer learning, hyperparameters, training loop, checkpointing, evaluation, export & serving |
| 05 | [`05-portion-estimation.md`](./05-portion-estimation.md) | Full 3-stage CV pipeline: container classification → reference-object calibration → shape priors → volume→mass → calibrated range |
| 06 | [`06-recommendation-engine.md`](./06-recommendation-engine.md) | Content-based filtering: profile vector, target macro distributions, gap computation, insight generation, thresholds, disclaimer |
| 07 | [`07-mobile-app.md`](./07-mobile-app.md) | React Native: all 9 screens, navigation, component tree, state, API client, camera/gallery, error/offline handling, design tokens |
| 08 | [`08-backend-services.md`](./08-backend-services.md) | FastAPI module structure, each service, USDA client, caching, rate limiting, config, error handling, logging |
| 09 | [`09-build-phases.md`](./09-build-phases.md) | **Ordered build plan** (M0–M8) with granular tasks, dependencies, and Definition of Done per milestone |
| 10 | [`10-coding-standards.md`](./10-coding-standards.md) | Conventions, linting, formatting, git workflow, branch strategy, commit format, env setup, Definition of Done |
| 11 | [`11-testing-qa.md`](./11-testing-qa.md) | Test strategy, unit/integration/e2e, acceptance criteria per FR, coverage targets, CI |

### Golden rules for the build

1. **Follow the milestones in `09` in order.** Each is independently runnable and testable.
2. **The contract documents are law.** `02` (DB), `03` (API) and the interface signatures in
   `04`–`06` are the source of truth. If a screen and an endpoint disagree, the endpoint spec wins;
   fix the caller.
3. **Never invent nutrition numbers.** All macronutrient values come from USDA FoodData Central
   or the local cache. If the API is unreachable and there is no cache entry, return a documented
   error — do not fabricate values.
4. **The AI is honest about uncertainty.** Portion estimation returns a **range** (min–max grams),
   never a single fake-precise number. The user always confirms before nutrition is computed.
5. **Every AI/recommendation surface shows the medical disclaimer** (see `06` and NFR list in `00`).
6. **Privacy: images are never persisted.** They live in memory only for the duration of the
   inference pipeline and are discarded immediately after the meal is confirmed (see `02`, `08`).
7. **Write tests as you go** (see `11`). A milestone is not done until its acceptance tests pass.

---

## 2. Scope decisions locked for GP2

These four decisions were confirmed with the project owner and are **fixed** for this build:

- **Format:** repository markdown docs (this set).
- **Portion estimation:** build the **full 3-stage research pipeline** (not a stub). See `05`.
- **Model:** **train EfficientNet-B4 ourselves** on Food-101 (transfer learning). See `04`.
- **Build target:** the **full core loop end-to-end**, all 14 FRs, prioritised so the core loop
  (photo → recognise → portion → nutrition → log → history → recommendation) is working first.

---

## 3. Definition of "done" for GP2 (the whole project)

The project is complete when a user can, on a real device or emulator:

1. Register an account and log in.
2. Photograph a meal and see it correctly recognised (top-1 or top-3 fallback).
3. See an estimated portion range and confirm/adjust it.
4. Pick a cooking method and receive real USDA macronutrients scaled to the portion.
5. Save the meal and see it in a chronological history with per-meal calories.
6. After ≥7 logged meals, receive personalised recommendations aligned to their health goal,
   each showing the medical disclaimer.

…and the backend, model, and app all pass their test suites in `11`.

---

*Authored for Middle East University — Graduation Project 2. Companion to the Phase 1 report
(`Report_Graduation_Project.pdf`, Chapters 1–5).*
