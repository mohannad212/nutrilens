# 00 — Product Overview & Scope

Companion doc: this is the "what and why". For the "how", see `01`–`11`.

---

## 1. Vision

> Make dietary tracking effortless by replacing manual data entry with a single photograph.

The user takes a photo of their meal. Within ~5 seconds they get: what the food is, how much of
it there is, and its lab-verified nutrition — logged automatically. Over time the app learns their
habits and nudges them toward their health goal.

## 2. Problem statement

Existing dietary trackers (MyFitnessPal, FoodVisor, Lose It!, Cal AI) all require significant
manual effort (searching, barcode scanning, manual portion entry) and none simultaneously deliver:
automated portion estimation, multi-component dish handling, ML-personalised recommendations from
real meal history, and lab-verified (non-crowdsourced) nutrition. NutriLens targets exactly this gap.

## 3. Target users

| User | Needs | Priority |
|------|-------|----------|
| **End user (general public)** | Fast, low-effort meal logging; trustworthy nutrition; gentle guidance | Primary |
| **Health-conscious individual / dieter** | Track macros against a goal (loss / maintain / gain) | Primary |
| **Healthcare professional (indirect)** | Recommend a reliable tracking aid to patients | Secondary |
| **System administrator** | Deploy, monitor, update model weights | Operational |

## 4. Goals (what GP2 must achieve)

1. A working native mobile app (Android + iOS via React Native).
2. A trained EfficientNet-B4 classifier on Food-101 reaching a **target top-1 ≥ 80%** on the test split.
3. A functional 3-stage portion-estimation pipeline returning calibrated **ranges**.
4. Real USDA FoodData Central integration with local caching.
5. A content-based recommendation engine that activates after 7 logged meals.
6. Secure auth, persistent storage, and a full end-to-end core loop.

## 5. Non-goals (explicitly out of scope for GP2)

- Real-time video food tracking.
- Social features / sharing / community.
- Wearable or smart-scale hardware integration.
- Barcode scanning (photo-first product).
- Payment / subscription billing.
- Arabic / Middle-Eastern cuisine classes **beyond** the Food-101 taxonomy (documented as future
  work; the pipeline must not break on out-of-distribution foods — it returns low-confidence + top-3).
- Clinical/medical advice — the app provides *informational* insights only, always with a disclaimer.

## 6. Feature list — Functional Requirements mapped to build priority

Priority key: **P0** = core loop, build first · **P1** = required, build after core loop ·
**P2** = required but last. All are "Essential" per the Phase-1 report; priority here is *build order*,
not importance. Milestones (M#) reference `09-build-phases.md`.

| FR | Feature | Priority | Milestone | Spec location |
|----|---------|----------|-----------|---------------|
| FR01 | User registration | P0 | M2 | `03` /auth/register |
| FR02 | User authentication (JWT, lockout after 5 fails) | P0 | M2 | `03` /auth/login |
| FR03 | Food image submission (JPEG/PNG, ≤10MB, ≥640×480) | P0 | M3 | `03` /recognise, `07` Camera |
| FR04 | Food item recognition (top-1, or top-3 if below threshold) | P0 | M3 | `04`, `03` /recognise |
| FR05 | Portion size estimation (calibrated mass range) | P0 | M4 | `05`, `03` /recognise |
| FR06 | User portion confirmation (slider) | P0 | M4 | `07` Recognition screen |
| FR07 | Cooking method selection (drop-down, appended to USDA query) | P0 | M5 | `03` /nutrition, `07` |
| FR08 | Nutritional data retrieval (USDA, scaled to portion) | P0 | M5 | `08` USDA client, `03` /nutrition |
| FR09 | Nutritional summary display (calorie ring + macro bars + disclaimer) | P0 | M5 | `07` Nutrition screen |
| FR10 | Meal log persistence (min 7 before recommendations; image discarded) | P0 | M5 | `02`, `03` /meals |
| FR11 | Meal history view (chronological, per-meal calories) | P1 | M6 | `03` /meals, `07` History |
| FR12 | Personalised recommendation (≤5 insights, needs ≥7 logs, disclaimer) | P1 | M7 | `06`, `03` /recommendations |
| FR13 | Health goal management (loss / maintenance / gain; editable) | P1 | M6 | `03` /users/me/goal |
| FR14 | Account data deletion (irreversible, complete) | P2 | M8 | `03` /users/me (DELETE) |

## 7. Non-Functional Requirements (acceptance targets)

| NFR | Requirement | Concrete target / test |
|-----|-------------|------------------------|
| **Performance** | Recognition + nutrition returned quickly | End-to-end ≤ **5 s** at p95 under normal load (single concurrent user, warm model) |
| **Accuracy** | Classifier quality | Top-1 **≥ 80%** on Food-101 test split; nutrition sourced only from USDA/cache |
| **Usability** | Low-friction core loop | First-time user completes photo→log in **≤ 3 interactions**; linear navigation |
| **Security** | Credentials & transport | Passwords = bcrypt (salted); all traffic HTTPS/TLS 1.2+; JWT; per-user data isolation |
| **Scalability** | Concurrency | ≥ **50** concurrent sessions without breaching the 5 s threshold (load-test in `11`) |
| **Reliability/Availability** | Graceful USDA degradation | Local cache fallback; target ≥ **95%** uptime in eval window |
| **Portability** | Cross-platform | Runs on Android 8.0+ **and** iOS 13+ from one codebase, no platform-specific forks |
| **Maintainability** | Code quality | PEP 8 + type hints (backend), ESLint + Prettier (app); model weights versioned & rollback-able |

Full acceptance criteria per FR/NFR live in `11-testing-qa.md`.

## 8. Success criteria (how we know GP2 passed)

- The end-to-end core loop works on a physical device or emulator (see README §3).
- Classifier hits the ≥80% top-1 target (or the documented fallback path in `04` is used).
- Portion pipeline returns sensible ranges on the demo food set.
- All P0 + P1 FRs pass acceptance tests; P2 (FR14) implemented.
- NFR load/latency tests pass at the stated targets.
- The demo script in `11 §Demo` runs start-to-finish without manual DB fixes.

## 9. Glossary

| Term | Meaning |
|------|---------|
| **Food-101** | Public benchmark dataset: 101 food classes × 1,000 images = 101,000 images |
| **EfficientNet-B4** | CNN architecture used as the classification backbone (transfer-learned from ImageNet) |
| **Transfer learning** | Start from ImageNet-pretrained weights, fine-tune on Food-101 |
| **Top-1 / Top-3** | Whether the correct label is the single best / among the best three predictions |
| **Portion range** | Estimated food mass as `[min_g, max_g]`, never a single fabricated number |
| **Reference object** | A known-size object in the photo (plate, fork, hand, coin) used to calibrate scale |
| **Shape prior** | A food-class-specific geometric model (disc, hemisphere, slab) for volume estimation |
| **USDA FDC** | USDA FoodData Central — the lab-verified nutrition database + REST API |
| **Content-based filtering** | Recommendation approach using the user's own aggregated meal profile |
| **Core loop** | photo → recognise → portion → nutrition → log → history → recommendation |
| **FR / NFR** | Functional / Non-Functional Requirement (from the Phase-1 report, Chapter 4) |

---

Next: [`01-architecture.md`](./01-architecture.md)
