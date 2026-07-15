# 03 — REST API Specification

Base URL (dev): `http://<host>:8000`  ·  All paths below are relative to `/api/v1`.
Format: JSON in, JSON out (except `/recognise` which takes multipart). FastAPI serves interactive
docs at `/docs` and the schema at `/openapi.json`.

**This document is a contract.** Request/response shapes here are authoritative; the Pydantic
schemas in `app/schemas/` and the TypeScript types in `mobile/src/types/` must match exactly.

---

## 1. Conventions

- **Auth:** protected endpoints require `Authorization: Bearer <access_token>` (JWT, HS256).
- **Content type:** `application/json` unless noted. `/recognise` uses `multipart/form-data`.
- **Timestamps:** ISO-8601 UTC (`2026-07-03T14:20:00Z`).
- **Money/none:** N/A. **Masses** in grams, **energy** in kcal.
- **IDs:** integers (see `02`).
- **Errors:** uniform envelope (see §9).
- **Rounding:** the server rounds nutrition to 1 decimal and calories to whole kcal in responses.

### Standard status codes
| Code | Meaning |
|------|---------|
| 200 | OK |
| 201 | Created |
| 400 | Validation / bad input |
| 401 | Missing/invalid/expired token |
| 403 | Authenticated but not allowed (e.g. other user's resource) |
| 404 | Not found |
| 409 | Conflict (duplicate email/username) |
| 413 | Payload too large (image > 10 MB) |
| 422 | Unprocessable (FastAPI validation) |
| 423 | Locked (account locked after 5 fails) |
| 429 | Rate limited (USDA or login throttle) |
| 502 | Upstream USDA error with no cache fallback |
| 500 | Unexpected server error |

---

## 2. Auth — `/auth`

### POST `/auth/register` — FR01
Create an account. Public.

Request:
```json
{
  "username": "mohannad",
  "email": "user@example.com",
  "password": "Str0ngPass!",
  "health_goal": "weight_loss"
}
```
Validation: `username` 3–50 chars unique; `email` valid + unique; `password` ≥ 8 chars (must
contain letter + digit); `health_goal` ∈ {weight_loss, maintenance, weight_gain}.

Response `201`:
```json
{ "user_id": 12, "username": "mohannad", "email": "user@example.com",
  "health_goal": "weight_loss", "created_at": "2026-07-03T14:20:00Z" }
```
Errors: `409` duplicate email/username; `422` weak/invalid fields.

### POST `/auth/login` — FR02
Exchange credentials for a JWT. Public.

Request:
```json
{ "email": "user@example.com", "password": "Str0ngPass!" }
```
Behaviour:
- On success: reset `failed_login_count = 0`, return token.
- On wrong password: increment `failed_login_count`; at 5 set `account_locked = TRUE`.
- If `account_locked`: return `423` regardless of password.

Response `200`:
```json
{ "access_token": "eyJhbGc...", "token_type": "bearer", "expires_in": 3600,
  "user": { "user_id": 12, "username": "mohannad", "health_goal": "weight_loss" } }
```
Errors: `401` bad credentials (body: `remaining_attempts`); `423` locked.

> Account-unlock flow for GP2: admin/manual (or a dev `POST /auth/unlock` guarded by `APP_ENV`).
> A full email-reset flow is out of scope (see `00 §Non-goals`).

---

## 3. Users — `/users`

### GET `/users/me`
Current profile. **Auth.**
```json
{ "user_id": 12, "username": "mohannad", "email": "user@example.com",
  "health_goal": "weight_loss", "created_at": "2026-07-03T14:20:00Z" }
```

### PATCH `/users/me/goal` — FR13
Update health goal.
Request: `{ "health_goal": "maintenance" }` → `200` returns updated profile.
Side effect: next recommendation refresh uses the new target distribution (`06`).

### DELETE `/users/me` — FR14
Irreversibly delete the account and all owned data (cascade per `02`). **Auth.**
Request body: `{ "confirm": true }` (must be literally true, else `400`).
Response `200`: `{ "deleted": true }`. Token is invalid thereafter.

---

## 4. Recognition — `/recognise` — FR03, FR04, FR05

Single call that runs classification **and** portion estimation (steps 3–6 of the core loop in `01`).
**Auth.** `multipart/form-data`.

Form fields:
| Field | Type | Notes |
|-------|------|-------|
| image | file | JPEG or PNG, ≤ 10 MB, ≥ 640×480 (FR03) |

Server validation → `400`/`413` as appropriate (see §9). Image is processed in memory and discarded.

Response `200`:
```json
{
  "request_id": "b3f1...",
  "items": [
    {
      "rank": 1,
      "food101_class": "grilled_salmon",
      "display_name": "Grilled Salmon",
      "confidence": 0.91,
      "portion": { "min_g": 180.0, "max_g": 240.0, "point_g": 210.0,
                   "method": "reference_object", "reference": "dinner_plate" }
    }
  ],
  "low_confidence": false,
  "alternatives": []
}
```
FR04 rule: if top-1 `confidence < CONFIDENCE_THRESHOLD` (default 0.60), set `low_confidence: true`
and populate `alternatives` with up to 3 candidates `{food101_class, display_name, confidence}` for
the user to choose from.

Portion object (FR05): always a **range**; `point_g` is a convenience midpoint for the slider default.
`method` ∈ {`reference_object`, `container_prior`, `shape_only`} documents which stage produced the
estimate (see `05`). Never returns a bare single value without a range.

Errors: `400` bad type/resolution; `413` too large; `422` no image; `500` inference failure.

---

## 5. Nutrition — `/nutrition` — FR07, FR08

Retrieve macronutrients for a confirmed item, scaled to the confirmed portion. **Auth.**

Request:
```json
{
  "food101_class": "grilled_salmon",
  "cooking_method": "grilled",
  "portion_g": 210.0
}
```
`cooking_method` ∈ predefined list (see §10) and is appended to the USDA query (FR07).
Behaviour (see `08` for the client): look up `(food101_class, cooking_method)` in `food_item`
cache; on miss call USDA FDC, map to macros/100 g, upsert cache; scale to `portion_g`.

Response `200`:
```json
{
  "food_id": 45,
  "food_name": "Fish, salmon, Atlantic, cooked, dry heat",
  "usda_fdc_id": "175168",
  "cooking_method": "grilled",
  "per_100g": { "calories": 206, "protein_g": 22.1, "carbs_g": 0.0, "fat_g": 12.4 },
  "portion_g": 210.0,
  "scaled":   { "calories": 433, "protein_g": 46.4, "carbs_g": 0.0, "fat_g": 26.0 },
  "source": "usda",              // "usda" | "cache"
  "disclaimer": "Nutritional values are estimates for informational purposes only and do not constitute medical advice."
}
```
Errors: `429` USDA rate limit hit and no cache; `502` USDA error and no cache; `404` no USDA match.

---

## 6. Meals — `/meals` — FR10, FR11

### POST `/meals` — FR10
Persist a confirmed meal. **Auth.** Server recomputes `total_calories` (never trusts client).
Image is already discarded before this call.

Request:
```json
{
  "meal_type": "lunch",
  "items": [
    {
      "food101_class": "grilled_salmon",
      "cooking_method": "grilled",
      "portion_confirmed_g": 210.0,
      "portion_min_g": 180.0,
      "portion_max_g": 240.0,
      "confidence_score": 0.91
    }
  ]
}
```
Server: resolves/creates each `food_item`, fetches macros (cache-first), computes per-item `*_est`
and the meal `total_calories`, sets `user_confirmed = TRUE`.

Response `201`:
```json
{ "log_id": 87, "logged_at": "2026-07-03T14:25:00Z", "meal_type": "lunch",
  "total_calories": 433, "items": [ { "item_id": 140, "food_name": "…", "calories_est": 433,
  "protein_est_g": 46.4, "carbs_est_g": 0.0, "fat_est_g": 26.0, "portion_confirmed_g": 210.0 } ],
  "meals_logged_total": 8, "recommendations_unlocked": true }
```
`recommendations_unlocked` = (`meals_logged_total ≥ MIN_MEALS_FOR_RECOMMENDATION`).

### GET `/meals` — FR11
Chronological history (newest first), paginated. **Auth.** Only `user_confirmed = TRUE`.
Query params: `?limit=20&offset=0&from=<iso>&to=<iso>` (all optional).
```json
{ "total": 8, "limit": 20, "offset": 0,
  "meals": [ { "log_id": 87, "logged_at": "…", "meal_type": "lunch", "total_calories": 433,
               "item_count": 1, "items": [ … ] } ] }
```

### GET `/meals/{log_id}`
Single meal detail. **Auth.** `403` if not owner; `404` if missing.

### DELETE `/meals/{log_id}`
Delete one meal (cascade removes its items). **Auth.** `403` if not owner. `200 {"deleted": true}`.

---

## 7. Recommendations — `/recommendations` — FR12

### GET `/recommendations`
Return the user's current insights (≤ 5). **Auth.**
- If `confirmed meal count < 7`: `200` with `unlocked:false` and a progress hint — **not** an error.
```json
{ "unlocked": false, "meals_logged": 4, "meals_required": 7,
  "message": "Log 3 more meals to unlock personalised insights.", "insights": [] }
```
- If unlocked:
```json
{ "unlocked": true, "generated_at": "2026-07-03T14:30:00Z",
  "insights": [
    { "rec_id": 5, "insight_text": "Your protein intake is below target for muscle gain. Consider adding lean protein like chicken breast or Greek yogurt to your next meals.",
      "nutrient_focus": "protein", "was_read": false } ],
  "disclaimer": "Nutritional information and dietary suggestions are for informational purposes only and do not constitute medical or clinical nutritional advice. Consult a qualified healthcare professional for specific dietary needs." }
```

### POST `/recommendations/refresh`
Force regeneration from the latest 7-day window. **Auth.** Runs the engine in `06`. `200` with the
same shape as GET (unlocked case). No-op-safe if locked.

### PATCH `/recommendations/{rec_id}/read`
Mark an insight read. `200`.

---

## 8. Health & meta

- `GET /health` → `{ "status": "ok", "model_loaded": true, "db": "ok" }` (public; for load balancer).
- `GET /api/v1/meta/cooking-methods` → the enum in §10 (so the app dropdown stays in sync).
- `GET /api/v1/meta/food-classes` → the 101 class labels + display names.

---

## 9. Error envelope

Every non-2xx returns:
```json
{ "error": { "code": "IMAGE_TOO_LARGE", "message": "Image exceeds the 10 MB limit.",
             "details": { "max_mb": 10, "received_mb": 14.2 } } }
```
Canonical `code` values: `VALIDATION_ERROR`, `INVALID_CREDENTIALS`, `ACCOUNT_LOCKED`,
`DUPLICATE_EMAIL`, `DUPLICATE_USERNAME`, `UNAUTHORIZED`, `FORBIDDEN`, `NOT_FOUND`,
`IMAGE_TOO_LARGE`, `IMAGE_BAD_FORMAT`, `IMAGE_TOO_SMALL`, `INFERENCE_FAILED`,
`USDA_RATE_LIMITED`, `USDA_UNAVAILABLE`, `NO_NUTRITION_MATCH`, `INTERNAL_ERROR`.

## 10. Enumerations (single source of truth)

- `health_goal`: `weight_loss`, `maintenance`, `weight_gain`
- `meal_type`: `breakfast`, `lunch`, `dinner`, `snack`
- `cooking_method`: `grilled`, `fried`, `boiled`, `baked`, `roasted`, `steamed`, `raw`, `other`
- `portion.method`: `reference_object`, `container_prior`, `shape_only`
- `nutrition.source`: `usda`, `cache`

## 11. Endpoint → FR → milestone map

| Endpoint | FR | Milestone |
|----------|----|-----------|
| POST /auth/register | FR01 | M2 |
| POST /auth/login | FR02 | M2 |
| POST /recognise | FR03, FR04, FR05 | M3–M4 |
| POST /nutrition | FR07, FR08 | M5 |
| POST /meals | FR10 | M5 |
| GET /meals, /meals/{id} | FR11 | M6 |
| PATCH /users/me/goal | FR13 | M6 |
| GET/POST /recommendations | FR12 | M7 |
| DELETE /users/me | FR14 | M8 |

---

Prev: [`02-database-schema.md`](./02-database-schema.md) · Next: [`04-ml-model.md`](./04-ml-model.md)
