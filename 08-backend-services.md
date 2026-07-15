# 08 — Backend Services & Modules

FastAPI backend. This doc specifies each service module, the USDA integration, caching, rate
limiting, config, error handling, and deployment. Folder layout is in `01 §3`; endpoints in `03`.

---

## 1. App bootstrap — `app/main.py`

- Create the FastAPI app with a **lifespan** that loads models once at startup:
  `FoodClassifier` (`04`), `PortionEstimator` (`05`), and opens the DB engine.
- Mount routers under `/api/v1`: `auth, users, recognise, nutrition, meals, recommendations, meta`.
- Register exception handlers (`§7`), CORS (allow the app origin/dev IP), and request-ID middleware.
- `GET /health` returns `{status, model_loaded, db}`.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = FoodClassifier(settings.classifier_path, settings.labels_path, settings.device)
    app.state.portion = PortionEstimator.load(settings.model_dir)
    yield
    # nothing to persist; models are read-only
```

## 2. Config — `app/config.py`
`pydantic-settings` `BaseSettings` reads the env vars from `01 §4`. Fail fast at startup if any
required secret (`SECRET_KEY`, `DATABASE_URL`, `USDA_API_KEY`) is missing.

## 3. Security — `app/security.py`
- `hash_password(pw)` / `verify_password(pw, hash)` via `passlib` bcrypt.
- `create_access_token(sub, expires)` / `decode_token(token)` via `python-jose`, HS256, `SECRET_KEY`.
- `get_current_user` dependency (`app/deps.py`): extracts Bearer token → decodes → loads `User` →
  401 on failure. Used by every protected route.

## 4. Services (business logic; routers stay thin)

### `auth_service.py` — FR01, FR02
- `register(data)`: uniqueness checks (→ 409), hash password, insert, return profile.
- `login(email, pw)`: load user; if `account_locked` → 423; verify password; on fail increment
  `failed_login_count`, lock at `MAX_FAILED_LOGINS` (5); on success reset counter, issue JWT.

### `recognition_service.py` — FR03–FR05
- Validate the upload (type ∈ {jpeg,png}, size ≤ `MAX_UPLOAD_MB`, dims ≥ 640×480) → structured errors.
- Decode to PIL **in memory**; run `preprocess` → `classifier.predict(top_k=3)`.
- For the top item(s), run `portion.estimate(...)`.
- Apply the FR04 threshold rule (top-1 vs top-3 alternatives).
- **Discard the image** (no writes; let it go out of scope). Return the `/recognise` payload (`03 §4`).

### `nutrition_service.py` — FR07, FR08
- Input: `food101_class`, `cooking_method`, `portion_g`.
- Cache-first: look up `food_item` by `(food101_class, cooking_method)`. Hit → use cached per-100g.
- Miss → `usda_client.search(...)` (`§5`), map to macros/100 g, **upsert** into `food_item`.
- Scale to `portion_g`, round, attach `source` and the disclaimer. Errors: 429/502/404 per `03 §5`.

### `meal_service.py` — FR10, FR11
- `create_meal(user, payload)`: resolve/create each `food_item`, fetch macros (reuse nutrition_service,
  cache-first), compute per-item `*_est`, sum `total_calories`, insert `meal_log` + `meal_item`s with
  `user_confirmed=TRUE`. Return created meal + `meals_logged_total` + `recommendations_unlocked`.
- `list_meals(user, filters)`: confirmed only, newest first, paginated.
- `get_meal` / `delete_meal`: ownership-checked (403 otherwise).

### `recommendation_service.py` — FR12
- As specified in `06`. Triggered after qualifying `POST /meals` and via `/recommendations/refresh`.

### `user_service.py` — FR13, FR14
- `update_goal(user, goal)`; `delete_account(user)` → single cascade delete (`02`), irreversible.

## 5. USDA integration — `app/integrations/usda_client.py`

- Async `httpx` client against `USDA_BASE_URL` (`https://api.nal.usda.gov/fdc/v1`), `api_key` from env.
- **Search:** `GET /foods/search?query=<mapped term>&dataType=Foundation,SR Legacy&pageSize=5`.
  Build the query by mapping the Food-101 class + cooking method to a search string
  (e.g. `grilled_salmon` + `grilled` → "salmon cooked grilled"). Keep a small mapping table for the
  trickier classes in `app/integrations/usda_query_map.py`.
- **Select best match:** prefer `Foundation`/`SR Legacy` data types (lab-verified) over `Branded`;
  pick the top-scored result; extract per-100 g values for energy (kcal), protein, carbohydrate, fat
  from the `foodNutrients` array (nutrient numbers: energy 1008/208, protein 1003/203, carb 1005/205,
  fat 1004/204 — handle both id schemes).
- **Normalisation:** USDA returns per 100 g already for these datasets; assert units and store.
- **Errors:** timeout/5xx → raise `UsdaUnavailable`; 429 → `UsdaRateLimited`; no match → `NoNutritionMatch`.
  The caller decides cache-fallback vs error (`§6`).

## 6. Caching & fallback — `app/integrations/nutrition_cache.py`

- The `food_item` table **is** the cache (`02 §2.3`), keyed by `(food101_class, cooking_method)`.
- Read path: cache hit → return immediately (`source:"cache"`). This also satisfies the Reliability
  NFR: if USDA is down but the value was cached, the user still gets nutrition.
- Write path: on USDA success, upsert (update if the row exists with null macros from seeding).
- Freshness: `cached_at` supports an optional TTL (e.g. refresh if older than 180 days); not required
  for GP2 but leave the hook.
- Fallback order when USDA errors: cached value → else the documented error (`429`/`502`). **Never
  fabricate.**

## 7. Errors, logging, middleware — `app/core/`

- `errors.py`: typed exceptions (`ValidationError`, `AccountLocked`, `Duplicate*`, `ImageTooLarge`,
  `ImageBadFormat`, `ImageTooSmall`, `InferenceFailed`, `UsdaRateLimited`, `UsdaUnavailable`,
  `NoNutritionMatch`, `NotFound`, `Forbidden`) → mapped to the envelope + status in `03 §9` via a
  single exception handler.
- `logging.py`: structured JSON logs; attach a `request_id` (uuid) per request; **never log image
  bytes, passwords, or tokens**.
- `rate_limit.py`: (a) USDA hourly cap (`USDA_RATE_LIMIT_PER_HOUR`, default 3600) enforced by a
  request counter (per report §4.5); (b) login throttle to blunt brute force (e.g. 10/min/IP).

## 8. Preprocessing — `app/ml/preprocess.py`
Decode bytes → PIL RGB → the **eval transform** (Resize 400 → CenterCrop 380 → ToTensor →
ImageNet-normalise), identical to `04 §2` so serving matches training. Returns a `1×3×380×380` tensor.
Also exposes the raw PIL image (for the portion pipeline's segmentation) — both live in memory only.

## 9. Testing hooks
Every service is import-testable without a running server (pass a DB session + fakes). The classifier
and USDA client are injected so tests can stub them (`11`). Provide `--seed` scripts (`02 §5`).

## 10. Deployment (evaluation)

- **Local dev:** `docker-compose up` brings Postgres + backend; app points at the host IP.
- **Eval VM:** single Ubuntu 22.04 cloud instance (Google Cloud / equivalent). Gunicorn + Uvicorn
  workers behind Nginx (TLS termination → HTTPS, satisfying the Security NFR). Env via `.env` on the
  box (not committed). Model `.pth` pulled from the GitHub Release at deploy time.
- **DB migrations:** `alembic upgrade head` on deploy.
- **Health:** Nginx/monitor hits `GET /health`.
- Model rollback = redeploy the previous GitHub Release artifact (Maintainability NFR).

---

Prev: [`07-mobile-app.md`](./07-mobile-app.md) · Next: [`09-build-phases.md`](./09-build-phases.md)
