# HANDOFF — building the NutriLens mobile app

You're picking up the **mobile app**. The backend is essentially finished: every
endpoint the app needs is built, tested and running. This document is what you
need to get from a cold clone to a running app, plus the four rules that keep new
screens from silently breaking the API contract.

Start from **`develop`**. It is the current state of the project.

---

## 1. Setup, from nothing

### Prerequisites
Docker Desktop, Python 3.11 or 3.12, Node 18+, and the Expo Go app on your phone
(or Xcode / Android Studio for a simulator).

### Clone
```bash
git clone https://github.com/mohannad212/nutrilens.git
cd nutrilens
git checkout develop
```

### Backend secrets
`.env` is gitignored — you need your own. Copy the template and fill it in:
```bash
cp .env.example .env
```
Two values need real content:

- **`SECRET_KEY`** — any long random string. `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- **`USDA_API_KEY`** — get your own free key (about a minute) at
  <https://fdc.nal.usda.gov/api-key-signup>. Please don't reuse Mohannad's: the
  quota is hourly and per-key, so sharing means you throttle each other.

Leave `DATABASE_URL` as it is; it matches the Docker Postgres below.

> `.env` lives at the **repo root**, not in `backend/`. `config.py` reads
> `("../.env", ".env")` so it resolves from either working directory.

### Database
```bash
docker compose up -d db          # Postgres 15 on :5432
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
alembic upgrade head             # creates all 5 tables
python scripts/seed_food101_classes.py   # 101 food classes
```

### Model weights (needed for the camera flow)
The three `.pth` files are ~118 MB and **not in git**. Without them the backend
still starts, but `POST /recognise` returns `500 INFERENCE_FAILED` — which is the
endpoint the Camera screen calls, so you'll want them. See
[`backend/models/README.md`](backend/models/README.md); ask Mohannad for the
files or the Release link.

Everything else works without them, including the whole nutrition and logging
flow you can already see on the Nutrition Summary screen.

### Run the backend
```bash
cd backend && ./.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Check <http://localhost:8000/health> → `{"status":"ok"}`, and
<http://localhost:8000/docs> for interactive API docs — that page is the fastest
way to see any endpoint's exact request and response shape.

### Run the app
```bash
cd mobile
npm install
cp .env.example .env      # then set the URL for your device
npm start
```
`EXPO_PUBLIC_API_BASE_URL` must be reachable **from the device**, not from your
laptop:

| Running on | Value |
|---|---|
| Android emulator | `http://10.0.2.2:8000/api/v1` |
| iOS simulator | `http://localhost:8000/api/v1` |
| Physical phone | `http://<your laptop's LAN IP>:8000/api/v1` |

The `EXPO_PUBLIC_` prefix is required — Expo only inlines prefixed variables, so
a plain `API_BASE_URL` is invisible at runtime.

### Sanity check
```bash
cd backend && ./.venv/bin/python -m pytest -q     # expect 472 passed
cd mobile && npx tsc --noEmit && npx eslint src --ext .ts,.tsx
```

---

## 2. What exists, and what's yours

### Backend — done
Every endpoint in [`03-api-specification.md`](03-api-specification.md) except
`DELETE /users/me` (M8) and `GET /meta/food-classes`. 472 tests, ruff + black +
mypy clean. **You shouldn't need to write backend code.** If something seems
missing, check `/docs` first — it's probably there.

### Mobile — the work

| Screen | State | Calls |
|---|---|---|
| **Login / Register** | ⬜ stub — **start here** | `POST /auth/login`, `POST /auth/register` |
| **Home / Dashboard** | ⬜ stub | `GET /meals?from=<today>` |
| **Camera Capture** | ⬜ stub (has a `__DEV__` shortcut — delete it) | `POST /recognise` |
| **Recognition Results** | ⬜ stub | uses the `/recognise` response |
| **Portion** | ⬜ stub (can be a section of Recognition Results) | local, then `/nutrition` |
| **Nutrition Summary** | ✅ **built** — use as the worked example | `POST /nutrition`, `POST /meals` |
| **Meal History** | ⬜ stub | `GET /meals`, `DELETE /meals/{id}` |
| **Profile & Goals** | ⬜ stub | `GET /users/me`, `PATCH /users/me/goal`, `PATCH /users/me/metrics` |
| **Insights** | ⬜ stub | `GET /recommendations` |

**Login/Register first.** Nothing else is reachable until you can sign in — which
is why `Capture.tsx` currently has a `__DEV__`-only shortcut into the Nutrition
screen. Delete that block once the real flow works.

### Already built for you
```
src/api/client.ts        axios instance: attaches the JWT, normalises every error
src/api/endpoints.ts     typed endpoint wrappers — add new ones here
src/types/api.ts         TypeScript types bound field-for-field to the backend
src/store/AuthContext.tsx  session + expo-secure-store (login/logout/isRestoring)
src/hooks/useNutrition.ts  the React Query pattern to copy
src/components/          CalorieRing, MacroBar, CookingMethodPicker,
                         PrimaryButton, DisclaimerBanner, Screen/Loading/Error/Empty
src/theme/index.ts       all colours, spacing, radii
```

---

## 3. Four rules for new screens

The design mockups (see `design/`) are the **visual target**. Whatever you use to
get there — hand-written, ChatGPT, Stitch's own export — the result has to respect
four things, because generated code reliably breaks all of them.

**1. No hex colours outside `theme/index.ts`.**
Generated code hard-codes `#3DDC97`. Use `theme.color.accent`. The palette is
specified in [`07-mobile-app.md §8`](07-mobile-app.md) and the macro colours
(protein blue, carbs amber, fat pink) must stay consistent across every screen.

**2. Never call `fetch`/`axios` from a screen.**
Add a wrapper to `src/api/endpoints.ts` and a hook in `src/hooks/`. The shared
client attaches the JWT and converts failures into one `ApiError` with a `code`.
A screen that calls `fetch` directly is unauthenticated and has no error handling.

**3. Types come from `src/types/api.ts`.**
[`03-api-specification.md`](03-api-specification.md) is a *contract*: those types
match the backend's Pydantic schemas field for field. Don't hand-write a response
interface inside a screen — if it drifts, nothing catches it and the app breaks at
runtime against a backend that's behaving correctly.

**4. Reuse the components.**
`CalorieRing` and `MacroBar` already exist and are accessible (44 px touch
targets, screen-reader labels, colour never the only signal — `07 §9`). A second
implementation will be neither.

> **Read [`src/screens/NutritionSummary.tsx`](mobile/src/screens/NutritionSummary.tsx) first.**
> It's a complete screen doing all four correctly: query hook, typed request,
> component reuse, theme tokens, loading/error states, and the disclaimer rendered
> from the server's response rather than a local copy.

### One rule that isn't stylistic
The **disclaimer must appear on the Nutrition and Insights screens** — it's a
stated requirement (`07 §3`, `03 §5`, `§7`), and it's rendered from the API
response, never from a string in the app, so the two can't drift apart.

---

## 4. Where to look things up

| Question | File |
|---|---|
| What does this endpoint return? | `http://localhost:8000/docs`, or [`03-api-specification.md`](03-api-specification.md) |
| What should this screen look like / do? | [`07-mobile-app.md`](07-mobile-app.md) + `design/` |
| What's the colour / spacing? | `mobile/src/theme/index.ts` (from `07 §8`) |
| What's still left overall? | [`09-build-phases.md`](09-build-phases.md) — M8 and the app |
| Why is the backend built this way? | Every backend module has a long header docstring explaining it |

---

## 5. Gotchas that will cost you an hour

- **`.env` at the repo root**, not `backend/.env`.
- **`EXPO_PUBLIC_` prefix** — without it the app silently uses the default URL.
- **`localhost` from a physical phone** points at the phone. Use the LAN IP.
- **`POST /nutrition` is a POST but behaves like a read** — it's modelled as a
  React Query `useQuery` so changing the portion or cooking method refetches. See
  `useNutrition.ts` for why.
- **Never send calories or macros in `POST /meals`.** The request type has no
  field for them by design: the server computes every derived value (`02 §6`).
- **`npm test` isn't configured** — there's no jest setup. `npx tsc --noEmit` and
  eslint are the checks that exist.
- **A 401 anywhere logs the user out** (token expiry is 60 minutes). If you're
  bounced to Login mid-session, that's why.
