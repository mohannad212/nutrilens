# 01 — Architecture & Tech Stack

---

## 1. Architectural style

Three-tier, layered, with a clean separation the app owner already defended in Chapter 5:

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENT TIER                                                  │
│  React Native app (Android 8.0+, iOS 13+)                    │
│  - Camera/gallery capture   - Screens & navigation           │
│  - API client (JWT)         - Local UI state only            │
└───────────────┬─────────────────────────────────────────────┘
                │ HTTPS / REST / JSON  (JWT in Authorization header)
┌───────────────▼─────────────────────────────────────────────┐
│  BACKEND TIER  (FastAPI, Python 3.11)                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ API Gateway │→ │ ML Pipeline  │  │ Data Services       │   │
│  │ auth, routes│  │ preprocess   │  │ usda_client         │   │
│  │ rate limit  │  │ classifier   │  │ nutrition_cache     │   │
│  │             │  │ portion      │  │ db (SQLAlchemy ORM) │   │
│  │             │  │ recommender  │  │                     │   │
│  └────────────┘  └──────────────┘  └────────────────────┘    │
└───────────┬───────────────────────┬─────────────────────────┘
            │                        │  outbound HTTPS
┌───────────▼──────────┐   ┌─────────▼───────────┐   ┌──────────────┐
│  PostgreSQL 15        │   │  USDA FoodData Central│  │ Model Store  │
│  (users, meals, etc.) │   │  REST API             │  │ (.pth files) │
└──────────────────────┘   └─────────────────────┘   └──────────────┘
```

**Key principles**
- The **client is thin**: no business logic, no nutrition math, no secrets. It renders and calls the API.
- The **backend owns all logic and all secrets** (USDA API key, DB creds, JWT secret).
- **Images never touch disk.** They flow request → memory → pipeline → discarded. Only structured
  results persist (see `02`).
- Inter-module calls inside the backend are **synchronous function calls**; the recommendation
  engine runs **asynchronously** off stored history (not on the request path).

## 2. Technology stack (exact versions)

### Backend
| Concern | Choice | Version | Notes |
|---------|--------|---------|-------|
| Language | Python | 3.11.x | type hints everywhere |
| Web framework | **FastAPI** | 0.110+ | async, automatic OpenAPI docs at `/docs` |
| ASGI server | Uvicorn (+ Gunicorn workers in prod) | latest | `uvicorn[standard]` |
| ORM | SQLAlchemy | 2.0+ | declarative models, `mapped_column` |
| Migrations | Alembic | 1.13+ | versioned schema |
| Validation | Pydantic | v2 | request/response schemas |
| DL runtime | PyTorch | 2.2+ | inference on backend |
| Vision utils | torchvision | matching torch | EfficientNet weights, transforms |
| CV | OpenCV (`opencv-python-headless`) | 4.9+ | portion pipeline |
| Image IO | Pillow | 10+ | decode/resize |
| Auth | `python-jose[cryptography]` (JWT), `passlib[bcrypt]` | latest | tokens + hashing |
| HTTP client | `httpx` | latest | async USDA calls |
| DB driver | `psycopg[binary]` | 3.x | PostgreSQL |
| Config | `pydantic-settings` | v2 | env-based settings |
| Testing | pytest, pytest-asyncio, httpx TestClient | latest | see `11` |
| Lint/format | ruff, black, mypy | latest | see `10` |

### Model training (offline, separate from serving)
| Concern | Choice | Notes |
|---------|--------|-------|
| Environment | Google Colab Pro / Kaggle | NVIDIA GPU, ≥16 GB VRAM |
| Framework | PyTorch 2.2+, torchvision | transfer learning |
| Tracking | TensorBoard (or Weights & Biases optional) | loss/acc curves |
| Export | TorchScript **or** state_dict `.pth` | consumed by backend `04` |

### Mobile app
| Concern | Choice | Version | Notes |
|---------|--------|---------|-------|
| Framework | **React Native** | 0.73+ | bare or Expo (Expo recommended for camera) |
| Language | TypeScript | 5.x | strict mode |
| Navigation | React Navigation | 6.x | native-stack + bottom-tabs |
| Data fetching | TanStack Query (React Query) | 5.x | caching, retries, loading states |
| HTTP | axios | latest | interceptors for JWT |
| Secure storage | expo-secure-store (or react-native-keychain) | latest | JWT storage |
| Camera | expo-camera / expo-image-picker | latest | capture + gallery |
| Charts | react-native-svg + custom, or victory-native | latest | calorie ring, macro bars |
| Forms | react-hook-form | latest | registration/login |
| Lint/format | ESLint + Prettier | latest | see `10` |

### Infrastructure
| Concern | Choice | Notes |
|---------|--------|-------|
| Database | PostgreSQL | 15 |
| Containerisation | Docker + docker-compose | one command local bring-up |
| Deploy target (eval) | Single Ubuntu 22.04 cloud VM | see `08 §Deployment` |
| Version control | GitHub (Git) | `main` / `develop` / `feature/*` (see `10`) |
| Model artifacts | GitHub Releases (versioned `.pth`) | keep large binaries out of git history |

> **Flask vs FastAPI.** The Phase-1 report allows "Flask 3.x or FastAPI 0.100+". **Use FastAPI**
> for GP2: async USDA calls, Pydantic validation, and auto-generated OpenAPI docs materially
> reduce build effort. This is a permitted, documented choice.

## 3. Monorepo folder structure

```
nutrilens/
├── README.md
├── docker-compose.yml              # postgres + backend for local dev
├── .env.example                    # documented; never commit real .env
├── .github/workflows/ci.yml        # lint + test (see 10, 11)
│
├── backend/
│   ├── pyproject.toml              # deps, ruff/black/mypy config
│   ├── alembic.ini
│   ├── alembic/                    # migrations (see 02)
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory, router mount
│   │   ├── config.py               # pydantic-settings (env)
│   │   ├── deps.py                 # DI: db session, current_user
│   │   ├── security.py             # JWT create/verify, bcrypt
│   │   ├── db/
│   │   │   ├── base.py             # SQLAlchemy Base, engine, session
│   │   │   └── models.py           # ORM models (see 02)
│   │   ├── schemas/                # Pydantic request/response (see 03)
│   │   │   ├── auth.py  user.py  meal.py  nutrition.py  recommendation.py
│   │   ├── api/                    # routers (see 03, 08)
│   │   │   ├── auth.py  users.py  recognise.py  nutrition.py  meals.py  recommendations.py
│   │   ├── services/               # business logic (see 08)
│   │   │   ├── auth_service.py
│   │   │   ├── recognition_service.py
│   │   │   ├── nutrition_service.py
│   │   │   ├── meal_service.py
│   │   │   └── recommendation_service.py
│   │   ├── ml/                     # inference-time ML (see 04, 05)
│   │   │   ├── preprocess.py
│   │   │   ├── classifier.py       # EfficientNet-B4 load + predict
│   │   │   ├── portion/            # 3-stage pipeline (see 05)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── container.py    # stage 1
│   │   │   │   ├── reference.py    # stage 2
│   │   │   │   ├── shape_prior.py  # stage 3
│   │   │   │   └── densities.py    # food-class density table
│   │   │   └── recommender.py      # (see 06)
│   │   ├── integrations/
│   │   │   ├── usda_client.py      # USDA FDC adapter (see 08)
│   │   │   └── nutrition_cache.py  # cache layer (see 08)
│   │   └── core/
│   │       ├── errors.py           # exception types + handlers
│   │       ├── rate_limit.py       # USDA + login rate limiting
│   │       └── logging.py
│   ├── models/                     # downloaded/exported weights (gitignored)
│   │   ├── food101_efficientnet_b4.pth
│   │   ├── container_classifier.pth
│   │   └── labels_food101.json
│   └── tests/                      # pytest (see 11)
│
├── training/                       # OFFLINE model training (see 04, 05)
│   ├── food101_train.ipynb         # Colab notebook — primary
│   ├── food101_train.py            # script version
│   ├── portion/                    # container-classifier training, calibration data
│   ├── configs/                    # yaml hyperparameters
│   └── README.md
│
└── mobile/                         # React Native app (see 07)
    ├── package.json
    ├── app.json                    # expo config
    ├── tsconfig.json
    ├── src/
    │   ├── App.tsx
    │   ├── navigation/             # stacks + tabs
    │   ├── screens/                # 9 screens (see 07)
    │   ├── components/             # reusable UI
    │   ├── api/                    # axios client + typed endpoints
    │   ├── hooks/                  # React Query hooks
    │   ├── store/                  # auth context / state
    │   ├── theme/                  # design tokens (see 07)
    │   └── types/                  # shared TS types (mirror 03 schemas)
    └── tests/
```

## 4. Configuration & secrets

All configuration is environment-based. **Never hard-code secrets; never commit `.env`.**
`.env.example` documents every variable:

```dotenv
# ---- Backend ----
APP_ENV=development                     # development | production
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=change-me-32+chars           # JWT signing (HS256)
ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_ALGORITHM=HS256

# ---- Database ----
DATABASE_URL=postgresql+psycopg://nutrilens:password@localhost:5432/nutrilens

# ---- USDA FoodData Central ----
USDA_API_KEY=your-usda-key              # free key from https://fdc.nal.usda.gov/api-key-signup
USDA_BASE_URL=https://api.nal.usda.gov/fdc/v1
USDA_RATE_LIMIT_PER_HOUR=3600           # per NFR / report §4.5

# ---- ML ----
MODEL_DIR=./models
CLASSIFIER_WEIGHTS=food101_efficientnet_b4.pth
CONFIDENCE_THRESHOLD=0.60               # below this → return top-3 (FR04)
DEVICE=cpu                              # cpu | cuda

# ---- Security policy ----
MAX_FAILED_LOGINS=5                     # FR02 lockout
MAX_UPLOAD_MB=10                        # FR03
MIN_IMAGE_WIDTH=640
MIN_IMAGE_HEIGHT=480

# ---- Recommendations ----
MIN_MEALS_FOR_RECOMMENDATION=7          # FR10 / FR12
RECOMMENDATION_WINDOW_DAYS=7
```

The mobile app has its own config (`app.json` / `.env` via `expo-constants`):
```
API_BASE_URL=http://<dev-machine-ip>:8000   # dev
```

## 5. Request data-flow (the core loop, precisely)

```
1. App: POST /recognise  (multipart: image)          [JWT required]
2. Gateway: validate JWT, size/type/resolution (FR03)
3. preprocess.py: decode → resize 380×380 → normalise → tensor      (image in RAM only)
4. classifier.py: EfficientNet-B4 → top-k labels + confidences (FR04)
5. portion/*: 3-stage pipeline → {min_g, max_g} per item (FR05)
6. Response → app: predictions + portion range (image discarded from memory)
7. App: user confirms/adjusts portion (FR06), picks cooking method (FR07)
8. App: POST /nutrition {food_id/label, cooking_method, portion_g}
9. nutrition_service → usda_client (or cache) → macros per 100g → scale to portion (FR08)
10. Response → app: calories/protein/carbs/fat  → Nutrition screen with disclaimer (FR09)
11. App: POST /meals  (confirmed items) → persisted; image already gone (FR10)
12. If user has ≥7 logs → recommendation_service refreshes insights (FR12)
```

Every step's contract is pinned in `03-api-specification.md`.

---

Prev: [`00-product-overview.md`](./00-product-overview.md) · Next: [`02-database-schema.md`](./02-database-schema.md)
