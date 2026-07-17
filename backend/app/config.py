"""
app/config.py
==============

WHY THIS FILE EXISTS
---------------------
Every setting NutriLens' backend needs to run (DB URL, JWT secret, USDA key,
lockout threshold...) lives in the environment, never in code. This file is
the single place that reads those environment variables, validates their
types, and hands back one typed `settings` object the rest of the app
imports. Nobody else in the codebase should call `os.getenv(...)` directly.

WHAT PROBLEM IT SOLVES
------------------------
Without this file you'd see `os.getenv("SECRET_KEY")` scattered across ten
files, each one silently returning `None` if the variable is missing, and
none of them agreeing on the type (is ACCESS_TOKEN_EXPIRE_MINUTES a str or
an int?). Centralising it means: one source of truth, fail-fast at startup
if something required is missing, and full type safety everywhere else.

HOW IT WORKS INTERNALLY
--------------------------
`pydantic-settings`'s `BaseSettings` reads from (in priority order) actual
process environment variables, then a `.env` file, then the field defaults
below. Pydantic parses and *validates* each value against its declared type
the moment the class is instantiated — so `ACCESS_TOKEN_EXPIRE_MINUTES=abc`
would crash immediately at startup instead of failing mysteriously three
requests later. The `settings = Settings()` singleton at the bottom is
created once, at import time, and reused everywhere (`from app.config
import settings`).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Plain `os.getenv` everywhere: no validation, no defaults, no types, easy
  to typo a variable name and get `None` silently.
- A hand-rolled `Config` class with `os.environ[...]`: works, but you
  reinvent validation, type coercion, and `.env` loading yourself.
- `pydantic-settings` (chosen): gets validation, typing, and `.env` support
  for free, and it's the same Pydantic already used for request/response
  schemas — one mental model for the whole project.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why not hard-code the DB URL for simplicity?"
A: Twelve-factor app principle — config that varies between environments
   (dev/staging/prod) must live in the environment, not in source control.
   It also means the same Docker image works in every environment.

Q: "What happens if SECRET_KEY is missing?"
A: Pydantic raises a `ValidationError` at import time (app fails to start),
   which is exactly what we want — a missing signing key should never
   silently fall back to something insecure.

Q: "Why pydantic-settings instead of python-dotenv directly?"
A: python-dotenv just loads variables into `os.environ`; it still leaves
   you to parse/validate types yourself. pydantic-settings wraps that and
   adds a typed schema on top.

COMMON STUDENT MISTAKES
--------------------------
- Committing a real `.env` file to git (always commit `.env.example` only,
  with placeholder values, and add `.env` to `.gitignore`).
- Reading `os.getenv` in random files "just this once" — breaks the single
  source of truth.
- Forgetting that `SECRET_KEY` must be a long random string in production,
  not the "change-me" placeholder.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- App ----
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---- Auth / JWT ----
    secret_key: str = "change-me-32-chars-minimum-please"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    max_failed_logins: int = 5

    # ---- Database ----
    database_url: str = "postgresql+psycopg://nutrilens:password@localhost:5432/nutrilens"

    # ---- USDA FoodData Central (used later, but declared now) ----
    usda_api_key: str = ""
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_rate_limit_per_hour: int = 3600

    # ---- ML / model serving (declared now, consumed by M3) ----
    model_dir: str = "./models"
    classifier_weights: str = "food101_efficientnet_b4.pth"
    confidence_threshold: float = 0.60
    device: str = "cpu"

    # ---- Upload / image validation (FR03, used by M3) ----
    max_upload_mb: int = 10
    min_image_width: int = 640
    min_image_height: int = 480

    # ---- Recommendations (used later by M7) ----
    min_meals_for_recommendation: int = 7
    recommendation_window_days: int = 7


# Single, importable instance — created once at process startup.
settings = Settings()


def get_settings() -> Settings:
    """Backwards-compatible accessor for the shared settings singleton."""
    return settings
