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
    # Two candidate .env files, because the backend legitimately runs from two
    # different working directories: `docker-compose` loads the repo-root `.env`
    # (see its `env_file:`), while local `uvicorn`/`pytest` run from `backend/`.
    # Listing both means one secret file at the repo root serves every entry point
    # — the alternative was copying USDA_API_KEY and SECRET_KEY into a second
    # gitignored file and keeping them in step by hand.
    #
    # Order matters: pydantic-settings gives *later* files priority, so a
    # `backend/.env` (if someone creates one) overrides the shared root file,
    # which is the more specific-wins behaviour you would expect. Real process
    # environment variables still beat both.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        extra="ignore",
    )

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

    # ---- USDA FoodData Central (M5; see 08 §5-6) ----
    usda_api_key: str = ""
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_rate_limit_per_hour: int = 3600
    # Per-request ceiling on the USDA call. Deliberately well under the 5 s
    # Performance NFR for the whole photo->nutrition path, since a lookup is only
    # one step of it and a slow upstream must not consume the entire budget.
    usda_timeout_seconds: float = 10.0
    # Cache freshness (08 §6). A cached food_item older than this is refreshed from
    # USDA on next use — but is still served if USDA is unreachable, which is the
    # whole point of tracking staleness separately from validity. 0 disables expiry.
    nutrition_cache_ttl_days: int = 180

    # ---- ML / model serving (declared now, consumed by M3) ----
    model_dir: str = "./models"
    classifier_weights: str = "food101_efficientnet_b4.pth"
    classifier_labels: str = "labels_food101.json"
    confidence_threshold: float = 0.60
    # Below this, the photo is treated as *not food at all* rather than as an
    # uncertain food. Food-101 is a closed set of 101 food classes with no
    # "not food" class, so softmax always names some food however unrelated the
    # subject — a laptop still comes back as "apple pie". A diffuse distribution
    # is the only signal available that the subject is out of distribution.
    # Kept well under `confidence_threshold` so a genuinely hard *food* photo
    # still reaches the FR04 pick-list instead of being refused outright.
    no_food_confidence_floor: float = 0.25

    # ---- Open-set food gate (see app/ml/food_gate.py) ----
    # The floor above is a backstop, not a solution: a laptop photographed close
    # up scored 0.46 on "apple pie", so no threshold on a closed-set score can
    # separate "not food" from "unsure which food". The gate asks a general
    # ImageNet model instead, which has actually seen laptops. Measured mass on
    # food-or-dining classes: laptop 0.0001, random noise 0.0074 — the default
    # sits far above both while staying permissive enough never to refuse a real
    # meal, which is the costlier mistake. Off => classification only, as before.
    food_gate_enabled: bool = True
    food_gate_min_prob: float = 0.05

    device: str = "cpu"

    # ---- Upload / image validation (FR03, used by M3) ----
    max_upload_mb: int = 10
    min_image_width: int = 640
    min_image_height: int = 480

    # ---- Image quality gate (FR03 extension; reject un-recognisable photos) ----
    # A *rejection* gate, not an enhancement: it never alters pixels (which would
    # cause train/serve skew), it only asks the user to retake a clearly bad
    # photo. Thresholds are deliberately conservative so real food photos always
    # pass; flip `quality_gate_enabled` off to disable during evaluation.
    quality_gate_enabled: bool = True
    brightness_min: float = 30.0  # mean luma 0-255; below this = too dark
    brightness_max: float = 235.0  # above this = washed out / overexposed
    blur_laplacian_min: float = 60.0  # variance of Laplacian; below this = too blurry

    # ---- Portion estimation (FR05, M4; see 05-portion-estimation.md) ----
    # Master switch: off => /recognise omits the `portion` object (classification
    # only, exactly like M3) so the endpoint still works if the pipeline is
    # disabled for evaluation.
    portion_enabled: bool = True
    # Symmetric uncertainty bands (± fraction) for the stages whose band is a fixed
    # constant (05 §4). A precise reference object gives a tight band; shape-only
    # (no scale cue at all) is the widest. NOTE: the container-prior band is NOT a
    # constant — it is derived from the container's own min/max size range (squared,
    # since mass ∝ size²) in `container_priors.uncertainty_band`, so it widens
    # honestly for vaguely-sized containers. `portion_eps_container` is retained
    # only as a reference/fallback value and is not used by the estimator.
    portion_eps_reference: float = 0.12  # method="reference_object"
    portion_eps_container: float = 0.25  # (reference only; band is range-derived)
    portion_eps_shape_only: float = 0.35  # method="shape_only"
    # Shape-only fallback (05 §4): with no reference and no container, assume the
    # image frame spans this many cm horizontally (typical arm's-length food
    # photo) to derive a coarse cm-per-pixel scale from the image width alone.
    portion_shape_only_frame_width_cm: float = 26.0
    # Two references agreeing within this fraction are averaged (05 §4).
    portion_reference_agree_tol: float = 0.15
    # Stage-1 container classifier (M4 aux model, trained in training/portion).
    # Loaded from `model_dir` at startup when present; absent -> portion falls back
    # to shape_only (graceful, like the food model). Same TorchScript convention.
    container_weights: str = "container_classifier.pth"
    container_labels: str = "labels_container.json"
    # Stage-2 reference-object detector (off-the-shelf COCO: fork/spoon/knife).
    # Enabling it downloads a pretrained torchvision detector once at startup
    # (~2 s load, ~0.15 s per request on CPU; startup degrades gracefully with
    # no network). On since the 2026-08 audit: cutlery in frame gives the
    # tightest, plate-independent scale (05 §4), and the audit's weighed fork
    # photo confirmed detection works. Flip off for offline evaluation.
    reference_detector_enabled: bool = True
    reference_score_threshold: float = 0.5
    # Credit-card reference (backlog §1). An ISO/IEC 7810 ID-1 card is 85.60 mm
    # exactly — the only zero-variance reference available, versus a fork's
    # assumed 19 cm (real forks 17-21 cm => ~20% mass error). Classical geometry,
    # no model, so it costs nothing to have on. The two thresholds are the
    # false-positive guard and are measured, not guessed: at 12% tolerance a
    # kitchen scale's display panel (ratio 1.42) was detected as a card; at 6%
    # plus an 80 px floor there are zero false positives across the weighed set.
    # Do not loosen either without re-running that audit
    # (`scripts/card_fp_audit.py`).
    # Stage-1 rim detection via surface growth from the food mask (backlog §4).
    # Runs only where the Otsu rim detector declines, so it cannot change a rim
    # that already resolved.
    #
    # DEFAULT OFF — IT WAS BUILT, MEASURED, AND IT MADE THINGS WORSE.
    # It lifts rim coverage on the weighed set from 3/6 to 6/6, but the rims it
    # finds are wrong, and a wrong ruler is worse than none: MAPE 18.2% -> 37.5%,
    # range coverage 66.7% -> 33.3%. Inspecting the overlays (the audit's first
    # rule: look at the artefact) shows exactly why, and the two failures are in
    # opposite directions:
    #   * chicken wings — the grown hull hugs the FOOD and never reaches the
    #     plate rim, so the rim reads too small, the scale too coarse, and the
    #     mass triples (326 g -> 600 g, clipped at the clamp, against 275 g).
    #   * spaghetti — the hull spills past the plate and swallows the KITCHEN
    #     SCALE underneath it, the identical failure the audit recorded when
    #     morphological opening "rescued" this same photo.
    # Neither is a bad container-size prior: re-running both with every prior in
    # the table (27/20/18/13 cm) leaves the best case at 36.2% and 24.9% error,
    # so no size makes the grown rim right.
    # The code stays (with the evidence) because the *approach* — seed from the
    # segmenter, grow the surface — is still the right idea; it is the colour
    # homogeneity assumption that fails on a scalloped white paper plate. The
    # real fix remains §4(a): give the segmenter a container class and retrain.
    # Turn this on only with a weighed set that shows it helping.
    rim_surface_growth_enabled: bool = False
    # §3: derive a bowl's fill depth from the rim/surface area ratio (exact
    # hemispherical-cap geometry, no free constants) instead of the per-class
    # height prior. DEFAULT OFF and blocked, not abandoned: it needs a reliable
    # container rim, §4 failed to supply one, and the single weighed bowl in the
    # set has no rim at all — so there is currently no way to validate it. The
    # backlog asks for two or three more weighed bowls (one shallow, one full)
    # before this is switched on. Do not enable it to make the soup photo come
    # out right; n=1 cannot justify replacing a prior that works for full bowls.
    portion_derive_bowl_depth: bool = False
    card_reference_enabled: bool = True
    card_ratio_tol: float = 0.06
    card_min_long_px: float = 80.0
    # Food segmentation model (05 §6). Loaded from `model_dir` when present; absent
    # -> the estimator uses the classical GrabCut fallback, then a central box.
    segmenter_weights: str = "food_segmenter.pth"

    # ---- Recommendations (FR12, M7; see 06-recommendation-engine.md) ----
    min_meals_for_recommendation: int = 7
    recommendation_window_days: int = 7
    # Idempotency window (06 §3): a GET inside this many hours of the last
    # generation, with no meal logged since, returns the stored rows instead of
    # regenerating. Stops the insight text churning on every screen open, which
    # would read as instability rather than as freshness. POST
    # /recommendations/refresh bypasses it.
    recommendation_cache_hours: int = 6


# Single, importable instance — created once at process startup.
settings = Settings()


def get_settings() -> Settings:
    """Backwards-compatible accessor for the shared settings singleton."""
    return settings
