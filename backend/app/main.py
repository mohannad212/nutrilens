"""
app/main.py
=============

WHY THIS FILE EXISTS
---------------------
Every FastAPI project needs one entry point that creates the `app` object,
mounts every router, and wires up startup/shutdown behaviour. This is that
file — the one Uvicorn actually imports and runs
(`uvicorn app.main:app --reload`).

WHAT PROBLEM IT SOLVES
------------------------
Without a single factory function, route registration tends to sprawl
across import side-effects that are hard to trace. `create_app()` makes
app construction an explicit, testable function: tests can call it fresh
without needing a running server, and it's obvious at a glance which
routers are mounted.

HOW IT WORKS INTERNALLY
--------------------------
`create_app()` builds a `FastAPI` instance and calls `app.include_router`
for each feature router (auth now in M2; recognise/nutrition/meals/
recommendations arrive in later milestones). `/health` is a trivial route
used by `docker-compose`'s healthcheck and by CI to confirm the container
actually started. The module-level `app = create_app()` is what Uvicorn
imports.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- App factory function (chosen): testable in isolation, and later
  supports creating multiple app instances with different settings (e.g.
  a test app pointed at a test database) without import-order headaches.
- A bare module-level `app = FastAPI()` with routes registered via
  decorators directly in this file: works for a toy project, but doesn't
  scale past a handful of endpoints and makes it hard to have a clean
  "app under test" vs "app under uvicorn" distinction.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why does /health exist and what does it check?"
A: It's a liveness probe — confirms the process is up and able to answer
   HTTP requests at all. It deliberately does NOT check the database here
   in M0/M1, to keep it a fast, dependency-free signal; DB connectivity is
   exercised instead by the M1 seed scripts and tests.

Q: "Why `include_router` and not just defining routes directly on `app`?"
A: `APIRouter` lets each feature (auth, meals, recognise, ...) own its own
   file with its own prefix/tags, then get mounted onto the main app —
   keeps `main.py` a thin composition point instead of a growing pile of
   route functions.

COMMON STUDENT MISTAKES
--------------------------
- Defining all routes directly in `main.py` — becomes unmaintainable
  once you have 5+ feature areas.
- Forgetting to prefix routers (e.g. everything ends up unprefixed and
  route paths collide).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import auth as auth_router
from app.api import meals as meals_router
from app.api import meta as meta_router
from app.api import nutrition as nutrition_router
from app.api import recognise as recognise_router
from app.api import recommendations as recommendations_router
from app.api import users as users_router
from app.config import settings
from app.core.errors import (
    AppError,
    app_error_handler,
    http_exception_handler,
    validation_error_handler,
)
from app.core.rate_limit import HourlyRateLimiter
from app.integrations.usda_client import UsdaClient
from app.ml.classifier import load_classifier
from app.ml.portion import PortionEstimator
from app.ml.portion.container_model import load_container_model
from app.ml.portion.reference_detector import load_coco_reference_detector
from app.ml.portion.segmenter import load_food_segmenter

logger = logging.getLogger(__name__)

# Every API route lives under this prefix (see 03-api-specification.md §Base URL).
API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the recognition model once, at startup (04 §8).

    The 69 MB weights are versioned via a GitHub Release, not git (04 §7), so an
    environment may legitimately start without the file present (e.g. CI). Rather
    than crash the whole API, we log a warning and leave `app.state.classifier =
    None`; `/recognise` then returns a clean 500 INFERENCE_FAILED (via
    `get_classifier`) while every other endpoint keeps working.
    """
    weights = Path(settings.model_dir) / settings.classifier_weights
    labels = Path(settings.model_dir) / settings.classifier_labels
    try:
        app.state.classifier = load_classifier(str(weights), str(labels), settings.device)
        logger.info("Recognition model loaded from %s (device=%s)", weights, settings.device)
    except Exception:  # missing/corrupt weights — degrade, don't crash the API
        app.state.classifier = None
        logger.warning("Recognition model NOT loaded (%s); /recognise will 500", weights)

    # M4 portion estimator (05). The trained aux models (container classifier,
    # reference detector, segmenter) arrive from the offline ML track on their own
    # schedule; until then the estimator runs with none of them wired and returns
    # a valid `shape_only` range (05 §10). Constructing it never fails, but we
    # still guard so a bad build degrades to "no portion" rather than crashing.
    if settings.portion_enabled:
        # Stage 1 container model is optional: load it if the trained weights are
        # present, else leave it None so portion degrades to shape_only (05 §10).
        container_model = None
        cw = Path(settings.model_dir) / settings.container_weights
        cl = Path(settings.model_dir) / settings.container_labels
        try:
            if cw.exists() and cl.exists():
                container_model = load_container_model(str(cw), str(cl), settings.device)
                logger.info("Container model loaded from %s", cw)
            else:
                logger.info("Container model not present (%s); portion uses shape_only", cw)
        except Exception:
            container_model = None
            logger.warning("Container model failed to load (%s); portion uses shape_only", cw)

        # Stage 2 reference detector is opt-in (downloads a COCO model). When off
        # or unavailable, portion still works via container/shape-only (05 §10).
        reference_detector = None
        if settings.reference_detector_enabled:
            try:
                reference_detector = load_coco_reference_detector(
                    settings.device, settings.reference_score_threshold
                )
                logger.info("COCO reference detector loaded (fork/spoon/knife)")
            except Exception:
                reference_detector = None
                logger.warning("Reference detector failed to load; portion skips Stage 2")

        # Food segmenter (05 §6): trained model if present, else classical fallback.
        segmenter = None
        sw = Path(settings.model_dir) / settings.segmenter_weights
        try:
            if sw.exists():
                segmenter = load_food_segmenter(str(sw), settings.device)
                logger.info("Food segmenter loaded from %s", sw)
            else:
                logger.info("Food segmenter not present (%s); using classical fallback", sw)
        except Exception:
            segmenter = None
            logger.warning("Food segmenter failed to load (%s); classical fallback", sw)

        try:
            app.state.portion_estimator = PortionEstimator(
                container_model=container_model,
                reference_detector=reference_detector,
                segmenter=segmenter,
            )
            logger.info(
                "Portion estimator ready (container=%s, reference=%s, segmenter=%s)",
                container_model is not None,
                reference_detector is not None,
                segmenter is not None,
            )
        except Exception:
            app.state.portion_estimator = None
            logger.warning("Portion estimator NOT built; /recognise omits portion")
    else:
        app.state.portion_estimator = None
        logger.info("Portion estimation disabled (portion_enabled=false)")

    # M5 USDA client (08 §5-6). Built once so its httpx connection pool and its
    # hourly budget are shared process-wide rather than per request. Construction
    # touches no network, so unlike the models there is nothing to fail here — a
    # missing API key becomes an honest 502 at call time (and is warned about now,
    # because "why is every nutrition lookup failing" is a question worth answering
    # in the startup log rather than the access log).
    if not settings.usda_api_key:
        logger.warning("USDA_API_KEY is not set; nutrition lookups will fail with 502")
    app.state.usda_client = UsdaClient(
        settings.usda_api_key,
        settings.usda_base_url,
        timeout=settings.usda_timeout_seconds,
        rate_limiter=HourlyRateLimiter(settings.usda_rate_limit_per_hour),
    )

    yield

    # Release the USDA connection pool. The models are read-only and need no
    # teardown, but an httpx.Client holds real sockets.
    app.state.usda_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NutriLens API",
        version="0.1.0",
        description="AI Food Recognition System — backend API",
        lifespan=lifespan,
    )

    # /health is intentionally un-prefixed — it's the load-balancer probe (03 §8).
    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth_router.router, prefix=f"{API_V1_PREFIX}/auth", tags=["auth"])
    app.include_router(recognise_router.router, prefix=API_V1_PREFIX, tags=["recognition"])
    app.include_router(nutrition_router.router, prefix=API_V1_PREFIX, tags=["nutrition"])
    app.include_router(meals_router.router, prefix=API_V1_PREFIX, tags=["meals"])
    app.include_router(users_router.router, prefix=API_V1_PREFIX, tags=["users"])
    app.include_router(meta_router.router, prefix=API_V1_PREFIX, tags=["meta"])
    app.include_router(
        recommendations_router.router, prefix=API_V1_PREFIX, tags=["recommendations"]
    )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    return app


app = create_app()
