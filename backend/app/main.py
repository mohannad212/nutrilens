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

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api import auth as auth_router
from app.core.errors import (
    AppError,
    app_error_handler,
    validation_error_handler,
)

# Every API route lives under this prefix (see 03-api-specification.md §Base URL).
API_V1_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="NutriLens API",
        version="0.1.0",
        description="AI Food Recognition System — backend API",
    )

    # /health is intentionally un-prefixed — it's the load-balancer probe (03 §8).
    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth_router.router, prefix=f"{API_V1_PREFIX}/auth", tags=["auth"])

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    return app


app = create_app()
