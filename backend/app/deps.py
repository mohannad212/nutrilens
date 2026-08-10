"""
app/deps.py
=============

WHY THIS FILE EXISTS
---------------------
FastAPI's "Dependency Injection" system needs small, reusable functions it
can call automatically before your route runs, and hand you the result as
an argument. This file defines the two dependencies almost every protected
route needs: a database session (`get_db`) and the currently authenticated
user (`get_current_user`). Instead of every route manually opening a
session or manually decoding the Authorization header, they just declare
`db: Session = Depends(get_db)` and FastAPI wires it up.

WHAT PROBLEM IT SOLVES
------------------------
Without this, every single endpoint would repeat: open a session, remember
to close it even on error, read the Authorization header, strip "Bearer ",
decode the JWT, handle a missing/invalid/expired token, look the user up
in the DB, handle "user not found" or "account locked". That's a lot of
copy-pasted (and easy-to-get-wrong) logic per endpoint. Centralising it
here means the security-critical "who is making this request" logic is
written and reviewed exactly once.

HOW IT WORKS INTERNALLY
--------------------------
`get_db` is a *generator* dependency: FastAPI calls it, receives the first
value from `yield`, runs your route handler with it, and — critically —
resumes the generator afterward (running the `finally: db.close()`)
regardless of whether the route succeeded or raised an exception. This
guarantees every request's session is always returned to the connection
pool, even on errors.

`get_current_user` uses FastAPI's `OAuth2PasswordBearer`, which knows how
to (a) read the `Authorization: Bearer <token>` header, (b) advertise the
`/auth/login` endpoint in the auto-generated OpenAPI docs so `/docs`
renders a working "Authorize" button, and (c) raise a 401 automatically if
the header is missing entirely. We then call `decode_access_token` (from
`security.py`); if that returns `None` (invalid/expired) or the user id in
the token no longer matches a real row in the database, we raise 401. This
function is itself declared as a dependency (`Depends(get_current_user)`),
so any route can require an authenticated user just by adding one
parameter — FastAPI resolves the whole dependency chain
(`get_current_user` → `get_db`) automatically.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Generator-based `get_db` with try/finally (chosen): guarantees cleanup
  even on exceptions — the standard, recommended FastAPI+SQLAlchemy
  pattern.
- Opening a session per-call inside each service function instead: leads
  to multiple, uncoordinated sessions/transactions within a single
  request if more than one service is called — harder to reason about
  atomicity.
- Middleware-based auth (checking the header globally before routing):
  works, but loses FastAPI's per-route dependency granularity (some
  routes, like `/health` or `/auth/register`, must stay public) and loses
  the automatic OpenAPI "Authorize" integration that `OAuth2PasswordBearer`
  provides for free.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "What HTTP status does an expired token produce, and why?"
A: 401 Unauthorized — the request has no (valid) credentials at all,
   which is distinct from 403 Forbidden (valid credentials, but not
   allowed to do this specific thing).

Q: "Why re-check the database in get_current_user instead of trusting
    the JWT payload entirely?"
A: The JWT proves the token was validly issued and hasn't expired, but it
   can't reflect account state that changed *after* issuance — e.g. the
   account was locked or the user was deleted 10 minutes into a 60-minute
   token's lifetime. The DB check catches that.

Q: "Why is get_db a generator instead of just returning a session?"
A: Only a generator lets FastAPI run cleanup code (`db.close()`) after the
   route finishes — a plain `return` gives no hook to run code once the
   request is done.

COMMON STUDENT MISTAKES
--------------------------
- Forgetting `db.close()` (or forgetting the try/finally) — under load
  this exhausts the connection pool and the app starts hanging.
- Trusting `user_id` from the JWT without confirming the user still
  exists / isn't locked — a deleted or locked account's old token would
  otherwise keep working until it naturally expires.
- Hard-coding `tokenUrl="/auth/login"` inconsistently with the actual
  route path — this string is only used for `/docs` UI, but a mismatch
  is a common source of "why doesn't the Authorize button work" confusion.
"""

from collections.abc import Iterator

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import AccountLockedError, InferenceFailedError, UnauthorizedError
from app.db.base import SessionLocal
from app.db.models import User
from app.integrations.usda_client import UsdaClient
from app.ml.classifier import FoodClassifier
from app.ml.food_gate import FoodGate
from app.ml.portion import PortionEstimator
from app.security import decode_access_token

# Tells FastAPI's OpenAPI docs where to send the username/password to get a
# token. It does NOT enforce anything on its own — it just knows how to
# extract "Bearer <token>" from the Authorization header. Must match the real
# mounted path (03 §Base URL: everything lives under /api/v1).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_db() -> Iterator[Session]:
    """Yield one DB session per request; always close it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolve the JWT in the Authorization header to a live User row.
    Raises 401 if the token is missing/invalid/expired, or if the account
    no longer exists; 423 if the account has since been locked. Both go
    through the shared error envelope (03 §9).
    """
    payload = decode_access_token(token)
    if payload is None or "sub" not in payload:
        raise UnauthorizedError()

    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise UnauthorizedError()
    if user.account_locked:
        raise AccountLockedError()

    return user


def get_classifier(request: Request) -> FoodClassifier:
    """Hand the request the one `FoodClassifier` loaded at startup.

    The model is loaded once by the app lifespan (`main.py`) and stashed on
    `app.state.classifier`. If it failed to load (e.g. the weights file is
    absent in an environment that only tracks it via a GitHub Release), that
    attribute is `None` and any `/recognise` call fails cleanly with a 500
    `INFERENCE_FAILED` envelope rather than an `AttributeError`. Tests override
    this dependency with a stub, so they never need the real 69 MB weights.
    """
    classifier: FoodClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise InferenceFailedError("The recognition model is not available.")
    return classifier


def get_usda_client(request: Request) -> UsdaClient:
    """Hand the request the one `UsdaClient` built at startup.

    Built once by the lifespan (`main.py`) so its `httpx.Client` connection pool
    is shared across requests rather than re-established per lookup. Unlike the
    classifier, construction cannot fail — there is no artefact to load — so this
    never raises; a *missing API key* surfaces later as the documented 502 from
    the client itself, which is honest about what actually went wrong.

    Injecting it as a dependency is also what makes M5 testable without a network:
    tests override this with a stub client (`08 §9`).
    """
    client: UsdaClient | None = getattr(request.app.state, "usda_client", None)
    if client is None:  # pragma: no cover - only reachable if the lifespan is skipped
        client = UsdaClient(
            settings.usda_api_key,
            settings.usda_base_url,
            timeout=settings.usda_timeout_seconds,
        )
        request.app.state.usda_client = client
    return client


def get_food_gate(request: Request) -> FoodGate | None:
    """Hand the request the open-set `FoodGate`, or None if it isn't loaded.

    None is a supported state, not an error: the gate is a guard in front of
    recognition, so an absent one means "don't guard", never "fail". Refusing
    every upload because an auxiliary model is missing would be a worse outcome
    than the non-food guesses the gate exists to prevent.
    """
    return getattr(request.app.state, "food_gate", None)


def get_portion_estimator(request: Request) -> PortionEstimator | None:
    """Hand the request the `PortionEstimator` built at startup, or None.

    Unlike the classifier, a missing portion estimator is *not* a 500: portion
    estimation is an optional enrichment of `/recognise` (FR05), so if it is
    disabled or was not constructed we simply return None and the service omits
    the `portion` object. The estimator itself never fails — with no trained aux
    models wired it degrades to a `shape_only` range (`05 §10`).
    """
    return getattr(request.app.state, "portion_estimator", None)
