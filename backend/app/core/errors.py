"""
app/core/errors.py
=====================

WHY THIS FILE EXISTS
---------------------
FastAPI's default error responses are fine, but the PRD/build-plan calls
for a consistent "error envelope" — every error the API returns, no
matter which endpoint or which failure, should have the same predictable
JSON shape so the mobile app can handle all errors with one piece of
client code. This file defines that shared shape and a couple of small
custom exception types for auth-specific failures (duplicate email,
account locked) that don't map cleanly onto FastAPI's built-in
`HTTPException` semantics alone.

WHAT PROBLEM IT SOLVES
------------------------
Without a shared envelope, one endpoint might return
`{"detail": "..."}`, another `{"error": "...", "code": 5}`, and the
mobile client would need bespoke parsing per endpoint. Standardising on
`{"error": {"code": ..., "message": ...}}` means the app's single HTTP
error interceptor (see `07-mobile-app.md`) can extract a user-facing
message from any failed response identically.

HOW IT WORKS INTERNALLY
--------------------------
`AppError` is a small custom exception carrying an HTTP status code, a
short machine-readable `code` string (e.g. `"EMAIL_ALREADY_REGISTERED"`),
and a human-readable `message`. Route/service code raises `AppError`
instances directly (e.g. `raise DuplicateEmailError()`). FastAPI's
generic mechanism for turning exceptions into HTTP responses is an
*exception handler*, registered once against the whole app in
`main.py`/`create_app()` via
`app.add_exception_handler(AppError, app_error_handler)` — that handler
reads the exception's `status_code`/`code`/`message` and serialises them
into the shared envelope shape, so no individual route needs to build
that JSON by hand.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- A small hierarchy of typed `AppError` subclasses (chosen): each failure
  mode (duplicate email, account locked) is a distinct, named Python
  exception that can be raised with a one-liner
  (`raise DuplicateEmailError()`) and unit-tested by asserting the right
  exception type/status is raised, independent of the exact wording.
- Raising bare `HTTPException(status_code=409, detail="...")` inline
  everywhere: works, but scatters magic status codes and message strings
  across services instead of naming the failure mode once.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why 409 for duplicate email and not 400?"
A: 409 Conflict specifically means "the request conflicts with the
   current state of the resource" — a perfect fit for "this email is
   already registered," distinct from 400 Bad Request's more generic
   "the request itself was malformed."

Q: "Why 423 for a locked account instead of 401?"
A: 423 Locked is the standard HTTP status for "the resource that is being
   accessed is locked" — it communicates something more specific than
   401 (invalid/missing credentials): the credentials might even be
   correct, but the account itself is temporarily unusable.

COMMON STUDENT MISTAKES
--------------------------
- Returning 500 for a duplicate email (an unhandled IntegrityError from
  the DB's unique constraint bubbling up) instead of catching it and
  raising the specific, intentional 409 — a graduation defense will often
  probe "what if two requests race to register the same email at the
  exact same time?"
- Inconsistent error shapes between endpoints, forcing per-endpoint
  parsing logic on the mobile client.
"""

from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    """Base class for all intentional, user-facing application errors.

    `code` values are the canonical machine-readable strings from
    `03-api-specification.md §9` — the mobile client maps on `error.code`,
    so they must not drift from that list.
    """

    status_code: int = 400
    code: str = "APP_ERROR"
    message: str = "An error occurred."

    def __init__(self, message: str | None = None, details: dict[str, Any] | None = None):
        self.message = message or self.message
        self.details = details
        super().__init__(self.message)


class DuplicateEmailError(AppError):
    status_code = 409
    code = "DUPLICATE_EMAIL"
    message = "An account with this email already exists."


class DuplicateUsernameError(AppError):
    status_code = 409
    code = "DUPLICATE_USERNAME"
    message = "This username is already taken."


class InvalidCredentialsError(AppError):
    status_code = 401
    code = "INVALID_CREDENTIALS"
    message = "Incorrect email or password."


class AccountLockedError(AppError):
    status_code = 423
    code = "ACCOUNT_LOCKED"
    message = "Account locked after too many failed login attempts."


class UnauthorizedError(AppError):
    status_code = 401
    code = "UNAUTHORIZED"
    message = "Missing or invalid authentication credentials."


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"
    message = "The requested resource does not exist."


class ForbiddenError(AppError):
    status_code = 403
    code = "FORBIDDEN"
    message = "You do not have access to this resource."


# ---- Image upload validation (FR03, raised by app/ml/preprocess.py in M3) ----
# These three map onto the distinct 4xx codes the /recognise contract promises
# (03 §4, §9) so the mobile client can tell "too big" from "wrong format" from
# "too small" and show the right message, instead of one generic "bad image".


class ImageTooLargeError(AppError):
    status_code = 413
    code = "IMAGE_TOO_LARGE"
    message = "Image exceeds the upload size limit."


class ImageBadFormatError(AppError):
    status_code = 400
    code = "IMAGE_BAD_FORMAT"
    message = "Image must be a valid JPEG or PNG."


class ImageTooSmallError(AppError):
    status_code = 400
    code = "IMAGE_TOO_SMALL"
    message = "Image resolution is below the minimum required."


class ImagePoorLightingError(AppError):
    status_code = 400
    code = "IMAGE_POOR_LIGHTING"
    message = "Image is too dark or too bright to recognise. Retake with even lighting."


class ImageTooBlurryError(AppError):
    status_code = 400
    code = "IMAGE_TOO_BLURRY"
    message = "Image is too blurry to recognise. Hold steady and retake."


class InferenceFailedError(AppError):
    status_code = 500
    code = "INFERENCE_FAILED"
    message = "The model failed to process this image."


def _envelope(code: str, message: str, details: Any | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn any AppError (raised anywhere in a route/service) into the shared
    error envelope shape defined in `03 §9`."""
    assert isinstance(exc, AppError)
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message, exc.details),
    )


# Map bare HTTP status codes (raised by FastAPI/Starlette internals, e.g.
# OAuth2PasswordBearer's missing-token 401, or an unmatched-route 404) onto the
# canonical `03 §9` code strings so *every* non-2xx uses the shared envelope.
_STATUS_TO_CODE: dict[int, str] = {
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    413: "IMAGE_TOO_LARGE",
}


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Reshape any bare `HTTPException` into the shared error envelope (`03 §9`).

    FastAPI/Starlette raise plain `HTTPException`s in places we don't control
    (a missing bearer token via `OAuth2PasswordBearer`, an unmatched route, a
    wrong method). Their default body is `{"detail": ...}`, which would break the
    "one error shape everywhere" contract the mobile client relies on, so we
    normalise them here."""
    assert isinstance(exc, StarletteHTTPException)
    fallback = "INTERNAL_ERROR" if exc.status_code >= 500 else "HTTP_ERROR"
    code = _STATUS_TO_CODE.get(exc.status_code, fallback)
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(status_code=exc.status_code, content=_envelope(code, message))


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Reshape FastAPI's default 422 body into the shared error envelope so
    *every* non-2xx response has the same shape (`03 §9`)."""
    assert isinstance(exc, RequestValidationError)
    # Keep only JSON-safe fields — a custom validator's ValueError lands in
    # each error's `ctx` and is not JSON-serialisable.
    details = [
        {"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_envelope(
            "VALIDATION_ERROR",
            "One or more fields are invalid.",
            details=details,
        ),
    )
