"""
app/api/auth.py
==================

WHY THIS FILE EXISTS
---------------------
This is the HTTP surface for FR01 (registration) and FR02 (login) — the
two endpoints the mobile app's Login/Register screens actually call.
It is deliberately thin: parse input (FastAPI + the Pydantic schemas do
this automatically), delegate to `auth_service`, and shape the response.

WHAT PROBLEM IT SOLVES
------------------------
Without a dedicated router, every feature's routes would be jammed into
`main.py`. `APIRouter` lets this file own everything under the `/auth`
prefix (mounted in `main.py`) while staying focused on exactly two
responsibilities: registering an account and issuing a token on login.

HOW IT WORKS INTERNALLY
--------------------------
Both routes take a Pydantic request schema as a parameter — FastAPI
automatically parses+validates the JSON body against it *before* the
function body runs (a malformed body never reaches this code; see
`schemas/auth.py`). Both routes also take `db: Session = Depends(get_db)`
— FastAPI resolves that dependency (opens a session, will close it after
the response is sent) and passes it in. Each route calls exactly one
`auth_service` function, then wraps the returned ORM `User` in a
`TokenResponse` (which itself embeds a `UserResponse`, built via Pydantic
v2's `from_attributes=True`, straight from the ORM object's attributes).
Any failure (`DuplicateEmailError`, `InvalidCredentialsError`,
`AccountLockedError`) is raised as an exception by the service layer and
converted into the shared error envelope by the `app_error_handler`
registered once in `main.py` — these routes never build error JSON by
hand.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Business logic entirely in `auth_service`, routes only orchestrate
  (chosen): keeps this file readable as a map of "what does /auth expose"
  and keeps the logic testable without an HTTP client.
- Register returns the created profile (no token), then the client calls
  /auth/login for a JWT: this matches the authoritative API contract
  (`03-api-specification.md §2`), which the mobile TypeScript types mirror
  field-for-field. Keeping register and login response shapes exactly as
  `03` specifies avoids client/server drift.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why doesn't /auth/register return a token, so the user must log in
    right after?"
A: The API contract (`03 §2`) defines register's 201 response as the
   created profile only; login is the single, dedicated token-issuing
   endpoint. Keeping one clear place that issues JWTs (and matching the
   documented contract the mobile client is built against) is worth the
   one extra call.

Q: "What status code does a successful registration return, and why?"
A: 201 Created — the semantically correct code for "a new resource (the
   user account) was created," distinct from 200 OK's more generic
   success meaning.

Q: "Where exactly does '5 failed attempts locks the account' get
    enforced — here, or in auth_service?"
A: Entirely in `auth_service.authenticate_user` — this route file has no
   knowledge of the lockout threshold at all, which is exactly the point
   of keeping business rules out of the HTTP layer.

COMMON STUDENT MISTAKES
--------------------------
- Returning 200 instead of 201 for registration.
- Putting the duplicate-email check or password hashing directly inside
  the route function — makes the logic untestable without spinning up a
  FastAPI TestClient, and duplicates logic if a second entry point
  (e.g. an admin script) ever needs to create users too.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.config import settings
from app.deps import get_db
from app.schemas.auth import (
    LoginRequest,
    LoginUser,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from app.security import create_access_token
from app.services.auth_service import authenticate_user, register_user

router = APIRouter()


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    # 03 §2: register returns the created profile (no token); the client then
    # calls /auth/login to obtain a JWT.
    user = register_user(db, payload)
    return RegisterResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate_user(db, payload)
    token = create_access_token(subject=user.user_id)
    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        user=LoginUser.model_validate(user),
    )
