"""
app/schemas/auth.py
======================

WHY THIS FILE EXISTS
---------------------
FastAPI routes need a *contract* describing exactly what shape of JSON is
accepted in a request body and returned in a response — that contract is
a Pydantic model, not the SQLAlchemy ORM model. This file defines those
contracts for the two M2 auth endpoints: registration and login.

WHAT PROBLEM IT SOLVES
------------------------
The ORM's `User` model (in `db/models.py`) describes the *database* row —
it includes `password_hash`, `account_locked`, `failed_login_count`,
none of which should ever be accepted from or returned to a client
directly. Pydantic request/response schemas draw a hard line: a
`RegisterRequest` accepts exactly `email`, `username`, `password`,
`health_goal` (never a client-supplied `user_id` or `account_locked`
flag); a `UserResponse` returns exactly the fields safe to expose (never
`password_hash`).

HOW IT WORKS INTERNALLY
--------------------------
Pydantic v2 validates incoming JSON against each field's declared type
and constraints the instant FastAPI tries to construct the model — a
request missing `email` or sending `password` as a number never reaches
the route function at all; FastAPI auto-returns a 422 with a structured
error body describing exactly which field failed and why.
`RegisterRequest.password` uses `Field(min_length=8, ...)` to enforce
FR01's "password must be at least 8 characters" directly in the schema
layer — the earliest, cheapest place to reject bad input, before any
database work happens. `health_goal` is typed as the same `HealthGoal`
str-enum used by the ORM, so an invalid value like `"bulking"` is
rejected by Pydantic before it could ever reach the database's own ENUM
constraint.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Separate Pydantic schemas per direction (chosen) — `RegisterRequest`
  (input) vs `UserResponse` (output) vs the ORM `User` (storage): three
  distinct shapes for three distinct concerns. This is the standard
  FastAPI pattern and is what makes "never return password_hash" a
  structural guarantee rather than a habit to remember in every endpoint.
- Reusing the ORM model directly as the request/response type: tempting,
  but would either accidentally expose `password_hash` in responses, or
  require remembering to manually exclude fields in every single route —
  fragile.
- `model_config = ConfigDict(from_attributes=True)` on `UserResponse`
  (rather than manually mapping each field): lets FastAPI build the
  response schema directly from an ORM `User` object's attributes
  (`user.email`, `user.username`, ...) instead of hand-writing a
  dict-conversion function per endpoint.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why can't a client set their own user_id or account_locked value at
    registration?"
A: Because those fields simply don't exist on `RegisterRequest` — Pydantic
   silently ignores unexpected extra fields by default (or rejects them,
   depending on model config), so there's no code path where client input
   could ever set them.

Q: "Where exactly does the 'password must be ≥ 8 characters' rule live?"
A: In the Pydantic `Field(min_length=8)` constraint on
   `RegisterRequest.password` — enforced before the request even reaches
   `auth_service.register_user`, so weak passwords never touch business
   logic or the database.

Q: "Why is TokenResponse's field called access_token and not just
    'token'?"
A: Matches the OAuth2 Bearer token convention FastAPI's
   `OAuth2PasswordBearer` (used in `deps.py`) expects, and is what the
   auto-generated `/docs` "Authorize" flow looks for by name.

COMMON STUDENT MISTAKES
--------------------------
- Returning the ORM `User` object directly from a route without a
  response_model — FastAPI would try to serialize every ORM attribute,
  including password_hash, straight into the JSON response.
- Putting password strength rules in the route function instead of the
  schema — works, but scatters validation logic instead of keeping it in
  one declarative place next to the field it governs.
- Forgetting EmailStr requires the `email-validator` package — a common
  "why does my email field accept garbage" debugging session.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.db.models import HealthGoal


class RegisterRequest(BaseModel):
    """FR01 — what a new user must supply to create an account."""

    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=72)
    health_goal: HealthGoal

    @field_validator("password")
    @classmethod
    def password_has_letter_and_digit(cls, value: str) -> str:
        """FR01 / 03 §2: password must contain at least one letter and one digit."""
        if not any(c.isalpha() for c in value) or not any(c.isdigit() for c in value):
            raise ValueError("password must contain at least one letter and one digit")
        return value


class LoginRequest(BaseModel):
    """FR02 — credentials submitted to /auth/login."""

    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    """201 body for POST /auth/register (03 §2) — the created profile, no token."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    username: str
    email: EmailStr
    health_goal: HealthGoal
    created_at: datetime


class LoginUser(BaseModel):
    """The `user` object embedded in the login token response (03 §2)."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    username: str
    health_goal: HealthGoal


class TokenResponse(BaseModel):
    """200 body for POST /auth/login (03 §2) — bearer token + minimal user."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: LoginUser
