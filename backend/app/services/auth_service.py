"""
app/services/auth_service.py
===============================

WHY THIS FILE EXISTS
---------------------
This is where FR01 (registration) and FR02 (authentication + lockout)
actually live. The API route functions in `api/auth.py` stay thin — they
parse the request, call one function here, and translate the result into
an HTTP response. All the *business logic* (uniqueness checks, password
hashing, failed-login counting, lockout) is concentrated in this one
file so it can be unit-tested directly, without spinning up HTTP.

WHAT PROBLEM IT SOLVES
------------------------
Auth logic is exactly the kind of code that's easy to get subtly wrong —
off-by-one in the failed-login counter, forgetting to reset it on
success, or checking `account_locked` in the wrong order relative to
password verification. Isolating it in a service module (rather than
inline in the route) means the whole flow can be tested with plain
Python function calls and a test database session, and reused
identically from both the HTTP route and, if needed later, an admin CLI.

HOW IT WORKS INTERNALLY
--------------------------
`register_user`: checks for an existing row with the same email first
(cheap, avoids relying solely on the DB's unique constraint + catching
`IntegrityError`, though the DB constraint is still the ultimate
guarantee against a race condition); if none is found, hashes the
password and inserts a new `User` row with `account_locked=False`,
`failed_login_count=0`.

`authenticate_user`: looks up the user by email. If the account is
already locked, raises `AccountLockedError` immediately — no point
checking the password of a locked account. Otherwise verifies the
password with `verify_password`. On failure: increments
`failed_login_count`; if that increment reaches
`settings.max_failed_logins` (5), flips `account_locked=True` in the same
transaction, then raises `InvalidCredentialsError` regardless (a locked
account and a merely-wrong password both look identical to the caller on
the failing attempt itself — the *next* attempt is what will surface the
423). On success: resets `failed_login_count` back to 0 (a successful
login should not carry forward a partial failure streak from days ago)
and returns the user.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Checking `account_locked` before verifying the password (chosen):
  avoids doing an expensive bcrypt comparison for an account that's going
  to be rejected regardless, and gives a precise 423 instead of a
  misleading "wrong password" on an already-locked account.
- Resetting `failed_login_count` to 0 on any successful login (chosen):
  matches FR02's "account locked after **5 consecutive** failed
  attempts" — a single success should clear the streak, not let failures
  silently accumulate across unrelated login sessions.
- Locking on the failed attempt that *reaches* 5, not checking `>= 5`
  before incrementing: an off-by-one here is a classic bug — this
  implementation increments first, then compares, so exactly the 5th
  consecutive bad attempt is the one that locks the account (matching the
  literal requirement "after 5 failed login attempts").

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "What happens on the request that locks the account — does it get a
    different error than a normal wrong password?"
A: No, intentionally — both a locked account and a wrong password return
   401/`InvalidCredentialsError` on that same request, so an attacker
   probing passwords can't distinguish "that guess was wrong" from
   "that guess happened to be the one that triggered the lock." The
   *next* attempt against that account is what returns 423.

Q: "Why not lock permanently, forever, with no unlock path?"
A: The current scope (FR02) only specifies lockout after 5 attempts; the
   PRD does not define an unlock flow (e.g. email reset) — that's an
   explicitly acknowledged gap suitable to raise as a "future work" item
   in the defense (an admin unlock endpoint or a timed auto-unlock after
   N minutes would be the natural next iteration).

Q: "Why look up by email before trying to catch a DB IntegrityError for
    duplicates?"
A: It's a friendlier, faster failure path for the overwhelmingly common
   case (no race condition) and lets us raise a precise, typed
   `DuplicateEmailError` immediately. The database's own UNIQUE
   constraint on `email` remains the real guarantee against the rare
   race where two requests check simultaneously and both see "no
   existing row" before either commits.

COMMON STUDENT MISTAKES
--------------------------
- Verifying the password before checking `account_locked` — leaks timing
  information and does needless bcrypt work on already-locked accounts.
- Forgetting to reset `failed_login_count` on a successful login — a user
  who once mistyped their password 4 times, left, and came back a week
  later would then get locked out on their very next single wrong guess.
- Locking the account but still returning "email or password incorrect"
  as if that's the final state, without a *separate*, distinguishable
  423 on the *following* attempt.
"""

from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import (
    AccountLockedError,
    DuplicateEmailError,
    DuplicateUsernameError,
    InvalidCredentialsError,
)
from app.db.models import User
from app.schemas.auth import LoginRequest, RegisterRequest
from app.security import hash_password, verify_password


def register_user(db: Session, payload: RegisterRequest) -> User:
    # Pre-checks give a precise 409 for the common case; the DB UNIQUE
    # constraints on email/username remain the ultimate guard against races.
    if db.query(User).filter(User.email == payload.email).first() is not None:
        raise DuplicateEmailError()
    if db.query(User).filter(User.username == payload.username).first() is not None:
        raise DuplicateUsernameError()

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        health_goal=payload.health_goal,
        # Optional body metrics (M7): absent -> NULL, and the recommendation
        # engine falls back to a descriptive calorie note (06 §2).
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        birth_year=payload.birth_year,
        sex=payload.sex,
        activity_level=payload.activity_level,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, payload: LoginRequest) -> User:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None:
        # Same error as "wrong password" — never reveal whether the
        # email itself exists in the system.
        raise InvalidCredentialsError()

    if user.account_locked:
        raise AccountLockedError()

    if not verify_password(payload.password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= settings.max_failed_logins:
            user.account_locked = True
        db.commit()
        remaining = max(settings.max_failed_logins - user.failed_login_count, 0)
        raise InvalidCredentialsError(details={"remaining_attempts": remaining})

    # Successful login: clear any prior failure streak.
    if user.failed_login_count != 0:
        user.failed_login_count = 0
        db.commit()

    return user
