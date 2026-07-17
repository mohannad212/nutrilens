"""
app/security.py
==================

WHY THIS FILE EXISTS
---------------------
Two of the most security-sensitive operations in the whole app live here,
and nowhere else: turning a plaintext password into something safe to
store (and back-checking it at login), and issuing/verifying the JWT that
proves "this request came from an authenticated user". Concentrating both
in one small, well-tested file means there is exactly one place in the
codebase that touches cryptographic primitives.

WHAT PROBLEM IT SOLVES
------------------------
FR01 requires passwords never be stored in a recoverable form. FR02
requires a stateless way to prove identity on every subsequent request
without re-sending the password each time. `security.py` solves both:
`hash_password`/`verify_password` solve the first, `create_access_token`/
`decode_access_token` solve the second.

HOW IT WORKS INTERNALLY
--------------------------
Password hashing: `passlib`'s `CryptContext` wraps bcrypt. Bcrypt
generates a random *salt* per call and embeds it inside the output hash
string itself (`$2b$12$<22-char-salt><31-char-hash>`), so two users with
the same password get completely different stored hashes, and no separate
salt column is ever needed. `verify_password` re-derives the hash using
the salt embedded in the stored value and compares in constant time.

JWT: `create_access_token` builds a payload dict — `sub` (subject, the
user id) and `exp` (expiry, computed as now + ACCESS_TOKEN_EXPIRE_MINUTES)
— and calls `jose.jwt.encode(payload, secret_key, algorithm="HS256")`.
HS256 is a *symmetric* signature: the same `SECRET_KEY` both signs the
token when issued and verifies it when decoded. The token itself is NOT
encrypted, only signed — anyone can read its payload (it's just
base64-encoded JSON), but nobody without `SECRET_KEY` can forge a valid
signature or tamper with the payload undetected. `decode_access_token`
calls `jose.jwt.decode(...)`, which automatically re-verifies the
signature and raises if it's invalid or expired.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- bcrypt (chosen) over plain SHA-256/MD5: general-purpose hash functions
  are FAST — great for hashing, terrible for passwords, because an
  attacker with a stolen hash database can brute-force billions of
  guesses per second. bcrypt is deliberately *slow* (tunable "cost
  factor") and includes salting by design, so brute-forcing even a
  single password becomes computationally expensive.
- bcrypt over Argon2 (the newer OWASP-recommended default): Argon2 is
  arguably stronger, but bcrypt is more universally supported, well
  understood, has no known practical breaks, and is what the report /
  PRD explicitly specifies — a defensible, still-secure choice for a
  graduation project.
- JWT + HS256 (chosen) over server-side session storage (session id in a
  DB/Redis table): JWT is stateless — the backend doesn't need to look up
  a session on every request, which simplifies horizontal scaling (any
  server instance can verify any token with just the shared secret). The
  tradeoff is that a JWT can't be instantly revoked before it expires
  (mitigated here by a short 60-minute expiry).
- HS256 (symmetric) over RS256 (asymmetric): RS256 is preferred when
  multiple *different* services need to verify tokens without being able
  to issue them (they'd hold only the public key). NutriLens has one
  backend that both issues and verifies tokens, so the simpler symmetric
  HS256 is sufficient and avoids managing a key pair.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "If someone steals the database, can they get user passwords back?"
A: No — bcrypt is a one-way function. They get the hash, not the
   password, and reversing a bcrypt hash by brute force is
   computationally infeasible for a well-chosen cost factor.

Q: "If someone steals a JWT, what can they do?"
A: Act as that user until the token expires (max 60 minutes here) or the
   account is otherwise locked. This is why tokens are sent only over
   HTTPS and stored in the mobile app's secure storage, not plain
   AsyncStorage.

Q: "Why include `exp` in the token instead of checking a DB timestamp?"
A: That's the entire point of a *stateless* token — expiry is verified
   locally from the signed payload, with zero database round-trip.

Q: "What happens if SECRET_KEY leaks?"
A: An attacker could forge valid tokens for any user. This is why it must
   be a long random value from the environment, never committed to git,
   and rotated if ever exposed.

COMMON STUDENT MISTAKES
--------------------------
- Storing passwords with `hashlib.sha256(password).hexdigest()` — no
  salt, fast to brute-force, a classic security-review red flag.
- Putting sensitive data (email, full name) inside the JWT payload —
  remember the payload is *readable*, just not *forgeable*. This project
  only stores the user id (`sub`) in the token.
- Comparing raw hashes with `==` instead of `verify_password` — passlib's
  verify does a safe comparison; string `==` on secrets can leak timing
  information in theory.
- Forgetting to catch `JWTError` when decoding — an expired or tampered
  token must fail closed (treated as unauthenticated), never crash the
  server with an unhandled exception.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# A CryptContext can support multiple hash schemes and migrate between them;
# we use only bcrypt, but this API leaves room to add/rotate schemes later
# without touching call sites.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage. Never store `plain_password`."""
    return str(pwd_context.hash(plain_password))


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Check a login attempt's password against the stored bcrypt hash."""
    return bool(pwd_context.verify(plain_password, password_hash))


def create_access_token(subject: str | int) -> str:
    """
    Issue a signed JWT for `subject` (the user_id).
    Expiry is controlled centrally by settings.access_token_expire_minutes.
    """
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": str(subject), "exp": expire}
    return str(jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm))


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Verify signature + expiry and return the payload, or None if the token
    is invalid/expired/tampered. Callers (deps.get_current_user) treat
    `None` as "unauthenticated".
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token, settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None
