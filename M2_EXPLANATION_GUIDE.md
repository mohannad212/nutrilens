# M2_EXPLANATION_GUIDE.md — Authentication (FR01, FR02)

This guide explains **M2**: turning the M1 plumbing into working `POST /auth/register` and
`POST /auth/login` endpoints, with password hashing, JWT issuance, and account lockout.

---

## 1. What we built

| File | Role |
|---|---|
| `app/schemas/auth.py` | Pydantic request/response contracts (`RegisterRequest`, `LoginRequest`, `TokenResponse`, `UserResponse`) |
| `app/core/errors.py` | Shared error envelope + typed exceptions (`DuplicateEmailError`, `InvalidCredentialsError`, `AccountLockedError`) |
| `app/services/auth_service.py` | Business logic: `register_user`, `authenticate_user` (incl. lockout counter) |
| `app/api/auth.py` | Thin HTTP routes: `/auth/register`, `/auth/login` |
| `tests/test_m2_auth.py` | The 4 DoD cases: duplicate email→409, weak password→422, 5 fails→423, happy path→JWT |

**Definition of Done, verified:** a new user can register and log in end-to-end; the token persists
(it's a self-contained, stateless JWT); lockout works exactly at the 5th consecutive failed attempt.
All 5 tests in `test_m2_auth.py` pass.

---

## 2. Authentication architecture

```
Client                         API Gateway (FastAPI)              Database
  │  POST /auth/register           │                                  │
  │  {email, username,             │                                  │
  │   password, health_goal}       │                                  │
  ├────────────────────────────────►                                  │
  │                          Pydantic validates shape                 │
  │                          (min_length=8 on password,                │
  │                           valid health_goal enum)                  │
  │                                │──── check email uniqueness ──────►
  │                                │◄──── not found ────────────────────
  │                                │  hash_password() (bcrypt)          │
  │                                │──── INSERT user ──────────────────►
  │                                │  create_access_token(user_id)      │
  │  ◄─────── 201 {access_token, user} ─────────────────────────────────
```

Login follows the same shape but looks the existing user up, checks `account_locked` first, then
verifies the password with bcrypt.

---

## 3. JWT architecture

- **Stateless**: the token itself carries `sub` (user id) and `exp` (expiry). No server-side session
  table to look up — any process holding `SECRET_KEY` can verify any token instantly.
- **Signed, not encrypted**: the payload is readable (base64 JSON) by anyone; what's protected is
  *tamper-resistance* — nobody without `SECRET_KEY` can produce a valid signature for a forged or
  modified payload.
- **HS256 (symmetric)**: the same secret signs and verifies. Chosen over RS256 (asymmetric) because
  NutriLens has exactly one backend that both issues and checks tokens — no need for a public/private
  key split that RS256 would justify (e.g. if a separate service needed to verify without being able
  to issue tokens).
- **Expiry (60 minutes)**: bounds the damage if a token is stolen — the stateless-token tradeoff is
  that a token can't be instantly revoked before it naturally expires, so keeping expiry short is the
  mitigation.

---

## 4. Password hashing architecture

- **bcrypt**, via `passlib`'s `CryptContext`. Deliberately *slow* (tunable cost factor) — the opposite
  of what you want from a general-purpose hash like SHA-256, which is fast (great for hashing large
  files, terrible for passwords, since a stolen hash database becomes cheap to brute-force).
- **Salting is automatic and built into the stored value** — bcrypt embeds a random salt inside the
  hash string itself, so no separate `salt` column is needed, and two users with the same password
  get completely different stored hashes.
- **Never store, never log, never return the plaintext password** — `RegisterRequest.password` and
  `LoginRequest.password` exist only as request-side Pydantic fields; the ORM `User` model has no
  `password` field at all, only `password_hash`.

---

## 5. Why bcrypt (security best practice)

A stolen password database is a realistic threat model — bcrypt's deliberate slowness means brute-
forcing even a single password from a stolen hash is computationally expensive, at a cost factor that
can be increased over time as hardware gets faster. Fast hashes (MD5, SHA-1, plain SHA-256) offer no
such protection and are considered unacceptable for password storage industry-wide.

## 6. Why JWT (security best practice)

Stateless verification avoids a server-side session store, simplifying horizontal scaling — any
instance of the backend can verify any token using only the shared secret, with zero database
round-trip. The tradeoff (can't instantly revoke a stolen token) is mitigated by a short expiry.

## 7. Why account locking (security best practice)

Without a lockout, a login endpoint is an open invitation to brute-force a password by trying
millions of guesses. Locking after 5 consecutive failures makes online brute-forcing impractical,
while resetting the counter on any success avoids permanently penalising a user who occasionally
mistypes their password.

---

## 8. Likely graduation-discussion questions (with model answers)

**Q: What exactly happens on the 5th wrong password attempt?**
A: `authenticate_user` increments `failed_login_count`; the moment it reaches
`settings.max_failed_logins` (5), `account_locked` is set to `True` in the *same* transaction. The
response to that 5th attempt is still 401 (indistinguishable from any other wrong password) — the
lock only becomes visible as 423 on the *next* attempt, so an attacker probing passwords can't tell
which specific guess triggered the lock.

**Q: Why not tell the user "3 attempts remaining" for better UX?**
A: That would leak information useful to an attacker (confirming the account exists and how close
they are to triggering different server behaviour). NutriLens keeps the failure response identical
regardless of remaining attempts.

**Q: What HTTP status codes does this API use for auth failures, and why those specific ones?**
A: 409 for duplicate email (Conflict — the request conflicts with existing state), 422 for a
malformed request body like a too-short password (Pydantic-level validation failure, distinct from
an application-logic failure), 401 for wrong credentials (Unauthorized), and 423 for a locked account
(the standard HTTP code for "this resource is locked," more specific than a generic 401).

**Q: Why does `/auth/register` also return a JWT instead of just a success message?**
A: Usability — the report's Usability NFR expects the fewest interactions to reach the core workflow;
requiring a separate manual login step immediately after registering would be an avoidable extra
step for a first-time user.

**Q: How would you support "forgot password" or unlocking a locked account?**
A: Out of scope for M2/FR02 as specified, but the natural next iteration would be an email-based
password-reset flow, and either a timed auto-unlock (e.g. after 30 minutes) or an admin-triggered
unlock endpoint — both explicitly acknowledged as future work rather than implemented here.

---

## 9. Common mistakes to watch for (and avoid saying you made)

- Checking the password *before* checking `account_locked` — wastes a bcrypt comparison on an
  account that's going to be rejected regardless, and can leak timing information.
- Forgetting to reset `failed_login_count` on a successful login — would let failures silently
  accumulate across unrelated login sessions days or weeks apart.
- Returning different error messages for "email not found" vs "wrong password" — reveals which
  emails are registered; NutriLens's `authenticate_user` returns the same `InvalidCredentialsError`
  for both.
- Building error JSON by hand in each route instead of raising a typed `AppError` — leads to
  inconsistent response shapes the mobile client can't parse uniformly.

Prev: `M1_EXPLANATION_GUIDE.md` · Next: M3 — Recognition (needs the trained model)
