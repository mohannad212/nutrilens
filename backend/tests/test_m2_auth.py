"""
tests/test_m2_auth.py
========================

WHY THIS FILE EXISTS
---------------------
M2's Definition of Done (`09-build-phases.md`) lists four specific
behaviours to verify: duplicate email → 409, weak password → 422, 5
consecutive failed logins → 423, and a happy-path login/register that
issues a JWT. This file is exactly those four tests, each using the
`client` fixture (a real FastAPI TestClient hitting real `/auth/*` HTTP
routes against a throwaway in-memory database) so they exercise the
full stack — schema validation, service logic, and the error envelope —
not just isolated units.

WHAT EACH TEST VALIDATES AND WHY IT MATTERS
-----------------------------------------------
- test_register_happy_path_issues_jwt: proves the core "it works" path —
  a well-formed registration returns 201 and a real, non-empty JWT the
  mobile app can store and send on subsequent requests.

- test_duplicate_email_returns_409: proves the uniqueness rule from FR01
  is enforced end-to-end through the HTTP layer, and that the specific
  status code (409, not a generic 400/500) is what a supervisor would
  check for during the API walkthrough.

- test_weak_password_returns_422: proves Pydantic's schema-level
  `min_length=8` constraint actually rejects a too-short password before
  any service logic runs, with FastAPI's standard 422 validation-error
  status.

- test_login_happy_path_issues_jwt: proves a registered user can log in
  with correct credentials and receive a working token — the second half
  of FR02, independent of the registration flow.

- test_five_failed_logins_locks_account_with_423: proves the lockout
  counter in `auth_service.authenticate_user` actually fires at the
  correct threshold — attempts 1–4 return 401 (wrong password), the 5th
  wrong attempt also returns 401 (locking happens on that same request,
  see auth_service's docstring for why), and critically the *6th*
  attempt — even with the CORRECT password — now returns 423, proving the
  lock, not just the wrong-password count, is what's being tested.
"""

REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"


def _register(client, email="student@meu.edu.jo", password="CorrectPass1", username="student1"):
    return client.post(
        REGISTER_URL,
        json={
            "email": email,
            "username": username,
            "password": password,
            "health_goal": "maintenance",
        },
    )


def test_register_happy_path_returns_profile(client):
    response = _register(client)
    assert response.status_code == 201

    body = response.json()
    # 03 §2: register returns the created profile (no token), incl. created_at.
    assert body["email"] == "student@meu.edu.jo"
    assert body["username"] == "student1"
    assert body["health_goal"] == "maintenance"
    assert "user_id" in body and "created_at" in body
    assert "access_token" not in body
    assert "password" not in body and "password_hash" not in body


def test_duplicate_email_returns_409(client):
    first = _register(client, email="dup@meu.edu.jo", username="user_one")
    assert first.status_code == 201

    second = _register(client, email="dup@meu.edu.jo", username="user_two")
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "DUPLICATE_EMAIL"


def test_duplicate_username_returns_409(client):
    first = _register(client, email="a@meu.edu.jo", username="samename")
    assert first.status_code == 201

    second = _register(client, email="b@meu.edu.jo", username="samename")
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "DUPLICATE_USERNAME"


def test_weak_password_too_short_returns_422(client):
    response = _register(client, password="Ab1")  # 3 chars, below min_length=8
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_password_without_digit_returns_422(client):
    # 03 §2: password must contain a letter AND a digit — this one has no digit.
    response = _register(client, password="onlyletters")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_login_happy_path_issues_jwt(client):
    _register(client, email="login@meu.edu.jo", password="CorrectPass1", username="loginuser")

    response = client.post(
        LOGIN_URL, json={"email": "login@meu.edu.jo", "password": "CorrectPass1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["access_token"], str) and len(body["access_token"]) > 20
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 3600
    assert body["user"]["user_id"] and body["user"]["username"] == "loginuser"
    assert "email" not in body["user"]  # login user object is minimal (03 §2)


def test_five_failed_logins_locks_account_with_423(client):
    email = "lockout@meu.edu.jo"
    correct_password = "CorrectPass1"
    _register(client, email=email, password=correct_password, username="lockoutuser")

    # Attempts 1 through 5 with the WRONG password: all return 401 and report
    # the decreasing number of remaining attempts (03 §2).
    for attempt in range(1, 6):
        response = client.post(LOGIN_URL, json={"email": email, "password": "WrongPassword"})
        assert response.status_code == 401, f"attempt {attempt} should be 401"
        body = response.json()
        assert body["error"]["code"] == "INVALID_CREDENTIALS"
        assert body["error"]["details"]["remaining_attempts"] == 5 - attempt

    # 6th attempt, even with the CORRECT password, must now be 423 — the
    # account is locked, not just "still has one guess left."
    response = client.post(LOGIN_URL, json={"email": email, "password": correct_password})
    assert response.status_code == 423
    assert response.json()["error"]["code"] == "ACCOUNT_LOCKED"
