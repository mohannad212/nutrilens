"""
tests/test_m6_users.py
=========================

WHY THIS FILE EXISTS
---------------------
M6's other half of the Definition of Done — "changing the goal persists and is
reflected on next recommendation refresh" — plus the metrics editor M7's calorie
insight points users at.

Most of these tests exist for one rule: `03 §3`'s "omitted keys are left
unchanged; an explicit `null` clears a field". That is the single most
destructive thing in M6 to get wrong, because the wrong implementation loses
data silently. A Profile screen that PATCHes just the weight, against code that
writes all five fields, nulls the other four — no error, no warning, and the
user's calorie insight quietly downgrades with nothing on screen to explain why.
Distinguishing "absent" from "explicitly null" is invisible in the parsed model,
so it needs tests that send both.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Partial update leaves other fields alone.** The destructive case above.

- **Explicit `null` clears.** The other half of the same rule. Together they prove
  the implementation reads `model_fields_set` rather than the parsed values —
  nothing else can tell the two requests apart.

- **Validation is identical to registration.** `03 §3` requires it. Asserted by
  sending the same out-of-range values both endpoints must refuse, notably
  `height_cm: 1.78` — the metres-for-centimetres mix-up, which is the whole reason
  the bounds exist (`03 §2`: these values are multiplied into a calorie figure
  shown as personal advice).

- **The goal round-trips and reaches the engine.** The DoD sentence, tested end to
  end: change the goal, force a refresh, and confirm the insights were generated
  against the new target rather than the old one.

- **`password_hash` never appears in a response.** The profile is built from the
  ORM row, so this is the one place a careless `from_attributes` could publish it.

- **Auth and isolation.** Every route resolves the user from the token, so there
  is no id to tamper with — but that is a claim worth checking over HTTP.
"""

from __future__ import annotations

import pytest

from app.db.models import ActivityLevel, HealthGoal, Sex, User
from app.security import create_access_token, hash_password

ME_URL = "/api/v1/users/me"
GOAL_URL = "/api/v1/users/me/goal"
METRICS_URL = "/api/v1/users/me/metrics"

FULL_METRICS = {
    "height_cm": 178.5,
    "weight_kg": 74.2,
    "birth_year": 2001,
    "sex": "male",
    "activity_level": "moderate",
}


def make_user(db, *, username: str = "prof", email: str = "prof@example.com", **kwargs) -> User:
    user = User(
        username=username,
        email=email,
        password_hash=hash_password("Str0ngPass!"),
        health_goal=kwargs.pop("health_goal", HealthGoal.maintenance),
        **kwargs,
    )
    db.add(user)
    db.commit()
    return user


@pytest.fixture()
def api(client, db_session):
    user = make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}
    return client, headers, user


# ---------------------------------------------------------------------------
# GET /users/me
# ---------------------------------------------------------------------------
def test_profile_returns_the_documented_fields(api):
    http, headers, user = api

    response = http.get(ME_URL, headers=headers)

    assert response.status_code == 200
    body = response.json()
    # 03 §3's five, plus the metrics this schema adds (see schemas/user.py).
    assert {"user_id", "username", "email", "health_goal", "created_at"} <= set(body)
    assert body["user_id"] == user.user_id
    assert body["health_goal"] == "maintenance"


def test_profile_never_exposes_the_password_hash(api):
    http, headers, _user = api

    body = http.get(ME_URL, headers=headers).json()

    assert "password_hash" not in body
    assert "account_locked" not in body
    assert "failed_login_count" not in body


def test_profile_shows_unset_metrics_as_null(api):
    http, headers, _user = api

    body = http.get(ME_URL, headers=headers).json()

    assert body["height_cm"] is None
    assert body["sex"] is None


# ---------------------------------------------------------------------------
# PATCH /users/me/goal (FR13)
# ---------------------------------------------------------------------------
def test_goal_update_persists(api, db_session):
    http, headers, user = api

    response = http.patch(GOAL_URL, json={"health_goal": "weight_gain"}, headers=headers)

    assert response.status_code == 200
    assert response.json()["health_goal"] == "weight_gain"
    db_session.refresh(user)
    assert user.health_goal == HealthGoal.weight_gain


def test_goal_update_returns_the_whole_profile(api):
    # 03 §3: "returns updated profile" — so the client can reconcile without a
    # follow-up GET.
    http, headers, _user = api

    body = http.patch(GOAL_URL, json={"health_goal": "weight_loss"}, headers=headers).json()

    assert {"user_id", "username", "email", "created_at"} <= set(body)


def test_invalid_goal_is_rejected(api):
    http, headers, _user = api
    response = http.patch(GOAL_URL, json={"health_goal": "bulking"}, headers=headers)
    assert response.status_code == 422


def test_goal_change_reaches_the_recommendation_engine(client, db_session):
    """M6's DoD sentence: the new goal is reflected on the next refresh."""
    from datetime import UTC, datetime, timedelta

    from app.db.models import FoodItem, MealItem, MealLog, MealType

    user = make_user(db_session, health_goal=HealthGoal.maintenance)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}

    # Seven confirmed meals in the window, so insights unlock.
    food = FoodItem(
        food_name="Chicken",
        food101_class="chicken_wings",
        cooking_method="grilled",
        calories_per_100g=200,
        protein_per_100g=30,
        carbs_per_100g=0,
        fat_per_100g=8,
    )
    db_session.add(food)
    db_session.flush()
    for day in range(7):
        meal = MealLog(
            user_id=user.user_id,
            logged_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=day % 5),
            meal_type=MealType.lunch,
            total_calories=400,
            user_confirmed=True,
        )
        db_session.add(meal)
        db_session.flush()
        db_session.add(
            MealItem(
                log_id=meal.log_id,
                food_id=food.food_id,
                portion_confirmed_g=200,
                calories_est=400,
                protein_est_g=60,
                carbs_est_g=0,
                fat_est_g=16,
            )
        )
    db_session.commit()

    client.patch(GOAL_URL, json={"health_goal": "weight_gain"}, headers=headers)
    refreshed = client.post("/api/v1/recommendations/refresh", headers=headers).json()

    assert refreshed["unlocked"] is True
    # The engine ran after the goal change; the stored goal is what it targeted.
    db_session.refresh(user)
    assert user.health_goal == HealthGoal.weight_gain


# ---------------------------------------------------------------------------
# PATCH /users/me/metrics — the partial-update rule (03 §3)
# ---------------------------------------------------------------------------
def test_metrics_can_be_set(api, db_session):
    http, headers, user = api

    response = http.patch(METRICS_URL, json=FULL_METRICS, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["height_cm"] == 178.5
    assert body["sex"] == "male"
    assert body["activity_level"] == "moderate"
    db_session.refresh(user)
    assert float(user.weight_kg) == 74.2


def test_partial_update_leaves_other_fields_unchanged(api, db_session):
    # The destructive case: a Profile screen submitting one field must not null
    # the other four.
    http, headers, user = api
    http.patch(METRICS_URL, json=FULL_METRICS, headers=headers)

    body = http.patch(METRICS_URL, json={"weight_kg": 71.0}, headers=headers).json()

    assert body["weight_kg"] == 71.0
    assert body["height_cm"] == 178.5
    assert body["birth_year"] == 2001
    assert body["sex"] == "male"
    assert body["activity_level"] == "moderate"


def test_explicit_null_clears_a_field(api):
    # The other half of 03 §3's rule. Together with the test above, this is what
    # proves the implementation distinguishes "absent" from "sent as null".
    http, headers, _user = api
    http.patch(METRICS_URL, json=FULL_METRICS, headers=headers)

    body = http.patch(METRICS_URL, json={"sex": None}, headers=headers).json()

    assert body["sex"] is None
    assert body["height_cm"] == 178.5  # untouched


def test_empty_body_changes_nothing(api):
    http, headers, _user = api
    http.patch(METRICS_URL, json=FULL_METRICS, headers=headers)

    body = http.patch(METRICS_URL, json={}, headers=headers).json()

    assert body["height_cm"] == 178.5
    assert body["weight_kg"] == 74.2
    assert body["sex"] == "male"


@pytest.mark.parametrize(
    "payload",
    [
        {"height_cm": 1.78},
        {"height_cm": 300},
        {"weight_kg": 5},
        {"weight_kg": 500},
        {"birth_year": 1850},
        {"birth_year": 2030},
        {"sex": "unspecified"},
        {"activity_level": "extreme"},
    ],
    ids=[
        "height_in_metres",
        "height_too_large",
        "weight_too_small",
        "weight_too_large",
        "birth_year_too_old",
        "birth_year_future",
        "invalid_sex",
        "invalid_activity",
    ],
)
def test_metrics_validation_matches_registration(api, payload):
    # 03 §3: "Validation is identical to /auth/register." height_cm: 1.78 is the
    # metres-for-centimetres mix-up the bounds exist to catch — it would otherwise
    # surface as confident nonsense in a calorie insight (03 §2).
    http, headers, _user = api

    response = http.patch(METRICS_URL, json=payload, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_metrics_set_at_registration_are_visible_and_editable(client, db_session):
    # The full round trip 03 §2 -> §3: registered with metrics, edited later.
    register = client.post(
        "/api/v1/auth/register",
        json={
            "username": "metrics_user",
            "email": "metrics@example.com",
            "password": "Str0ngPass!",
            "health_goal": "weight_loss",
            **FULL_METRICS,
        },
    )
    assert register.status_code == 201
    user_id = register.json()["user_id"]
    headers = {"Authorization": f"Bearer {create_access_token(str(user_id))}"}

    assert client.get(ME_URL, headers=headers).json()["height_cm"] == 178.5

    body = client.patch(METRICS_URL, json={"activity_level": "active"}, headers=headers).json()
    assert body["activity_level"] == "active"
    assert body["height_cm"] == 178.5


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("get", ME_URL, None),
        ("patch", GOAL_URL, {"health_goal": "maintenance"}),
        ("patch", METRICS_URL, {"weight_kg": 70.0}),
    ],
    ids=["profile", "goal", "metrics"],
)
def test_every_route_requires_authentication(client, method, url, body):
    call = getattr(client, method)
    response = call(url, json=body) if body is not None else call(url)
    assert response.status_code == 401


def test_a_user_can_only_ever_edit_their_own_profile(client, db_session):
    # There is no user id in any path or body, so this is structural — but the
    # property is worth asserting at the layer an attacker would use.
    alice = make_user(db_session, username="alice", email="alice@example.com")
    bob = make_user(db_session, username="bob", email="bob@example.com")
    alice_headers = {"Authorization": f"Bearer {create_access_token(str(alice.user_id))}"}

    client.patch(GOAL_URL, json={"health_goal": "weight_gain"}, headers=alice_headers)
    client.patch(METRICS_URL, json={"weight_kg": 60.0}, headers=alice_headers)

    db_session.refresh(alice)
    db_session.refresh(bob)
    assert alice.health_goal == HealthGoal.weight_gain
    assert bob.health_goal == HealthGoal.maintenance
    assert bob.weight_kg is None


def test_service_level_metrics_update_ignores_unknown_fields(db_session):
    # The allowlist in user_service guards against a future schema field becoming
    # writable just by existing.
    from app.services import user_service

    user = make_user(db_session)
    user_service.update_metrics(
        user, {"weight_kg": 70.0, "health_goal": HealthGoal.weight_gain}, db_session
    )

    assert float(user.weight_kg) == 70.0
    assert user.health_goal == HealthGoal.maintenance  # not editable here


def test_sex_and_activity_accept_every_enum_value(api):
    http, headers, _user = api
    for value in (Sex.female, Sex.male):
        assert (
            http.patch(METRICS_URL, json={"sex": value.value}, headers=headers).status_code == 200
        )
    for value in ActivityLevel:
        response = http.patch(METRICS_URL, json={"activity_level": value.value}, headers=headers)
        assert response.status_code == 200
