"""
tests/test_m7_api.py
=======================

WHY THIS FILE EXISTS
---------------------
The service tests proved the engine works against real rows. These prove the
*contract* works: that the JSON matches `03 §7` field for field, that the status
codes are the ones the mobile client branches on, and that authentication actually
gates every route. They go through a real `TestClient` and a real JWT, so a route
that forgot its `Depends(get_current_user)` fails here and nowhere else.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Auth on every route.** Asserted per-endpoint rather than once, because the
  protection is per-decorator: adding a fourth route and forgetting the dependency
  is a realistic mistake, and the consequence is one user reading another's
  dietary insights.

- **200 when locked, not 4xx.** `03 §7` is explicit that fewer than 7 meals is a
  normal state. If this ever regressed to a 403, the app's error handler would
  show "something went wrong" to a user whose only problem is that they are new.

- **The exact response keys.** Compared as a set against `03 §7`, because a
  missing `meals_required` or a renamed `insight_text` breaks the TypeScript types
  in `mobile/src/types/` — and the PRD calls this document a contract.

- **The disclaimer on every insight response.** The `06 §5` NFR, checked at the
  HTTP boundary. Everything earlier proves the string exists in the right module;
  only this proves it actually reaches the wire.

- **`user_id` never appears in a response.** The ORM row has one. Publishing it
  would leak an internal identifier, so the omission is asserted rather than
  assumed from the schema definition.

- **Per-user isolation over HTTP.** Two registered users, two tokens: one must not
  see or mark the other's insights. This is the security property `11` requires,
  tested at the layer an attacker would actually use.
"""

from datetime import UTC, datetime, timedelta

from app.db.models import FoodItem, HealthGoal, MealItem, MealLog, MealType, User
from app.security import create_access_token

RECOMMENDATIONS_URL = "/api/v1/recommendations"
REFRESH_URL = "/api/v1/recommendations/refresh"

# 03 §7 — the documented keys of the recommendations payload.
EXPECTED_KEYS = {
    "unlocked",
    "insights",
    "generated_at",
    "meals_logged",
    "meals_required",
    "message",
    "disclaimer",
}
EXPECTED_INSIGHT_KEYS = {"rec_id", "insight_text", "nutrient_focus", "was_read"}


def make_user(db, *, email="api@meu.edu.jo", username="apiuser") -> User:
    user = User(
        username=username,
        email=email,
        password_hash="x" * 20,
        health_goal=HealthGoal.weight_loss,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(subject=user.user_id)}"}


def seed_meals(db, user: User, *, count: int, confirmed: bool = True) -> None:
    """`count` confirmed meals spread over the last few days, each with macros."""
    food = FoodItem(
        food_name="Grilled Chicken",
        food101_class="grilled_chicken",
        calories_per_100g=200.0,
        protein_per_100g=30.0,
        carbs_per_100g=2.0,
        fat_per_100g=5.0,
    )
    db.add(food)
    db.flush()

    # Naive UTC, matching the TIMESTAMP columns in 02 and the service's `_now()`.
    now = datetime.now(UTC).replace(tzinfo=None)
    for index in range(count):
        log = MealLog(
            user_id=user.user_id,
            logged_at=now - timedelta(hours=6 * index),
            meal_type=MealType.lunch,
            total_calories=620.0,
            user_confirmed=confirmed,
        )
        db.add(log)
        db.flush()
        db.add(
            MealItem(
                log_id=log.log_id,
                food_id=food.food_id,
                portion_confirmed_g=200.0,
                calories_est=620.0,
                protein_est_g=35.0,
                carbs_est_g=60.0,
                fat_est_g=20.0,
            )
        )
    db.commit()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_get_requires_a_token(client):
    assert client.get(RECOMMENDATIONS_URL).status_code == 401


def test_refresh_requires_a_token(client):
    assert client.post(REFRESH_URL).status_code == 401


def test_mark_read_requires_a_token(client):
    assert client.patch(f"{RECOMMENDATIONS_URL}/1/read").status_code == 401


def test_an_invalid_token_is_rejected(client):
    response = client.get(RECOMMENDATIONS_URL, headers={"Authorization": "Bearer nonsense"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# The locked state (03 §7)
# ---------------------------------------------------------------------------


def test_locked_user_gets_200_not_an_error(client, db_session):
    """
    A new user is not a failure. A 4xx here would make the app show an error
    screen to someone whose only problem is that they have just signed up.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=3)

    response = client.get(RECOMMENDATIONS_URL, headers=auth_header(user))

    assert response.status_code == 200
    body = response.json()
    assert body["unlocked"] is False
    assert body["insights"] == []
    assert body["meals_logged"] == 3
    assert body["meals_required"] == 7
    assert "4 more meals" in body["message"]


def test_locked_response_has_the_documented_keys(client, db_session):
    user = make_user(db_session)
    response = client.get(RECOMMENDATIONS_URL, headers=auth_header(user))
    assert set(response.json()) == EXPECTED_KEYS


def test_locked_response_still_carries_the_disclaimer(client, db_session):
    """06 §5: every response, not only the ones with insights in them."""
    user = make_user(db_session)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()
    assert "qualified healthcare professional" in body["disclaimer"]


def test_refresh_is_a_no_op_while_locked(client, db_session):
    """03 §7: refresh must be safe to call in the locked state, not an error."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=2)

    response = client.post(REFRESH_URL, headers=auth_header(user))

    assert response.status_code == 200
    assert response.json()["unlocked"] is False


# ---------------------------------------------------------------------------
# The unlocked state (03 §7)
# ---------------------------------------------------------------------------


def test_unlocked_user_gets_insights(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)

    response = client.get(RECOMMENDATIONS_URL, headers=auth_header(user))

    assert response.status_code == 200
    body = response.json()
    assert body["unlocked"] is True
    assert 1 <= len(body["insights"]) <= 5
    assert body["generated_at"] is not None


def test_unlocked_response_has_the_documented_keys(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()
    assert set(body) == EXPECTED_KEYS


def test_each_insight_has_exactly_the_documented_fields(client, db_session):
    """
    Field-for-field against 03 §7 — a rename here breaks mobile/src/types/.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()

    for insight in body["insights"]:
        assert set(insight) == EXPECTED_INSIGHT_KEYS


def test_insights_never_expose_user_id(client, db_session):
    """The ORM row carries one; the response must not."""
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()

    assert "user_id" not in body
    for insight in body["insights"]:
        assert "user_id" not in insight


def test_unlocked_response_carries_the_disclaimer(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()

    assert "informational purposes only" in body["disclaimer"]
    assert "qualified healthcare professional" in body["disclaimer"]


def test_insights_start_unread(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()

    assert all(insight["was_read"] is False for insight in body["insights"])


def test_unconfirmed_meals_do_not_unlock_over_http(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=9, confirmed=False)

    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(user)).json()
    assert body["unlocked"] is False
    assert body["meals_logged"] == 0


# ---------------------------------------------------------------------------
# Caching and refresh
# ---------------------------------------------------------------------------


def test_repeated_gets_return_the_same_insights(client, db_session):
    """
    06 §3's idempotency, from the client's side: opening the Insights screen twice
    must not reshuffle the advice.
    """
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)

    first = client.get(RECOMMENDATIONS_URL, headers=headers).json()
    second = client.get(RECOMMENDATIONS_URL, headers=headers).json()

    assert [i["rec_id"] for i in first["insights"]] == [i["rec_id"] for i in second["insights"]]


def test_refresh_regenerates(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)

    first = client.get(RECOMMENDATIONS_URL, headers=headers).json()
    refreshed = client.post(REFRESH_URL, headers=headers).json()

    first_ids = {i["rec_id"] for i in first["insights"]}
    new_ids = {i["rec_id"] for i in refreshed["insights"]}
    assert first_ids.isdisjoint(new_ids)


def test_refresh_returns_the_same_shape_as_get(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)

    assert set(client.post(REFRESH_URL, headers=headers).json()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# PATCH /recommendations/{rec_id}/read
# ---------------------------------------------------------------------------


def test_mark_read_flips_the_flag(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)
    rec_id = client.get(RECOMMENDATIONS_URL, headers=headers).json()["insights"][0]["rec_id"]

    response = client.patch(f"{RECOMMENDATIONS_URL}/{rec_id}/read", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"rec_id": rec_id, "was_read": True}


def test_read_state_survives_a_subsequent_get(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)
    body = client.get(RECOMMENDATIONS_URL, headers=headers).json()
    rec_id = body["insights"][0]["rec_id"]

    client.patch(f"{RECOMMENDATIONS_URL}/{rec_id}/read", headers=headers)
    after = client.get(RECOMMENDATIONS_URL, headers=headers).json()

    marked = next(i for i in after["insights"] if i["rec_id"] == rec_id)
    assert marked["was_read"] is True


def test_mark_read_is_idempotent_over_http(client, db_session):
    user = make_user(db_session)
    seed_meals(db_session, user, count=8)
    headers = auth_header(user)
    rec_id = client.get(RECOMMENDATIONS_URL, headers=headers).json()["insights"][0]["rec_id"]

    client.patch(f"{RECOMMENDATIONS_URL}/{rec_id}/read", headers=headers)
    second = client.patch(f"{RECOMMENDATIONS_URL}/{rec_id}/read", headers=headers)

    assert second.status_code == 200


def test_mark_read_unknown_id_returns_404_with_the_error_envelope(client, db_session):
    user = make_user(db_session)
    response = client.patch(f"{RECOMMENDATIONS_URL}/424242/read", headers=auth_header(user))

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_mark_read_invalid_id_type_returns_422(client, db_session):
    user = make_user(db_session)
    response = client.patch(f"{RECOMMENDATIONS_URL}/abc/read", headers=auth_header(user))
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Per-user isolation (11, NFR security)
# ---------------------------------------------------------------------------


def test_one_user_cannot_see_anothers_insights(client, db_session):
    mine = make_user(db_session, email="a@x.com", username="usera")
    theirs = make_user(db_session, email="b@x.com", username="userb")
    seed_meals(db_session, theirs, count=8)

    body = client.get(RECOMMENDATIONS_URL, headers=auth_header(mine)).json()

    assert body["unlocked"] is False
    assert body["insights"] == []


def test_one_user_cannot_mark_anothers_insight_read(client, db_session):
    mine = make_user(db_session, email="a@x.com", username="usera")
    theirs = make_user(db_session, email="b@x.com", username="userb")
    seed_meals(db_session, theirs, count=8)
    their_rec_id = client.get(RECOMMENDATIONS_URL, headers=auth_header(theirs)).json()["insights"][
        0
    ]["rec_id"]

    response = client.patch(f"{RECOMMENDATIONS_URL}/{their_rec_id}/read", headers=auth_header(mine))

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"

    # And the target row is untouched.
    after = client.get(RECOMMENDATIONS_URL, headers=auth_header(theirs)).json()
    target = next(i for i in after["insights"] if i["rec_id"] == their_rec_id)
    assert target["was_read"] is False


# ---------------------------------------------------------------------------
# OpenAPI registration
# ---------------------------------------------------------------------------


def test_all_three_routes_are_registered(client):
    """Guards against a router that exists but was never mounted in main.py."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "get" in paths["/api/v1/recommendations"]
    assert "post" in paths["/api/v1/recommendations/refresh"]
    assert "patch" in paths["/api/v1/recommendations/{rec_id}/read"]
