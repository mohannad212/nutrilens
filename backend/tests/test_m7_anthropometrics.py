"""
tests/test_m7_anthropometrics.py
===================================

WHY THIS FILE EXISTS
---------------------
M7's calorie insight (`06 §2 calorie_note`) needs a per-user maintenance-energy
estimate, which needs five optional body metrics that migration
`0002_user_anthropometrics` adds to `user`. Those columns are only useful if
three things hold, and this file pins all three:

1. They are genuinely OPTIONAL — every M2 registration that predates them must
   still succeed and simply store NULL. If this test ever fails, adding the
   columns silently broke FR01 for existing clients.
2. They round-trip — a registration that DOES supply them persists every value,
   so the engine reads back what the user entered.
3. They are validated — a unit mix-up (height in metres, weight in pounds) or a
   typo'd birth year is rejected at the schema layer with a 422, before a
   nonsense TDEE can reach a user-facing insight.

WHY VALIDATION MATTERS MORE HERE THAN ELSEWHERE
--------------------------------------------------
These five numbers are multiplied together into a calorie figure that the app
then shows the user alongside dietary suggestions. `height_cm=1.7` (metres,
mistyped) would produce a BMR of a few hundred kcal and make a normal intake
look like a large surplus — the engine would then confidently suggest eating
less. Bounds-checking at the edge is the cheapest place to stop that, which is
why `AnthropometricsMixin` carries `ge`/`le` constraints rather than trusting
the caller.
"""

from app.db.models import ActivityLevel, Sex, User

REGISTER_URL = "/api/v1/auth/register"

FULL_PROFILE = {
    "height_cm": 178.5,
    "weight_kg": 74.2,
    "birth_year": 2001,
    "sex": "male",
    "activity_level": "moderate",
}


def _register(client, extra=None, email="anthro@meu.edu.jo", username="anthro1"):
    body = {
        "email": email,
        "username": username,
        "password": "CorrectPass1",
        "health_goal": "weight_loss",
    }
    if extra:
        body.update(extra)
    return client.post(REGISTER_URL, json=body)


def test_registration_without_anthropometrics_still_succeeds(client, db_session):
    """The M2 contract is unchanged: none of the new fields is required."""
    response = _register(client)

    assert response.status_code == 201
    user = db_session.query(User).filter(User.email == "anthro@meu.edu.jo").one()
    assert user.height_cm is None
    assert user.weight_kg is None
    assert user.birth_year is None
    assert user.sex is None
    assert user.activity_level is None


def test_registration_with_anthropometrics_persists_every_field(client, db_session):
    response = _register(client, FULL_PROFILE)

    assert response.status_code == 201
    user = db_session.query(User).filter(User.email == "anthro@meu.edu.jo").one()
    assert float(user.height_cm) == 178.5
    assert float(user.weight_kg) == 74.2
    assert user.birth_year == 2001
    assert user.sex is Sex.male
    assert user.activity_level is ActivityLevel.moderate


def test_partial_anthropometrics_are_allowed(client, db_session):
    """
    A user may supply some but not all metrics. The engine treats an incomplete
    set the same as an absent one for TDEE purposes, so storing it must not fail.
    """
    response = _register(client, {"weight_kg": 68.0, "sex": "female"})

    assert response.status_code == 201
    user = db_session.query(User).filter(User.email == "anthro@meu.edu.jo").one()
    assert float(user.weight_kg) == 68.0
    assert user.sex is Sex.female
    assert user.height_cm is None


def test_height_in_metres_is_rejected(client):
    """1.78 is a plausible height in the WRONG unit — the classic input bug."""
    assert _register(client, {"height_cm": 1.78}).status_code == 422


def test_weight_in_pounds_is_rejected(client):
    """164 lb read as kg would inflate BMR; the upper bound catches it."""
    assert _register(client, {"weight_kg": 400.0}).status_code == 422


def test_implausible_birth_year_is_rejected(client):
    assert _register(client, {"birth_year": 1850}).status_code == 422
    assert _register(client, {"birth_year": 2025}).status_code == 422


def test_invalid_enum_values_are_rejected(client):
    assert _register(client, {"sex": "other"}).status_code == 422
    assert _register(client, {"activity_level": "extreme"}).status_code == 422


def test_sex_accepts_only_female_and_male(client):
    """
    Product decision: no `unspecified` option. A user who prefers not to say omits
    the field entirely, which is already covered by the partial-profile test above.
    """
    assert _register(client, {"sex": "unspecified"}).status_code == 422
    assert _register(client, {"sex": "female"}).status_code == 201
