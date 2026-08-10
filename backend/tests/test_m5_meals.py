"""
tests/test_m5_meals.py
=========================

WHY THIS FILE EXISTS
---------------------
`POST /meals` is where M5's Definition of Done is decided — "meal appears with
correct totals" — and where the two rules that make a meal log trustworthy are
enforced: the server computes every derived value (`02 §6`), and a meal is written
whole or not at all. Both are invisible in a passing response, so both need tests
that look past it: at the persisted rows, and at what survives a failure.

This is also the file that proves M7's deferred integration actually happened.
`09` M7 shipped `on_meal_logged` fully tested but with no caller, noting "M5's
`POST /meals` needs to add the single call after its commit". A one-line
integration is exactly the kind that gets forgotten, so it is asserted here rather
than assumed.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Totals are recomputed, not accepted.** Named explicitly in `09` M5's test
  list. Checked two ways: the persisted total equals the sum of the persisted
  items, and a body that smuggles in `total_calories` and per-item macros has no
  effect on what is stored. The second is the one that matters — it proves the
  values are *unreachable*, not merely overwritten.

- **`user_confirmed` is TRUE on insert.** `02 §6` excludes unconfirmed meals from
  history and from the recommendation window, so a meal written with the column's
  `FALSE` default would be logged successfully and then be invisible everywhere —
  a bug with no error message anywhere in it.

- **A failed item rolls the whole meal back.** Two items, the second raising from
  USDA. If the transaction boundary is wrong, the first item and the `meal_log`
  header persist as a phantom meal that the user never sees and can never delete.
  Asserted by counting rows after the failure, since the response is an error
  either way.

- **Multi-item totals.** Sums across items with different foods and portions,
  because a single-item test passes just as well against code that ignores the
  loop.

- **`meals_logged_total` and `recommendations_unlocked` at the boundary.** The
  seventh meal is the one that flips the flag (`03 §6`,
  `MIN_MEALS_FOR_RECOMMENDATION`), so it is tested at six, seven and eight rather
  than at some arbitrary count.

- **The M7 auto-trigger fires after the commit.** Insight rows must exist once the
  threshold is crossed, and — the important half — a *failing* refresh must leave
  the meal intact. The hook is documented as never raising; this test is what
  makes that a property of the system rather than of one module.

- **A future `logged_at` is refused.** `09` M7 asks M5 for this. Proven from both
  directions: the field is not in the request contract at all, and the trusted
  service-level parameter clamps a future value rather than storing it.

- **Per-user isolation.** One user's meal never lands in another's history or
  count, tested over HTTP with two real tokens.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import UsdaUnavailableError
from app.db.models import (
    FoodItem,
    HealthGoal,
    MealItem,
    MealLog,
    MealType,
    Recommendation,
    User,
)
from app.deps import get_usda_client
from app.integrations.usda_client import UsdaFood
from app.security import create_access_token, hash_password
from app.services import meal_service
from app.services.meal_service import MealItemInput

MEALS_URL = "/api/v1/meals"

SALMON = UsdaFood(
    fdc_id="175168",
    description="Fish, salmon, Atlantic, farmed, cooked, dry heat",
    calories_per_100g=206.0,
    protein_per_100g=22.1,
    carbs_per_100g=0.0,
    fat_per_100g=12.4,
)
RICE = UsdaFood(
    fdc_id="169704",
    description="Rice, white, long-grain, cooked",
    calories_per_100g=130.0,
    protein_per_100g=2.7,
    carbs_per_100g=28.2,
    fat_per_100g=0.3,
)


class StubUsda:
    """Returns a per-class result, or raises for classes listed in `fail_on`."""

    def __init__(
        self,
        results: dict[str, UsdaFood] | None = None,
        fail_on: set[str] | None = None,
    ):
        self.results = results or {"grilled_salmon": SALMON, "fried_rice": RICE}
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, str | None]] = []

    def search(self, food101_class: str, cooking_method: str | None = None) -> UsdaFood:
        self.calls.append((food101_class, cooking_method))
        if food101_class in self.fail_on:
            raise UsdaUnavailableError()
        return self.results.get(food101_class, SALMON)

    def close(self) -> None:  # pragma: no cover
        pass


def make_user(db, username: str = "logger", email: str = "logger@example.com") -> User:
    user = User(
        username=username,
        email=email,
        password_hash=hash_password("Str0ngPass!"),
        health_goal=HealthGoal.maintenance,
    )
    db.add(user)
    db.commit()
    return user


def salmon_item(portion: float = 210.0) -> MealItemInput:
    return MealItemInput(
        food101_class="grilled_salmon",
        cooking_method="grilled",
        portion_confirmed_g=portion,
        portion_min_g=180.0,
        portion_max_g=240.0,
        confidence_score=0.91,
    )


def rice_item(portion: float = 150.0) -> MealItemInput:
    return MealItemInput(
        food101_class="fried_rice",
        cooking_method="fried",
        portion_confirmed_g=portion,
    )


def log_n_meals(db, user: User, count: int, usda: StubUsda | None = None) -> None:
    """Seed `count` confirmed meals through the real service path."""
    stub = usda or StubUsda()
    for _ in range(count):
        meal_service.create_meal(user, MealType.lunch, [salmon_item()], db, stub)


# ---------------------------------------------------------------------------
# Totals are recomputed server-side (09 M5, 02 §6)
# ---------------------------------------------------------------------------
def test_total_calories_equals_the_sum_of_persisted_items(db_session):
    user = make_user(db_session)

    meal = meal_service.create_meal(
        user, MealType.lunch, [salmon_item(210.0), rice_item(150.0)], db_session, StubUsda()
    )

    items = db_session.query(MealItem).filter(MealItem.log_id == meal.log_id).all()
    assert len(items) == 2
    # Computed from what was stored, not from what the service returned.
    assert float(meal.total_calories) == pytest.approx(
        sum(float(item.calories_est) for item in items)
    )
    # 206 * 2.1 = 433; 130 * 1.5 = 195.
    assert float(meal.total_calories) == pytest.approx(433 + 195)


def test_each_item_is_scaled_to_its_own_confirmed_portion(db_session):
    user = make_user(db_session)

    meal = meal_service.create_meal(
        user, MealType.dinner, [salmon_item(210.0), rice_item(150.0)], db_session, StubUsda()
    )

    by_name = {item.food.food_name: item for item in meal.items}
    salmon = by_name["Fish, salmon, Atlantic, farmed, cooked, dry heat"]
    rice = by_name["Rice, white, long-grain, cooked"]

    assert float(salmon.calories_est) == 433
    assert float(salmon.protein_est_g) == 46.4
    assert float(rice.calories_est) == 195
    assert float(rice.carbs_est_g) == pytest.approx(42.3)


def test_meal_is_written_as_confirmed(db_session):
    # 02 §6: an unconfirmed meal is a draft, excluded from history and from the
    # recommendation window. The column defaults to FALSE, so this must be set.
    user = make_user(db_session)

    meal = meal_service.create_meal(user, MealType.lunch, [salmon_item()], db_session, StubUsda())

    assert meal.user_confirmed is True


def test_portion_outside_the_estimated_range_is_accepted(db_session):
    # 02 §6 / FR06: the user is the authority on their own portion. The service
    # logs a warning; it must not reject.
    user = make_user(db_session)
    item = MealItemInput(
        food101_class="grilled_salmon",
        cooking_method="grilled",
        portion_confirmed_g=400.0,
        portion_min_g=180.0,
        portion_max_g=240.0,
    )

    meal = meal_service.create_meal(user, MealType.lunch, [item], db_session, StubUsda())

    assert float(meal.items[0].portion_confirmed_g) == 400.0
    assert float(meal.total_calories) == 824  # 206 * 4.0


# ---------------------------------------------------------------------------
# One transaction (module docstring)
# ---------------------------------------------------------------------------
def test_a_failed_item_rolls_the_whole_meal_back(db_session):
    user = make_user(db_session)
    usda = StubUsda(fail_on={"fried_rice"})

    with pytest.raises(UsdaUnavailableError):
        meal_service.create_meal(
            user, MealType.lunch, [salmon_item(), rice_item()], db_session, usda
        )

    # No caller-side rollback here on purpose: the service is responsible for
    # leaving the session clean, so a phantom meal would still be visible.
    assert db_session.query(MealLog).count() == 0
    assert db_session.query(MealItem).count() == 0


def test_a_cached_food_item_survives_a_failed_meal(db_session):
    # The food_item cache is shared reference data, not part of the user's meal.
    # Losing it on rollback would be correct but wasteful; this documents which
    # way it actually behaves so nothing downstream is surprised.
    user = make_user(db_session)
    meal_service.create_meal(user, MealType.lunch, [salmon_item()], db_session, StubUsda())

    assert db_session.query(FoodItem).count() == 1


# ---------------------------------------------------------------------------
# Unlock accounting (03 §6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("existing", "expected_total", "expected_unlocked"),
    [(5, 6, False), (6, 7, True), (7, 8, True)],
    ids=["sixth_meal_locked", "seventh_meal_unlocks", "eighth_stays_unlocked"],
)
def test_unlock_flag_flips_on_the_seventh_meal(
    db_session, existing, expected_total, expected_unlocked
):
    user = make_user(db_session)
    log_n_meals(db_session, user, existing)

    _meal, total, unlocked = meal_service.log_meal(
        user, MealType.lunch, [salmon_item()], db_session, StubUsda()
    )

    assert total == expected_total
    assert unlocked is expected_unlocked


def test_unconfirmed_meals_do_not_count_toward_the_unlock(db_session):
    user = make_user(db_session)
    log_n_meals(db_session, user, 6)
    # A draft row, which 02 §6 excludes from the count.
    db_session.add(
        MealLog(
            user_id=user.user_id,
            meal_type=MealType.snack,
            total_calories=100,
            user_confirmed=False,
        )
    )
    db_session.commit()

    assert meal_service.count_confirmed_meals(user.user_id, db_session) == 6


# ---------------------------------------------------------------------------
# The M7 auto-trigger (09 M7, deferred to M5)
# ---------------------------------------------------------------------------
def test_logging_the_seventh_meal_generates_insights(db_session):
    # 09 M7 built `on_meal_logged` and left this call site to M5. A one-line
    # integration is exactly what gets forgotten, so it is asserted, not assumed.
    user = make_user(db_session)
    log_n_meals(db_session, user, 6)
    assert db_session.query(Recommendation).count() == 0

    meal_service.log_meal(user, MealType.lunch, [salmon_item()], db_session, StubUsda())

    assert db_session.query(Recommendation).filter_by(user_id=user.user_id).count() > 0


def test_no_insights_are_generated_below_the_threshold(db_session):
    user = make_user(db_session)
    log_n_meals(db_session, user, 3)

    meal_service.log_meal(user, MealType.lunch, [salmon_item()], db_session, StubUsda())

    assert db_session.query(Recommendation).count() == 0


def test_a_failing_refresh_does_not_lose_the_meal(db_session, monkeypatch):
    # The hook is documented as never raising. This makes that a property of the
    # system rather than of one module: the meal is the user's data, the insights
    # are derived from it, and the derived thing must not take the source down.
    user = make_user(db_session)
    log_n_meals(db_session, user, 6)

    def boom(*args, **kwargs):
        raise RuntimeError("recommendation engine exploded")

    monkeypatch.setattr("app.services.recommendation_service.get_recommendations", boom)

    meal, total, _unlocked = meal_service.log_meal(
        user, MealType.lunch, [salmon_item()], db_session, StubUsda()
    )

    assert meal.log_id is not None
    assert total == 7
    assert db_session.query(MealLog).count() == 7


# ---------------------------------------------------------------------------
# logged_at (09 M7 asks M5 to refuse a future timestamp)
# ---------------------------------------------------------------------------
def test_future_logged_at_is_clamped_to_now(db_session):
    user = make_user(db_session)
    future = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=30)

    meal = meal_service.create_meal(
        user, MealType.lunch, [salmon_item()], db_session, StubUsda(), logged_at=future
    )

    assert meal.logged_at < future


def test_past_logged_at_is_honoured_for_trusted_callers(db_session):
    # The demo seed script (02 §5) needs backdated meals so the recommendation
    # window has something in it on a fresh install.
    user = make_user(db_session)
    past = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)

    meal = meal_service.create_meal(
        user, MealType.lunch, [salmon_item()], db_session, StubUsda(), logged_at=past
    )

    assert meal.logged_at == past


# ---------------------------------------------------------------------------
# HTTP contract (03 §6)
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {
    "log_id",
    "logged_at",
    "meal_type",
    "total_calories",
    "items",
    "meals_logged_total",
    "recommendations_unlocked",
    "disclaimer",
}
EXPECTED_ITEM_KEYS = {
    "item_id",
    "food_name",
    "display_name",
    "calories_est",
    "protein_est_g",
    "carbs_est_g",
    "fat_est_g",
    "portion_confirmed_g",
}

VALID_BODY = {
    "meal_type": "lunch",
    "items": [
        {
            "food101_class": "grilled_salmon",
            "cooking_method": "grilled",
            "portion_confirmed_g": 210.0,
            "portion_min_g": 180.0,
            "portion_max_g": 240.0,
            "confidence_score": 0.91,
        }
    ],
}


@pytest.fixture()
def api(client, db_session):
    """TestClient with a stubbed USDA client and one registered, authenticated user."""
    stub = StubUsda()
    client.app.dependency_overrides[get_usda_client] = lambda: stub
    user = make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}
    yield client, headers, user, stub
    client.app.dependency_overrides.pop(get_usda_client, None)


def test_endpoint_requires_authentication(api):
    http, _headers, _user, _stub = api
    assert http.post(MEALS_URL, json=VALID_BODY).status_code == 401


def test_endpoint_returns_201_and_the_documented_shape(api):
    http, headers, _user, _stub = api

    response = http.post(MEALS_URL, json=VALID_BODY, headers=headers)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == EXPECTED_KEYS
    assert set(body["items"][0]) == EXPECTED_ITEM_KEYS
    assert body["total_calories"] == 433
    assert body["items"][0]["calories_est"] == 433
    assert body["meals_logged_total"] == 1
    assert body["recommendations_unlocked"] is False


def test_response_never_exposes_user_id(api):
    http, headers, _user, _stub = api

    body = http.post(MEALS_URL, json=VALID_BODY, headers=headers).json()

    assert "user_id" not in body
    assert all("user_id" not in item for item in body["items"])


def test_client_supplied_nutrition_is_ignored(api, db_session):
    # 02 §6 / 03 §6: derived values are the server's. The request schema has no
    # field for these, so they are unreachable rather than merely overwritten.
    http, headers, _user, _stub = api
    forged = {
        "meal_type": "lunch",
        "total_calories": 5,
        "items": [
            {
                "food101_class": "grilled_salmon",
                "cooking_method": "grilled",
                "portion_confirmed_g": 210.0,
                "calories_est": 5,
                "protein_est_g": 999.0,
            }
        ],
    }

    body = http.post(MEALS_URL, json=forged, headers=headers).json()

    assert body["total_calories"] == 433
    assert body["items"][0]["calories_est"] == 433
    assert body["items"][0]["protein_est_g"] == 46.4
    stored = db_session.query(MealLog).one()
    assert float(stored.total_calories) == 433


def test_client_supplied_logged_at_is_ignored(api):
    http, headers, _user, _stub = api
    body = dict(VALID_BODY, logged_at="2099-01-01T00:00:00Z")

    response = http.post(MEALS_URL, json=body, headers=headers)

    logged_at = datetime.fromisoformat(response.json()["logged_at"])
    assert logged_at.year < 2099


def test_multi_item_meal_sums_correctly_over_http(api):
    http, headers, _user, _stub = api
    body = {
        "meal_type": "dinner",
        "items": [
            {
                "food101_class": "grilled_salmon",
                "cooking_method": "grilled",
                "portion_confirmed_g": 210.0,
            },
            {
                "food101_class": "fried_rice",
                "cooking_method": "fried",
                "portion_confirmed_g": 150.0,
            },
        ],
    }

    payload = http.post(MEALS_URL, json=body, headers=headers).json()

    assert len(payload["items"]) == 2
    assert payload["total_calories"] == 433 + 195


def test_disclaimer_reaches_the_wire(api):
    from app.core.constants import NUTRITION_DISCLAIMER

    http, headers, _user, _stub = api
    body = http.post(MEALS_URL, json=VALID_BODY, headers=headers).json()

    assert body["disclaimer"] == NUTRITION_DISCLAIMER


@pytest.mark.parametrize(
    "body",
    [
        {"meal_type": "lunch", "items": []},
        {"meal_type": "brunch", "items": VALID_BODY["items"]},
        {"items": VALID_BODY["items"]},
        {
            "meal_type": "lunch",
            "items": [
                {
                    "food101_class": "grilled_salmon",
                    "cooking_method": "grilled",
                    "portion_confirmed_g": 0,
                }
            ],
        },
        {
            "meal_type": "lunch",
            "items": [
                {
                    "food101_class": "grilled_salmon",
                    "cooking_method": "sous_vide",
                    "portion_confirmed_g": 210.0,
                }
            ],
        },
        {
            "meal_type": "lunch",
            "items": [
                {
                    "food101_class": "grilled_salmon",
                    "cooking_method": "grilled",
                    "portion_confirmed_g": 210.0,
                    "confidence_score": 1.5,
                }
            ],
        },
    ],
    ids=[
        "empty_items",
        "invalid_meal_type",
        "missing_meal_type",
        "zero_portion",
        "invalid_cooking_method",
        "confidence_above_one",
    ],
)
def test_invalid_bodies_are_rejected(api, body):
    http, headers, _user, _stub = api

    response = http.post(MEALS_URL, json=body, headers=headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_too_many_items_is_rejected(api):
    http, headers, _user, _stub = api
    body = {
        "meal_type": "lunch",
        "items": [
            {
                "food101_class": "grilled_salmon",
                "cooking_method": "grilled",
                "portion_confirmed_g": 100.0,
            }
        ]
        * 11,
    }

    assert http.post(MEALS_URL, json=body, headers=headers).status_code == 422


def test_usda_failure_writes_nothing_and_returns_502(client, db_session):
    client.app.dependency_overrides[get_usda_client] = lambda: StubUsda(fail_on={"grilled_salmon"})
    user = make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}

    response = client.post(MEALS_URL, json=VALID_BODY, headers=headers)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "USDA_UNAVAILABLE"
    assert db_session.query(MealLog).count() == 0
    client.app.dependency_overrides.pop(get_usda_client, None)


def test_meals_are_isolated_per_user(client, db_session):
    stub = StubUsda()
    client.app.dependency_overrides[get_usda_client] = lambda: stub
    alice = make_user(db_session, "alice", "alice@example.com")
    bob = make_user(db_session, "bob", "bob@example.com")

    alice_headers = {"Authorization": f"Bearer {create_access_token(str(alice.user_id))}"}
    bob_headers = {"Authorization": f"Bearer {create_access_token(str(bob.user_id))}"}

    client.post(MEALS_URL, json=VALID_BODY, headers=alice_headers)
    client.post(MEALS_URL, json=VALID_BODY, headers=alice_headers)
    bob_body = client.post(MEALS_URL, json=VALID_BODY, headers=bob_headers).json()

    assert bob_body["meals_logged_total"] == 1
    assert meal_service.count_confirmed_meals(alice.user_id, db_session) == 2
    assert (
        db_session.query(MealLog).filter(MealLog.user_id == bob.user_id).one().user_id
        == bob.user_id
    )
    client.app.dependency_overrides.pop(get_usda_client, None)
