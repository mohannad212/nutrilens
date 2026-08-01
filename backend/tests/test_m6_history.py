"""
tests/test_m6_history.py
===========================

WHY THIS FILE EXISTS
---------------------
M6's Definition of Done is "history shows logged meals newest-first with correct
calories". Two of those three claims are orderings and filters that a passing
response cannot demonstrate on its own — a single-meal history looks identical
whether or not the sort, the confirmed-only filter and the ownership check exist.
These tests are what make them observable.

The ownership tests are the important ones. `GET /meals` takes no user id, so the
isolation is structural rather than checked — but "structural" is a claim about
code that has to be verified at the layer an attacker would actually use, which is
HTTP with a real token.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Newest first, with a tiebreak.** Meals are seeded out of order so a naive
  "insertion order" implementation fails. A separate test logs two meals at the
  *same* timestamp, because `ORDER BY logged_at DESC` alone leaves those two in an
  unspecified order — which makes pagination silently drop or repeat a row
  between pages, the kind of bug that only shows up once a user has more than one
  page of history.

- **Unconfirmed meals are excluded.** `02 §6` calls an unconfirmed row a draft and
  excludes it from history. NutriLens never writes one today, so this asserts the
  filter rather than the current behaviour — history is exactly where a stray
  draft would surface.

- **`total` counts matches, not the page.** The History screen decides whether to
  keep paging from this number. If it were the page length, "20 returned" would be
  indistinguishable from "20 exist" and the list would stop at the first full page.

- **Date filters use the contract's wire names.** `from` is a Python keyword, so
  the parameter is aliased. A test that passed `date_from` would exercise the
  Python identifier and prove nothing about the `?from=` the app actually sends.

- **403 vs 404 on a single meal.** `03 §1`'s distinction, asserted separately
  because they come from different branches and collapsing them (both 404, or both
  403) is an easy simplification to make by accident.

- **Delete removes the items but not the shared food.** `02 §1`'s cascade policy in
  one test: the owned chain goes, the shared USDA cache row stays. A cascade that
  reached `food_item` would delete nutrition data other users' history depends on.

- **Per-user isolation on all three routes.** One user must not list, read or
  delete another's meals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.db.models import FoodItem, HealthGoal, MealItem, MealLog, MealType, User
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


class StubUsda:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def search(self, food101_class: str, cooking_method: str | None = None) -> UsdaFood:
        self.calls.append((food101_class, cooking_method))
        return SALMON

    def close(self) -> None:  # pragma: no cover
        pass


def make_user(db, username: str = "hist", email: str = "hist@example.com") -> User:
    user = User(
        username=username,
        email=email,
        password_hash=hash_password("Str0ngPass!"),
        health_goal=HealthGoal.maintenance,
    )
    db.add(user)
    db.commit()
    return user


def item(portion: float = 100.0) -> MealItemInput:
    return MealItemInput(
        food101_class="grilled_salmon", cooking_method="grilled", portion_confirmed_g=portion
    )


def seed_meal(
    db,
    user: User,
    *,
    days_ago: float = 0,
    meal_type: MealType = MealType.lunch,
    portion: float = 100.0,
) -> MealLog:
    when = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days_ago)
    return meal_service.create_meal(
        user, meal_type, [item(portion)], db, StubUsda(), logged_at=when
    )


@pytest.fixture()
def api(client, db_session):
    client.app.dependency_overrides[get_usda_client] = lambda: StubUsda()
    user = make_user(db_session)
    headers = {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}
    yield client, headers, user
    client.app.dependency_overrides.pop(get_usda_client, None)


# ---------------------------------------------------------------------------
# Ordering (M6 DoD: "newest-first")
# ---------------------------------------------------------------------------
def test_history_is_newest_first(api, db_session):
    http, headers, user = api
    # Seeded out of chronological order, so insertion order is not the answer.
    seed_meal(db_session, user, days_ago=3, meal_type=MealType.breakfast)
    seed_meal(db_session, user, days_ago=1, meal_type=MealType.dinner)
    seed_meal(db_session, user, days_ago=5, meal_type=MealType.snack)

    body = http.get(MEALS_URL, headers=headers).json()

    stamps = [m["logged_at"] for m in body["meals"]]
    assert stamps == sorted(stamps, reverse=True)
    assert body["meals"][0]["meal_type"] == "dinner"
    assert body["meals"][-1]["meal_type"] == "snack"


def test_identical_timestamps_have_a_stable_order(api, db_session):
    # ORDER BY logged_at DESC alone leaves same-instant rows unordered, which makes
    # pagination drop or repeat rows between pages.
    http, headers, user = api
    same = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(4):
        meal_service.create_meal(
            user, MealType.lunch, [item()], db_session, StubUsda(), logged_at=same
        )

    first = [m["log_id"] for m in http.get(MEALS_URL, headers=headers).json()["meals"]]
    second = [m["log_id"] for m in http.get(MEALS_URL, headers=headers).json()["meals"]]

    assert first == second
    assert first == sorted(first, reverse=True)


# ---------------------------------------------------------------------------
# Filtering (02 §6, 03 §6)
# ---------------------------------------------------------------------------
def test_unconfirmed_meals_are_excluded(api, db_session):
    # 02 §6: an unconfirmed meal is a draft, excluded from history.
    http, headers, user = api
    seed_meal(db_session, user)
    db_session.add(
        MealLog(
            user_id=user.user_id,
            meal_type=MealType.snack,
            total_calories=999,
            user_confirmed=False,
        )
    )
    db_session.commit()

    body = http.get(MEALS_URL, headers=headers).json()

    assert body["total"] == 1
    assert all(m["total_calories"] != 999 for m in body["meals"])


def test_date_filters_use_the_contract_wire_names(api, db_session):
    # `?from=`/`?to=` per 03 §6 — the aliases, not the Python identifiers.
    http, headers, user = api
    seed_meal(db_session, user, days_ago=10)
    seed_meal(db_session, user, days_ago=1)
    cutoff = (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)).isoformat()

    body = http.get(f"{MEALS_URL}?from={cutoff}", headers=headers).json()

    assert body["total"] == 1


def test_to_filter_bounds_the_other_end(api, db_session):
    http, headers, user = api
    seed_meal(db_session, user, days_ago=10)
    seed_meal(db_session, user, days_ago=1)
    cutoff = (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)).isoformat()

    body = http.get(f"{MEALS_URL}?to={cutoff}", headers=headers).json()

    assert body["total"] == 1


# ---------------------------------------------------------------------------
# Pagination (03 §6)
# ---------------------------------------------------------------------------
def test_total_counts_matches_not_the_page(api, db_session):
    http, headers, user = api
    for day in range(5):
        seed_meal(db_session, user, days_ago=day)

    body = http.get(f"{MEALS_URL}?limit=2", headers=headers).json()

    assert len(body["meals"]) == 2
    assert body["total"] == 5  # not 2 — the client needs this to keep paging
    assert body["limit"] == 2
    assert body["offset"] == 0


def test_paging_covers_every_meal_exactly_once(api, db_session):
    http, headers, user = api
    for day in range(5):
        seed_meal(db_session, user, days_ago=day)

    page1 = http.get(f"{MEALS_URL}?limit=2&offset=0", headers=headers).json()["meals"]
    page2 = http.get(f"{MEALS_URL}?limit=2&offset=2", headers=headers).json()["meals"]
    page3 = http.get(f"{MEALS_URL}?limit=2&offset=4", headers=headers).json()["meals"]

    ids = [m["log_id"] for m in page1 + page2 + page3]
    assert len(ids) == 5
    assert len(set(ids)) == 5  # no repeats, no drops


@pytest.mark.parametrize(
    "query",
    ["?limit=0", "?limit=101", "?offset=-1", "?limit=abc"],
    ids=["zero", "over_max", "negative_offset", "non_numeric"],
)
def test_invalid_pagination_is_rejected(api, query):
    http, headers, _user = api
    assert http.get(f"{MEALS_URL}{query}", headers=headers).status_code == 422


# ---------------------------------------------------------------------------
# Single meal (03 §6, §1)
# ---------------------------------------------------------------------------
def test_get_single_meal_returns_its_items(api, db_session):
    http, headers, user = api
    meal = seed_meal(db_session, user, portion=210.0)

    body = http.get(f"{MEALS_URL}/{meal.log_id}", headers=headers).json()

    assert body["log_id"] == meal.log_id
    assert body["item_count"] == 1
    assert body["items"][0]["calories_est"] == 433
    assert body["total_calories"] == 433


def test_missing_meal_is_404(api):
    http, headers, _user = api
    response = http.get(f"{MEALS_URL}/99999", headers=headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_another_users_meal_is_403_not_404(api, db_session):
    # 03 §1's split: 404 means "no such meal", 403 means "not yours".
    http, headers, _user = api
    other = make_user(db_session, "other", "other@example.com")
    theirs = seed_meal(db_session, other)

    response = http.get(f"{MEALS_URL}/{theirs.log_id}", headers=headers)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# Delete (03 §6, 02 §1 cascade policy)
# ---------------------------------------------------------------------------
def test_delete_removes_the_meal_and_its_items_but_not_the_shared_food(api, db_session):
    http, headers, user = api
    meal = seed_meal(db_session, user)
    assert db_session.query(MealItem).count() == 1
    food_rows = db_session.query(FoodItem).count()

    response = http.delete(f"{MEALS_URL}/{meal.log_id}", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert db_session.query(MealLog).count() == 0
    assert db_session.query(MealItem).count() == 0
    # food_item is shared reference data behind ON DELETE RESTRICT — other users'
    # history points at the same row, so it must survive.
    assert db_session.query(FoodItem).count() == food_rows


def test_delete_is_reflected_in_history(api, db_session):
    http, headers, user = api
    keep = seed_meal(db_session, user, days_ago=1)
    drop = seed_meal(db_session, user, days_ago=2)

    http.delete(f"{MEALS_URL}/{drop.log_id}", headers=headers)
    body = http.get(MEALS_URL, headers=headers).json()

    assert body["total"] == 1
    assert body["meals"][0]["log_id"] == keep.log_id


def test_cannot_delete_another_users_meal(api, db_session):
    http, headers, _user = api
    other = make_user(db_session, "other", "other@example.com")
    theirs = seed_meal(db_session, other)

    response = http.delete(f"{MEALS_URL}/{theirs.log_id}", headers=headers)

    assert response.status_code == 403
    assert db_session.query(MealLog).count() == 1  # still there


def test_deleting_a_missing_meal_is_404(api):
    http, headers, _user = api
    assert http.delete(f"{MEALS_URL}/99999", headers=headers).status_code == 404


# ---------------------------------------------------------------------------
# Auth and isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path"),
    [("get", MEALS_URL), ("get", f"{MEALS_URL}/1"), ("delete", f"{MEALS_URL}/1")],
    ids=["list", "detail", "delete"],
)
def test_every_route_requires_authentication(api, method, path):
    http, _headers, _user = api
    assert getattr(http, method)(path).status_code == 401


def test_history_never_shows_another_users_meals(api, db_session):
    http, headers, user = api
    other = make_user(db_session, "other", "other@example.com")
    seed_meal(db_session, user)
    seed_meal(db_session, other)
    seed_meal(db_session, other)

    body = http.get(MEALS_URL, headers=headers).json()

    assert body["total"] == 1
    assert len(body["meals"]) == 1


def test_response_never_exposes_user_id(api, db_session):
    http, headers, user = api
    seed_meal(db_session, user)

    body = http.get(MEALS_URL, headers=headers).json()

    assert "user_id" not in body["meals"][0]
