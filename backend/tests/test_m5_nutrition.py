"""
tests/test_m5_nutrition.py
=============================

WHY THIS FILE EXISTS
---------------------
`09` M5 names four things its tests must prove: "cache hit vs USDA miss, scaling
math, USDA-down->cache fallback, totals recomputed". Three of those four are about
this module (the fourth is `test_m5_meals.py`), and all three are about a rule that
is easy to state and easy to violate quietly: **the number shown to the user came
from USDA, or from a value USDA previously gave us, or the request failed.** There
is no fourth option.

That rule is not enforceable by reading the code alone — a stub that returns zeros
looks identical at a glance to one that returns real values. It is enforceable by
tests that count how many times USDA was called and assert what happens when it
refuses.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Cache hit does not call USDA.** Asserted by call count, not by response
  content, because a version that calls USDA and *then* returns the cached value
  would produce identical JSON while spending quota on every lookup and inheriting
  USDA's latency and downtime. The count is the only observable difference.

- **A seeded placeholder is not a cache hit.** `seed_food101_classes` creates 101
  rows with null macros (`02 §5`). If `get_cached` tested for row existence, those
  would satisfy it and the user would be shown an empty nutrition object rendered
  as zeroes. This is the single most plausible way to accidentally ship fabricated
  values, so it is tested from both directions: the placeholder is not a hit, and
  the lookup that follows claims it rather than orphaning it.

- **Scaling arithmetic, against `03 §5`'s own worked example.** The spec publishes
  206 kcal/100 g at 210 g -> 433 kcal, 22.1 -> 46.4, 12.4 -> 26.0. Using the
  document's numbers rather than invented ones means the test fails if either the
  arithmetic or the rounding policy drifts from what the contract advertises.

- **Rounding happens once, at the end.** A separate test with values chosen so
  that rounding per-100 g first gives a visibly different answer. Without it, the
  "compounding error" reasoning in `nutrition_service` is an unverified claim.

- **USDA down + stale cache -> serve the stale value.** The Reliability NFR
  (`08 §6`). Both halves matter: the response arrives, *and* it is labelled
  `source: "cache"` so the fallback is visible rather than silent.

- **USDA down + nothing cached -> the documented error.** The complement, and the
  more important of the two: this is where a well-meaning `except: return zeros`
  would live. Each upstream failure is checked against the status code `03 §5`
  promises.

- **No cross-method fallback.** Grilled salmon cached, fried salmon requested,
  USDA down -> error, not the grilled numbers. Frying adds oil; serving one for the
  other is a different food's data wearing this food's label. This test exists
  because that fallback is a tempting one-line "improvement".

- **HTTP contract.** Auth on the endpoint (it spends a shared quota), the exact
  response keys from `03 §5`, the disclaimer on the wire, and 422s for the inputs
  the schema is there to stop.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.errors import (
    NoNutritionMatchError,
    UsdaRateLimitedError,
    UsdaUnavailableError,
)
from app.db.models import FoodItem, HealthGoal, User
from app.deps import get_usda_client
from app.integrations import nutrition_cache as cache
from app.integrations.usda_client import UsdaFood
from app.security import create_access_token, hash_password
from app.services import nutrition_service

NUTRITION_URL = "/api/v1/nutrition"

# 03 §5's worked example — the numbers the contract itself publishes.
SALMON = UsdaFood(
    fdc_id="175168",
    description="Fish, salmon, Atlantic, farmed, cooked, dry heat",
    calories_per_100g=206.0,
    protein_per_100g=22.1,
    carbs_per_100g=0.0,
    fat_per_100g=12.4,
)


class StubUsda:
    """A `UsdaClient` stand-in that counts calls and can be told to fail.

    Counting is the point: several assertions below are about how many times the
    upstream was consulted, which no amount of response inspection can reveal.
    """

    def __init__(self, result: UsdaFood | None = SALMON, raises: Exception | None = None):
        self.result = result
        self.raises = raises
        self.calls: list[tuple[str, str | None]] = []

    def search(self, food101_class: str, cooking_method: str | None = None) -> UsdaFood:
        self.calls.append((food101_class, cooking_method))
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result

    def close(self) -> None:  # pragma: no cover - parity with the real client
        pass


def cached_row(
    db,
    food101_class: str = "grilled_salmon",
    cooking_method: str = "grilled",
    *,
    calories: float = 206.0,
    protein: float = 22.1,
    carbs: float = 0.0,
    fat: float = 12.4,
    age_days: int = 0,
    fdc_id: str | None = "175168",
) -> FoodItem:
    row = FoodItem(
        usda_fdc_id=fdc_id,
        food_name="Fish, salmon, Atlantic, farmed, cooked, dry heat",
        food101_class=food101_class,
        cooking_method=cooking_method,
        calories_per_100g=calories,
        protein_per_100g=protein,
        carbs_per_100g=carbs,
        fat_per_100g=fat,
        cached_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=age_days),
    )
    db.add(row)
    db.commit()
    return row


def placeholder_row(db, food101_class: str = "grilled_salmon") -> FoodItem:
    """The shape `seed_food101_classes` inserts: class known, macros unknown."""
    row = FoodItem(
        usda_fdc_id=None,
        food_name=food101_class.replace("_", " ").title(),
        food101_class=food101_class,
        cooking_method=None,
        calories_per_100g=None,
        protein_per_100g=None,
        carbs_per_100g=None,
        fat_per_100g=None,
    )
    db.add(row)
    db.commit()
    return row


# ---------------------------------------------------------------------------
# Cache hit vs USDA miss (09 M5)
# ---------------------------------------------------------------------------
def test_cache_hit_does_not_call_usda(db_session):
    cached_row(db_session)
    usda = StubUsda()

    result = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 210.0, usda)

    assert result.source == "cache"
    # The assertion that matters: the upstream was never consulted.
    assert usda.calls == []


def test_cache_miss_calls_usda_and_caches_the_result(db_session):
    usda = StubUsda()

    first = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 210.0, usda)
    db_session.commit()

    assert first.source == "usda"
    assert usda.calls == [("grilled_salmon", "grilled")]

    # A second identical lookup must now be served locally.
    second = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 210.0, usda)
    assert second.source == "cache"
    assert len(usda.calls) == 1


def test_cooking_method_is_part_of_the_cache_key(db_session):
    # Grilled and fried are different foods nutritionally; one must not serve the
    # other from cache.
    cached_row(db_session, cooking_method="grilled")
    usda = StubUsda()

    result = nutrition_service.get_nutrition(db_session, "grilled_salmon", "fried", 210.0, usda)

    assert result.source == "usda"
    assert usda.calls == [("grilled_salmon", "fried")]


def test_seeded_placeholder_is_not_a_cache_hit(db_session):
    # 02 §5's placeholder rows exist but hold no macros. Treating one as a hit
    # would render zeroes to the user as though they were measured values.
    placeholder_row(db_session)
    usda = StubUsda()

    result = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, usda)

    assert result.source == "usda"
    assert result.scaled.calories == 206


def test_lookup_claims_the_seeded_placeholder_instead_of_orphaning_it(db_session):
    placeholder = placeholder_row(db_session)
    placeholder_id = placeholder.food_id

    result = nutrition_service.get_nutrition(
        db_session, "grilled_salmon", "grilled", 100.0, StubUsda()
    )
    db_session.commit()

    assert result.food_id == placeholder_id
    assert db_session.query(FoodItem).count() == 1


def test_second_cooking_method_inserts_a_new_row(db_session):
    placeholder_row(db_session)

    nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, StubUsda())
    db_session.commit()
    nutrition_service.get_nutrition(db_session, "grilled_salmon", "baked", 100.0, StubUsda())
    db_session.commit()

    rows = db_session.query(FoodItem).all()
    assert {r.cooking_method for r in rows} == {"grilled", "baked"}


def test_duplicate_fdc_id_is_dropped_rather_than_violating_the_unique_index(db_session):
    # Two cooking methods can resolve to the same USDA food. food_item.usda_fdc_id
    # is UNIQUE (02 §2.3), so the second row keeps the macros and loses only the
    # provenance id — see integrations/nutrition_cache.py.
    nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, StubUsda())
    db_session.commit()
    nutrition_service.get_nutrition(db_session, "grilled_salmon", "baked", 100.0, StubUsda())
    db_session.commit()

    by_method = {r.cooking_method: r for r in db_session.query(FoodItem).all()}
    assert by_method["grilled"].usda_fdc_id == "175168"
    assert by_method["baked"].usda_fdc_id is None
    # The macros — the part the user sees — survive intact on both.
    assert float(by_method["baked"].calories_per_100g) == 206.0


# ---------------------------------------------------------------------------
# Scaling math (09 M5)
# ---------------------------------------------------------------------------
def test_scaling_matches_the_contracts_worked_example(db_session):
    # 03 §5: 206 kcal/100 g at 210 g -> 433 kcal, 22.1 -> 46.4, 12.4 -> 26.0.
    cached_row(db_session)

    result = nutrition_service.get_nutrition(
        db_session, "grilled_salmon", "grilled", 210.0, StubUsda()
    )

    assert result.per_100g.calories == 206
    assert result.scaled.calories == 433
    assert result.scaled.protein_g == 46.4
    assert result.scaled.carbs_g == 0.0
    assert result.scaled.fat_g == 26.0


def test_scaling_is_linear_in_mass(db_session):
    cached_row(db_session)
    usda = StubUsda()

    half = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 50.0, usda)
    whole = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, usda)
    double = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 200.0, usda)

    assert half.scaled.calories == 103
    assert whole.scaled.calories == 206
    assert double.scaled.calories == 412


def test_rounding_happens_once_at_the_end_not_per_100g(db_session):
    # 10.44 g protein/100 g at 1000 g is 104.4 g. Rounding the per-100 g value to
    # 10.4 first and then scaling would give 104.0 — visibly wrong, and the error
    # grows with the portion.
    cached_row(db_session, protein=10.44)

    result = nutrition_service.get_nutrition(
        db_session, "grilled_salmon", "grilled", 1000.0, StubUsda()
    )

    assert result.per_100g.protein_g == 10.4
    assert result.scaled.protein_g == 104.4


def test_scaling_reads_decimal_columns_without_type_errors(db_session):
    # NUMERIC(7,2) comes back as decimal.Decimal; multiplying it by a float
    # portion raises TypeError unless coerced. Only reproducible via a real
    # round-trip through the database, which is why the row is re-fetched.
    cached_row(db_session)
    db_session.expire_all()

    result = nutrition_service.get_nutrition(
        db_session, "grilled_salmon", "grilled", 210.0, StubUsda()
    )
    assert result.scaled.calories == 433


# ---------------------------------------------------------------------------
# USDA down -> cache fallback (09 M5, 08 §6)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure",
    [UsdaUnavailableError(), UsdaRateLimitedError()],
    ids=["usda_down", "rate_limited"],
)
def test_stale_cache_is_served_when_usda_is_unreachable(db_session, failure):
    # The Reliability NFR: a 200-day-old lab value beats "you cannot log dinner".
    cached_row(db_session, age_days=400)
    usda = StubUsda(raises=failure)

    result = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 210.0, usda)

    assert result.scaled.calories == 433
    # Labelled honestly, so the fallback is observable rather than silent.
    assert result.source == "cache"
    assert len(usda.calls) == 1  # it did try to refresh first


def test_stale_cache_is_refreshed_when_usda_is_reachable(db_session):
    cached_row(db_session, calories=100.0, age_days=400)
    usda = StubUsda()

    result = nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, usda)

    assert result.source == "usda"
    assert result.scaled.calories == 206


def test_fresh_cache_is_not_refreshed(db_session):
    cached_row(db_session, age_days=1)
    usda = StubUsda()

    nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 100.0, usda)

    assert usda.calls == []


@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (UsdaUnavailableError(), 502),
        (UsdaRateLimitedError(), 429),
        (NoNutritionMatchError(), 404),
    ],
    ids=["unavailable", "rate_limited", "no_match"],
)
def test_usda_failure_with_no_cache_raises_the_documented_error(db_session, failure, status):
    # The complement of the fallback test, and the more important one: with
    # nothing to fall back to, the request must fail rather than invent a number.
    usda = StubUsda(raises=failure)

    with pytest.raises(type(failure)) as exc:
        nutrition_service.get_nutrition(db_session, "grilled_salmon", "grilled", 210.0, usda)

    assert exc.value.status_code == status


def test_no_fallback_to_a_different_cooking_method(db_session):
    # Grilled salmon is cached; fried salmon is requested; USDA is down. Serving
    # the grilled numbers would be a different food's data under this food's name.
    cached_row(db_session, cooking_method="grilled")
    usda = StubUsda(raises=UsdaUnavailableError())

    with pytest.raises(UsdaUnavailableError):
        nutrition_service.get_nutrition(db_session, "grilled_salmon", "fried", 210.0, usda)


def test_no_match_is_not_cached_as_an_empty_row(db_session):
    usda = StubUsda(raises=NoNutritionMatchError())

    with pytest.raises(NoNutritionMatchError):
        nutrition_service.get_nutrition(db_session, "takoyaki", "fried", 100.0, usda)

    assert db_session.query(FoodItem).count() == 0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def test_is_stale_respects_the_ttl(db_session):
    # Distinct fdc ids: these rows are inserted directly, bypassing the `upsert`
    # path that normally resolves the UNIQUE-vs-cache-key tension.
    fresh = cached_row(db_session, cooking_method="grilled", age_days=10, fdc_id="1")
    old = cached_row(db_session, cooking_method="baked", age_days=400, fdc_id="2")

    assert cache.is_stale(fresh, 180) is False
    assert cache.is_stale(old, 180) is True
    # A TTL of 0 disables expiry entirely.
    assert cache.is_stale(old, 0) is False


def test_has_complete_macros_rejects_a_partial_row(db_session):
    partial = cached_row(db_session, fat=0.0)
    partial.fat_per_100g = None

    assert cache.has_complete_macros(partial) is False


# ---------------------------------------------------------------------------
# HTTP contract (03 §5)
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {
    "food_id",
    "food_name",
    "usda_fdc_id",
    "cooking_method",
    "per_100g",
    "portion_g",
    "scaled",
    "source",
    "disclaimer",
}
EXPECTED_MACRO_KEYS = {"calories", "protein_g", "carbs_g", "fat_g"}


@pytest.fixture()
def auth_headers(db_session):
    user = User(
        username="nutri",
        email="nutri@example.com",
        password_hash=hash_password("Str0ngPass!"),
        health_goal=HealthGoal.maintenance,
    )
    db_session.add(user)
    db_session.commit()
    return {"Authorization": f"Bearer {create_access_token(str(user.user_id))}"}


@pytest.fixture()
def usda_client(client):
    """Wire a `StubUsda` into the app so no test touches the network."""
    stub = StubUsda()
    client.app.dependency_overrides[get_usda_client] = lambda: stub
    yield client, stub
    client.app.dependency_overrides.pop(get_usda_client, None)


def test_endpoint_requires_authentication(usda_client):
    http, _ = usda_client
    response = http.post(
        NUTRITION_URL,
        json={"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 210.0},
    )
    # An endpoint that spends a shared API quota must not be open.
    assert response.status_code == 401


def test_endpoint_returns_the_documented_shape(usda_client, auth_headers):
    http, _ = usda_client
    response = http.post(
        NUTRITION_URL,
        json={"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 210.0},
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == EXPECTED_KEYS
    assert set(body["per_100g"]) == EXPECTED_MACRO_KEYS
    assert set(body["scaled"]) == EXPECTED_MACRO_KEYS
    assert body["scaled"]["calories"] == 433
    assert body["source"] == "usda"


def test_disclaimer_reaches_the_wire(usda_client, auth_headers):
    from app.core.constants import NUTRITION_DISCLAIMER

    http, _ = usda_client
    response = http.post(
        NUTRITION_URL,
        json={"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 210.0},
        headers=auth_headers,
    )
    assert response.json()["disclaimer"] == NUTRITION_DISCLAIMER


def test_endpoint_persists_the_cache_row(usda_client, auth_headers, db_session):
    http, stub = usda_client
    payload = {"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 210.0}

    http.post(NUTRITION_URL, json=payload, headers=auth_headers)
    second = http.post(NUTRITION_URL, json=payload, headers=auth_headers)

    # The first call committed the row, so the second is served from cache.
    assert second.json()["source"] == "cache"
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"food101_class": "grilled_salmon", "cooking_method": "sous_vide", "portion_g": 210.0},
        {"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 0},
        {"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": -50},
        {"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 99999},
        {"food101_class": "", "cooking_method": "grilled", "portion_g": 210.0},
        {"food101_class": "Grilled Salmon!", "cooking_method": "grilled", "portion_g": 210.0},
        {"cooking_method": "grilled", "portion_g": 210.0},
    ],
    ids=[
        "unknown_method",
        "zero_portion",
        "negative_portion",
        "implausible_portion",
        "empty_class",
        "malformed_class",
        "missing_class",
    ],
)
def test_invalid_requests_are_rejected_at_the_edge(usda_client, auth_headers, payload):
    http, stub = usda_client
    response = http.post(NUTRITION_URL, json=payload, headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    # Rejected before any quota was spent.
    assert stub.calls == []


def test_usda_failure_surfaces_as_the_documented_status(client, auth_headers):
    client.app.dependency_overrides[get_usda_client] = lambda: StubUsda(
        raises=UsdaUnavailableError()
    )
    response = client.post(
        NUTRITION_URL,
        json={"food101_class": "grilled_salmon", "cooking_method": "grilled", "portion_g": 210.0},
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "USDA_UNAVAILABLE"
    client.app.dependency_overrides.pop(get_usda_client, None)


def test_cooking_methods_meta_endpoint_matches_the_enum(client):
    # 03 §8: the app's picker is populated from here, so it must agree with the
    # validator exactly — a mismatch is a 422 on the most recently added option.
    from app.core.enums import COOKING_METHODS

    response = client.get("/api/v1/meta/cooking-methods")

    assert response.status_code == 200
    assert response.json()["cooking_methods"] == COOKING_METHODS
    assert COOKING_METHODS == [
        "grilled",
        "fried",
        "boiled",
        "baked",
        "roasted",
        "steamed",
        "raw",
        "other",
    ]
