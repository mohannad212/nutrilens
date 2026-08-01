"""
tests/test_m5_usda_client.py
===============================

WHY THIS FILE EXISTS
---------------------
`usda_client` is the only module in NutriLens that depends on a service we do not
control, and `08 §6`'s central rule — never fabricate a nutrition value — is
enforced almost entirely by *how this module fails*. These tests pin that
behaviour: which upstream condition becomes which typed error, and what the client
does with a response that is well-formed but incomplete.

They run with no network and no API key. `httpx.MockTransport` intercepts the
request inside the real `httpx.Client`, so the code under test is the production
path — real URL building, real parameter encoding, real status handling — with
only the socket replaced.

WHAT EACH GROUP OF TESTS VALIDATES AND WHY IT MATTERS
--------------------------------------------------------
- **Query construction (`build_query`).** The one piece of pure logic with real
  branches. `08 §5`'s worked example is asserted verbatim because it is the
  contract's own illustration of what the mapping should produce.

- **Both nutrient ID schemes.** USDA's search and detail endpoints disagree about
  where a nutrient's identity lives. A parser that handles only the shape it was
  written against reports "no match" for a food that came back perfectly well —
  a silent, confusing 404. Both shapes are tested with the same food so the two
  paths must agree on the answer.

- **Data-type preference.** Asserted with a Branded hit deliberately placed
  *first* in the response, because that is the case where honouring USDA's own
  ordering would pick the manufacturer-reported values over the lab-analysed
  ones. If this test passes only because the fixture happened to be in the right
  order, it proves nothing.

- **Incomplete candidates are skipped, then refused.** The behavioural expression
  of "a missing nutrient is not zero". One test proves the client falls through
  to a complete candidate; the other proves that when nothing is complete it
  raises rather than filling gaps with `0.0`. The second is the one that would
  catch a well-meaning `.get(key, 0)` creeping in later.

- **Every upstream failure maps to its documented error.** Timeout, transport
  failure, 5xx, 429, and a rejected API key each become the status code `03 §5`
  promises. Tested one by one because they are independent `if` branches, and a
  wrong mapping here is invisible until the API is genuinely down.

- **The local rate limiter is consulted before the request.** Asserted by
  checking that no HTTP call was attempted at all — the point of a local budget
  is to *not spend* it, so a version that called USDA and then raised would pass a
  weaker test while defeating the purpose.

- **Fixed-window limiter mechanics.** Including the reset after an hour, driven
  by an injected clock rather than a sleep.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.errors import (
    NoNutritionMatchError,
    UsdaRateLimitedError,
    UsdaUnavailableError,
)
from app.core.rate_limit import HourlyRateLimiter
from app.integrations.usda_client import UsdaClient, select_best
from app.integrations.usda_query_map import build_query

BASE_URL = "https://api.nal.usda.gov/fdc/v1"


# ---------------------------------------------------------------------------
# Helpers — USDA response fixtures in both nutrient shapes
# ---------------------------------------------------------------------------
def abridged_nutrients(
    kcal: float = 206.0, protein: float = 22.1, carbs: float = 0.0, fat: float = 12.4
) -> list[dict[str, object]]:
    """The shape `/foods/search` returns: identity flattened onto each entry."""
    return [
        {"nutrientId": 1008, "nutrientNumber": "208", "unitName": "KCAL", "value": kcal},
        {"nutrientId": 1003, "nutrientNumber": "203", "unitName": "G", "value": protein},
        {"nutrientId": 1005, "nutrientNumber": "205", "unitName": "G", "value": carbs},
        {"nutrientId": 1004, "nutrientNumber": "204", "unitName": "G", "value": fat},
    ]


def full_nutrients(
    kcal: float = 206.0, protein: float = 22.1, carbs: float = 0.0, fat: float = 12.4
) -> list[dict[str, object]]:
    """The shape `/food/{id}` returns: identity nested, value under `amount`."""
    return [
        {"nutrient": {"id": 1008, "number": "208", "unitName": "kcal"}, "amount": kcal},
        {"nutrient": {"id": 1003, "number": "203", "unitName": "g"}, "amount": protein},
        {"nutrient": {"id": 1005, "number": "205", "unitName": "g"}, "amount": carbs},
        {"nutrient": {"id": 1004, "number": "204", "unitName": "g"}, "amount": fat},
    ]


def food(
    fdc_id: int = 175168,
    description: str = "Fish, salmon, Atlantic, farmed, cooked, dry heat",
    data_type: str = "SR Legacy",
    nutrients: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "fdcId": fdc_id,
        "description": description,
        "dataType": data_type,
        "foodNutrients": abridged_nutrients() if nutrients is None else nutrients,
    }


def client_returning(
    payload: object, status: int = 200, captured: list[httpx.Request] | None = None
) -> UsdaClient:
    """A real `UsdaClient` whose socket is replaced by a canned response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=payload)

    transport = httpx.MockTransport(handler)
    return UsdaClient("test-key", BASE_URL, client=httpx.Client(transport=transport))


def client_raising(exc: Exception) -> UsdaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    transport = httpx.MockTransport(handler)
    return UsdaClient("test-key", BASE_URL, client=httpx.Client(transport=transport))


# ---------------------------------------------------------------------------
# Query construction (08 §5)
# ---------------------------------------------------------------------------
def test_build_query_matches_the_spec_example():
    # 08 §5 states this mapping explicitly; it is the contract's own illustration.
    assert build_query("grilled_salmon", "grilled") == "salmon cooked grilled"


def test_build_query_strips_the_class_adjective_so_the_user_method_wins():
    # FR06 makes the user the authority: a salmon they baked must not be searched
    # as grilled just because Food-101 named the class that way.
    assert build_query("grilled_salmon", "baked") == "salmon cooked baked"


def test_build_query_uses_the_override_table_for_dishes_usda_lacks():
    assert build_query("beignets", "fried") == "doughnut cooked fried"
    assert build_query("french_onion_soup", "boiled") == "soup onion cooked boiled"


def test_build_query_replaces_underscores_for_unmapped_classes():
    assert build_query("chicken_curry", "other") == "chicken curry"


def test_build_query_deduplicates_repeated_tokens():
    # `sashimi` -> "fish" + raw -> "fish raw"; no doubled term.
    assert build_query("sashimi", "raw") == "fish raw"
    # `onion_rings` override already contains "breaded"; fried adds cooked+fried.
    assert build_query("fried_rice", "fried") == "rice cooked fried"


def test_build_query_tolerates_a_missing_cooking_method():
    assert build_query("pizza", None) == "pizza"
    assert build_query("pizza", "other") == "pizza"


# ---------------------------------------------------------------------------
# Nutrient parsing — both ID schemes
# ---------------------------------------------------------------------------
def test_parses_the_abridged_search_shape():
    result = client_returning({"foods": [food()]}).search("grilled_salmon", "grilled")

    assert result.fdc_id == "175168"
    assert result.description.startswith("Fish, salmon")
    assert result.calories_per_100g == 206.0
    assert result.protein_per_100g == 22.1
    assert result.carbs_per_100g == 0.0
    assert result.fat_per_100g == 12.4


def test_parses_the_full_detail_shape_identically():
    result = client_returning({"foods": [food(nutrients=full_nutrients())]}).search(
        "grilled_salmon", "grilled"
    )

    assert result.calories_per_100g == 206.0
    assert result.protein_per_100g == 22.1
    assert result.fat_per_100g == 12.4


def test_converts_kilojoules_when_kcal_is_absent():
    # 1000 kJ / 4.184 = 239.0 kcal. A unit conversion of a measured value, not an
    # invented one — and better than 404-ing a food USDA fully describes.
    kj_only: list[dict[str, object]] = [
        {"nutrientId": 1062, "nutrientNumber": "268", "unitName": "kJ", "value": 1000.0},
        {"nutrientId": 1003, "value": 10.0},
        {"nutrientId": 1005, "value": 5.0},
        {"nutrientId": 1004, "value": 2.0},
    ]
    result = client_returning({"foods": [food(nutrients=kj_only)]}).search("pizza", "baked")

    assert result.calories_per_100g == pytest.approx(1000.0 / 4.184)


def test_prefers_kcal_over_kilojoules_when_both_are_present():
    both: list[dict[str, object]] = [
        {"nutrientId": 1062, "value": 9999.0},
        *abridged_nutrients(kcal=206.0),
    ]
    result = client_returning({"foods": [food(nutrients=both)]}).search("pizza", "baked")

    assert result.calories_per_100g == 206.0


# ---------------------------------------------------------------------------
# Match selection (08 §5)
# ---------------------------------------------------------------------------
def test_prefers_lab_verified_data_types_over_branded():
    # Branded is placed FIRST, so passing this test requires actually re-ranking
    # rather than trusting USDA's relevance order.
    payload = {
        "foods": [
            food(fdc_id=1, description="Salmon, branded", data_type="Branded"),
            food(fdc_id=2, description="Fish, salmon, raw", data_type="Foundation"),
        ]
    }
    assert client_returning(payload).search("grilled_salmon", "grilled").fdc_id == "2"


def test_prefers_foundation_over_sr_legacy():
    payload = {
        "foods": [
            food(fdc_id=1, data_type="SR Legacy"),
            food(fdc_id=2, data_type="Foundation"),
        ]
    }
    assert select_best(payload["foods"]).fdc_id == "2"


def test_keeps_usda_relevance_order_within_one_data_type():
    payload = {
        "foods": [
            food(fdc_id=1, data_type="Foundation"),
            food(fdc_id=2, data_type="Foundation"),
        ]
    }
    assert select_best(payload["foods"]).fdc_id == "1"


def test_skips_a_candidate_missing_a_macro_and_uses_the_next():
    incomplete = food(fdc_id=1, data_type="Foundation")
    # Drop fat entirely — absent, not zero.
    incomplete["foodNutrients"] = abridged_nutrients()[:3]
    payload = {"foods": [incomplete, food(fdc_id=2, data_type="Foundation")]}

    assert client_returning(payload).search("grilled_salmon", "grilled").fdc_id == "2"


def test_refuses_rather_than_defaulting_a_missing_macro_to_zero():
    # 08 §6's core rule. If this ever regresses to `.get(key, 0)`, the user is
    # shown a fabricated 0 g of fat as though it were measured.
    incomplete = food(fdc_id=1, data_type="Foundation")
    incomplete["foodNutrients"] = abridged_nutrients()[:3]

    with pytest.raises(NoNutritionMatchError):
        client_returning({"foods": [incomplete]}).search("grilled_salmon", "grilled")


def test_no_results_raises_no_nutrition_match():
    with pytest.raises(NoNutritionMatchError):
        client_returning({"foods": []}).search("takoyaki", "fried")


# ---------------------------------------------------------------------------
# Failure mapping (03 §5, 08 §5)
# ---------------------------------------------------------------------------
def test_timeout_becomes_usda_unavailable():
    with pytest.raises(UsdaUnavailableError) as exc:
        client_raising(httpx.TimeoutException("timed out")).search("pizza", "baked")
    assert exc.value.status_code == 502
    assert exc.value.code == "USDA_UNAVAILABLE"


def test_transport_error_becomes_usda_unavailable():
    with pytest.raises(UsdaUnavailableError):
        client_raising(httpx.ConnectError("no route to host")).search("pizza", "baked")


def test_server_error_becomes_usda_unavailable():
    with pytest.raises(UsdaUnavailableError):
        client_returning({}, status=503).search("pizza", "baked")


def test_upstream_429_becomes_usda_rate_limited():
    with pytest.raises(UsdaRateLimitedError) as exc:
        client_returning({}, status=429).search("pizza", "baked")
    assert exc.value.status_code == 429
    assert exc.value.code == "USDA_RATE_LIMITED"


def test_rejected_api_key_becomes_usda_unavailable_not_unauthorized():
    # A 401 would tell the *user* their credentials were wrong. They were not —
    # ours were. The user-facing failure is "upstream unusable".
    with pytest.raises(UsdaUnavailableError):
        client_returning({}, status=403).search("pizza", "baked")


def test_non_json_body_becomes_usda_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    usda = UsdaClient("k", BASE_URL, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(UsdaUnavailableError):
        usda.search("pizza", "baked")


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------
def test_request_carries_the_documented_search_parameters():
    captured: list[httpx.Request] = []
    client_returning({"foods": [food()]}, captured=captured).search("grilled_salmon", "grilled")

    assert len(captured) == 1
    request = captured[0]
    assert request.url.path.endswith("/foods/search")
    params = request.url.params
    assert params["query"] == "salmon cooked grilled"
    assert params["dataType"] == "Foundation,SR Legacy"
    assert params["pageSize"] == "5"
    assert params["api_key"] == "test-key"


def test_description_is_truncated_to_the_column_width():
    # food_item.food_name is VARCHAR(200) (02 §2.3); an over-long USDA description
    # must be cut here, not by a database error on insert.
    long_name = "X" * 400
    result = client_returning({"foods": [food(description=long_name)]}).search("pizza", "baked")
    assert len(result.description) == 200


# ---------------------------------------------------------------------------
# Local hourly budget (08 §7)
# ---------------------------------------------------------------------------
def test_exhausted_budget_raises_without_making_a_request():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"foods": [food()]})

    usda = UsdaClient(
        "k",
        BASE_URL,
        rate_limiter=HourlyRateLimiter(limit=1),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    usda.search("pizza", "baked")  # spends the only unit of budget
    with pytest.raises(UsdaRateLimitedError):
        usda.search("hamburger", "grilled")

    # The whole point of a local budget is not spending it: exactly one call out.
    assert len(captured) == 1


def test_limiter_resets_after_an_hour():
    now = [0.0]
    limiter = HourlyRateLimiter(limit=2, clock=lambda: now[0])

    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is False

    now[0] = 3600.0
    assert limiter.acquire() is True
    assert limiter.remaining() == 1


def test_refused_acquire_does_not_increment_the_counter():
    limiter = HourlyRateLimiter(limit=1)
    limiter.acquire()
    for _ in range(5):
        assert limiter.acquire() is False
    assert limiter.remaining() == 0


def test_zero_limit_refuses_everything():
    assert HourlyRateLimiter(limit=0).acquire() is False
