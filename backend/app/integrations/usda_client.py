"""
app/integrations/usda_client.py
==================================

WHY THIS FILE EXISTS
---------------------
FR07/FR08 promise the user real nutrition numbers, and `08 §6` states the rule
that makes that promise meaningful: **never fabricate**. Every calorie NutriLens
displays therefore has to come from somewhere authoritative, and the chosen
source is USDA FoodData Central — a government-maintained database of
laboratory-analysed foods (`00`, report §3.2.3). This file is the only place in
the codebase that talks to it.

WHAT PROBLEM IT SOLVES
------------------------
USDA's API is a public HTTP service with its own vocabulary, its own two
incompatible JSON shapes for nutrients, its own quota, and its own downtime. Left
unwrapped, all of that leaks into `nutrition_service`, which would then be a
business-logic module full of `response.json()["foodNutrients"]` indexing and
`except httpx.TimeoutException`. Confining it here means the rest of the app deals
in one small dataclass (`UsdaFood`) and three named failures, and it means the
whole USDA dependency can be replaced with a stub in tests by passing a different
object — which is exactly how M5's tests run without a network or an API key
(`08 §9`).

HOW IT WORKS INTERNALLY
--------------------------
`search(food101_class, cooking_method)` runs four steps:

1. **Build the query.** Delegated to `usda_query_map.build_query` — the Food-101
   class and the user's confirmed cooking method become one search string.
2. **Spend budget.** The `HourlyRateLimiter` is consulted *before* the request, so
   an exhausted budget costs nothing and raises `UsdaRateLimitedError`
   immediately.
3. **Call `GET /foods/search`** with `dataType=Foundation,SR Legacy`,
   `pageSize=5`. Transport failures, 5xx and 429 are translated into the three
   typed errors below; nothing `httpx` raises escapes this module.
4. **Pick a match and read its nutrients.** Candidates are re-sorted by data-type
   trust, then the first one that yields all four macros wins.

WHY THE RESULTS ARE RE-SORTED
--------------------------------
USDA returns results by its own relevance score, which does not know that the
datasets differ in trustworthiness. `08 §5` requires preferring **Foundation** and
**SR Legacy** — foods whose values come from laboratory analysis — over
**Branded**, whose values are self-reported by manufacturers from a label. The
request already filters to the two lab-verified types, so this ordering is
belt-and-braces: it keeps the preference correct if the filter is ever relaxed,
and it makes the ranking policy explicit and testable rather than implicit in a
query parameter.

WHY BOTH NUTRIENT ID SCHEMES ARE HANDLED
-------------------------------------------
USDA identifies nutrients two ways and its endpoints disagree about which to use.
`/foods/search` returns the *abridged* shape, with `nutrientId` (the modern
INFOODS id, energy = 1008) and `nutrientNumber` (the legacy SR number,
energy = "208") flattened onto each entry. `/food/{id}` returns the *full* shape,
where those live in a nested `nutrient` object and the value is `amount` rather
than `value`. Reading only one shape works right up until someone points this
client at the other endpoint, and then it silently finds no nutrients and reports
"no match" for a food that was returned successfully. `08 §5` calls for handling
both, and the cost is a few lines.

WHY A CANDIDATE MISSING A MACRO IS SKIPPED
---------------------------------------------
An absent nutrient is not a zero. If USDA does not report fat for a food, that is
missing data, and defaulting it to `0.0` would put a fabricated number on the
user's plate — the precise thing `08 §6` prohibits, and worse than an error
because it looks like an answer. So a candidate that cannot supply all four
macros is passed over, and if none of the five can, the caller gets
`NoNutritionMatchError`. In practice Foundation and SR Legacy foods carry all
four; this is the guard for the exception, not the common path.

WHY SYNCHRONOUS, WHEN `08 §5` SAYS ASYNC
-------------------------------------------
A deliberate, isolated deviation from the design doc. The rest of the request path
is synchronous — SQLAlchemy's `Session` is a blocking API and every route is a
`def`, which FastAPI already runs in a threadpool. An async client called from
that world would need `asyncio.run()` (or a bridging event loop) inside each
threadpool worker, adding a loop setup per lookup and a class of "attached to a
different loop" bugs, in exchange for concurrency the caller cannot use: a meal
has one to three items and nearly all of them are cache hits (`08 §6`). Should the
backend ever move to async SQLAlchemy, swapping `httpx.Client` for
`httpx.AsyncClient` here is a local change — which is the point of the wrapper.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is a bad API key a 502 and not a 500 or a 401?"
A: From the caller's perspective the upstream dependency is unusable, which is
   what `USDA_UNAVAILABLE` means; a 401 would suggest *the user's* credentials
   were wrong, and they were not. It is logged distinctly so the operator can see
   it is a configuration fault rather than a USDA outage.

Q: "The client raises instead of returning a default. Isn't that fragile?"
A: It is the requirement. `08 §6` sets the fallback order as cached value, else
   the documented error — and the cache lives one layer up, in
   `nutrition_service`, which is the only layer that can see both options. A
   default invented here would be invisible to it.

Q: "What if USDA reports energy only in kilojoules?"
A: It is converted at the defined constant (1 kcal = 4.184 kJ). That is a unit
   change on a measured value, not an estimate — no information is invented — and
   the alternative is a spurious 404 on a food USDA has full data for.

Q: "Why `pageSize=5` rather than 1?"
A: The top hit occasionally lacks a macro or is a less apt data type. Five gives
   the selection logic something to fall through to, and the response is small.

COMMON STUDENT MISTAKES
--------------------------
- Letting `httpx.TimeoutException` propagate, so a slow upstream becomes an
  unhandled 500 with a stack trace instead of the documented 502.
- Putting the API key in the URL path or in source control instead of reading it
  from settings (`08 §2`), or logging the full request URL, which leaks the key.
- Reading `foodNutrients[0]` positionally and assuming the order is stable.
- Treating a missing nutrient as zero.
- Creating a fresh `httpx.Client` per request, discarding connection pooling and
  paying a TLS handshake on every lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.errors import (
    NoNutritionMatchError,
    UsdaRateLimitedError,
    UsdaUnavailableError,
)
from app.core.rate_limit import HourlyRateLimiter
from app.integrations.usda_query_map import build_query

logger = logging.getLogger(__name__)

# 08 §5 — the lab-verified datasets, most trusted first. Anything not listed
# (notably `Branded`, whose values are manufacturer-supplied) sorts last.
_DATA_TYPE_RANK: dict[str, int] = {
    "Foundation": 0,
    "SR Legacy": 1,
    "Survey (FNDDS)": 2,
}
_UNRANKED_DATA_TYPE = 99

# The `dataType` filter sent to USDA. Comma-separated per their API.
SEARCH_DATA_TYPES = "Foundation,SR Legacy"
SEARCH_PAGE_SIZE = 5

# Nutrient identities, in both schemes USDA uses (see the module docstring).
# (modern INFOODS id, legacy SR number)
_ENERGY_KCAL: tuple[int, str] = (1008, "208")
_ENERGY_KJ: tuple[int, str] = (1062, "268")
_PROTEIN: tuple[int, str] = (1003, "203")
_CARBS: tuple[int, str] = (1005, "205")
_FAT: tuple[int, str] = (1004, "204")

KJ_PER_KCAL = 4.184


@dataclass(frozen=True)
class UsdaFood:
    """One food's per-100 g macros, as returned by USDA.

    Frozen because it is a *record of what an external source said*. Nothing
    downstream should be able to adjust these numbers in place — scaling to a
    portion produces new values (`nutrition_service`), it does not edit these.
    """

    fdc_id: str
    description: str
    calories_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float


def _nutrient_ids(entry: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract (id, number) from either the abridged or the full nutrient shape."""
    nested = entry.get("nutrient")
    if isinstance(nested, dict):
        raw_id = nested.get("id")
        raw_number = nested.get("number")
    else:
        raw_id = entry.get("nutrientId")
        raw_number = entry.get("nutrientNumber")

    nutrient_id = int(raw_id) if isinstance(raw_id, int | float | str) and _is_int(raw_id) else None
    number = str(raw_number) if raw_number is not None else None
    return nutrient_id, number


def _is_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _nutrient_amount(entry: dict[str, Any]) -> float | None:
    """The measured value, whichever key this shape uses (`value` vs `amount`)."""
    for key in ("value", "amount"):
        raw = entry.get(key)
        if isinstance(raw, int | float):
            return float(raw)
    return None


def _find_nutrient(nutrients: list[dict[str, Any]], target: tuple[int, str]) -> float | None:
    """The per-100 g value for one nutrient, matched on either id scheme."""
    target_id, target_number = target
    for entry in nutrients:
        if not isinstance(entry, dict):
            continue
        nutrient_id, number = _nutrient_ids(entry)
        if nutrient_id == target_id or (number is not None and number == target_number):
            amount = _nutrient_amount(entry)
            if amount is not None:
                return amount
    return None


def _energy_kcal(nutrients: list[dict[str, Any]]) -> float | None:
    """Energy in kcal, converting from kJ only if kcal is genuinely absent."""
    kcal = _find_nutrient(nutrients, _ENERGY_KCAL)
    if kcal is not None:
        return kcal
    kj = _find_nutrient(nutrients, _ENERGY_KJ)
    if kj is not None:
        return kj / KJ_PER_KCAL
    return None


def parse_food(raw: dict[str, Any]) -> UsdaFood | None:
    """Turn one USDA search hit into a `UsdaFood`, or `None` if it is incomplete.

    `None` rather than an exception because an incomplete candidate is an
    ordinary, expected outcome that the caller handles by trying the next hit.
    """
    nutrients = raw.get("foodNutrients")
    if not isinstance(nutrients, list):
        return None

    calories = _energy_kcal(nutrients)
    protein = _find_nutrient(nutrients, _PROTEIN)
    carbs = _find_nutrient(nutrients, _CARBS)
    fat = _find_nutrient(nutrients, _FAT)
    # All four or nothing — a missing macro is missing data, not zero.
    if calories is None or protein is None or carbs is None or fat is None:
        return None

    fdc_id = raw.get("fdcId")
    description = raw.get("description")
    if fdc_id is None or not isinstance(description, str) or not description:
        return None

    return UsdaFood(
        fdc_id=str(fdc_id),
        description=description[:200],  # food_item.food_name is VARCHAR(200) (02 §2.3)
        calories_per_100g=float(calories),
        protein_per_100g=float(protein),
        carbs_per_100g=float(carbs),
        fat_per_100g=float(fat),
    )


def select_best(foods: list[dict[str, Any]]) -> UsdaFood | None:
    """Pick the most trustworthy complete hit (`08 §5`).

    Sorted by data-type trust with `sorted`'s stability preserving USDA's own
    relevance order inside each tier, so the choice is "the best-scored result of
    the most trusted available dataset".
    """
    ranked = sorted(
        (f for f in foods if isinstance(f, dict)),
        key=lambda f: _DATA_TYPE_RANK.get(str(f.get("dataType", "")), _UNRANKED_DATA_TYPE),
    )
    for candidate in ranked:
        parsed = parse_food(candidate)
        if parsed is not None:
            return parsed
    return None


class UsdaClient:
    """Synchronous wrapper over USDA FoodData Central's search endpoint.

    Construct one per process (the FastAPI lifespan does) so the underlying
    connection pool is reused across requests.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = 10.0,
        rate_limiter: HourlyRateLimiter | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate_limiter = rate_limiter
        # An injected client is how tests supply a `MockTransport` without a network.
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def search(self, food101_class: str, cooking_method: str | None = None) -> UsdaFood:
        """Look up per-100 g macros for a food, or raise one of the three typed errors.

        Raises:
            UsdaRateLimitedError: local hourly budget spent, or USDA returned 429.
            UsdaUnavailableError: timeout, transport failure, 5xx, or a rejected key.
            NoNutritionMatchError: USDA answered, but with nothing usable.
        """
        query = build_query(food101_class, cooking_method)

        if self._rate_limiter is not None and not self._rate_limiter.acquire():
            logger.warning("USDA hourly budget exhausted; refusing lookup for %r", query)
            raise UsdaRateLimitedError()

        payload = self._get_search(query)

        foods = payload.get("foods")
        if not isinstance(foods, list) or not foods:
            raise NoNutritionMatchError(
                f"USDA has no entry matching '{query}'.",
                details={"query": query, "food101_class": food101_class},
            )

        best = select_best(foods)
        if best is None:
            # USDA returned rows, but none carried a complete macro set.
            raise NoNutritionMatchError(
                f"USDA returned no complete nutrition data for '{query}'.",
                details={"query": query, "food101_class": food101_class},
            )

        logger.info("USDA match for %r -> fdc_id=%s (%s)", query, best.fdc_id, best.description)
        return best

    def _get_search(self, query: str) -> dict[str, Any]:
        """Perform the HTTP call, mapping every upstream failure onto a typed error."""
        try:
            response = self._client.get(
                f"{self._base_url}/foods/search",
                params={
                    "query": query,
                    "dataType": SEARCH_DATA_TYPES,
                    "pageSize": SEARCH_PAGE_SIZE,
                    "api_key": self._api_key,
                },
            )
        except httpx.TimeoutException as exc:
            logger.warning("USDA timeout for %r: %s", query, exc)
            raise UsdaUnavailableError("The nutrition database did not respond in time.") from exc
        except httpx.HTTPError as exc:
            logger.warning("USDA transport error for %r: %s", query, exc)
            raise UsdaUnavailableError() from exc

        status = response.status_code
        if status == 429:
            logger.warning("USDA rejected %r with 429 (upstream quota)", query)
            raise UsdaRateLimitedError()
        if status in (401, 403):
            # A configuration fault, not a user fault — surfaced as 502 so the
            # client message stays truthful ("upstream unusable"), logged loudly
            # so the operator sees the real cause.
            logger.error("USDA rejected the API key (HTTP %s) — check USDA_API_KEY", status)
            raise UsdaUnavailableError()
        if status >= 500:
            logger.warning("USDA returned HTTP %s for %r", status, query)
            raise UsdaUnavailableError()
        if status >= 400:
            logger.warning("USDA returned HTTP %s for %r", status, query)
            raise UsdaUnavailableError()

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("USDA returned a non-JSON body for %r", query)
            raise UsdaUnavailableError() from exc

        if not isinstance(payload, dict):
            raise UsdaUnavailableError()
        return payload
