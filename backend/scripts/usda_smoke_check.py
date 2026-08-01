"""
scripts/usda_smoke_check.py
==============================

WHY THIS SCRIPT EXISTS
-----------------------
`usda_query_map.py` maps 101 Food-101 classes onto USDA search terms, and no unit
test can tell you whether those mappings are *right* — a stubbed test proves the
query was built as designed, not that USDA answers it with the correct food. The
only way to know is to ask the real API and read what comes back.

This script does that: it runs every class (or a named subset) through the real
`UsdaClient` and prints the food USDA actually returns, so a human can scan the
list and spot "pizza -> Pork, cured, bacon". That scan is what produced the
`+`-required-terms finding documented in `usda_query_map`, where the original
query form was returning a different food entirely for many classes.

WHY IT IS A SCRIPT AND NOT A TEST
------------------------------------
Three reasons, and they are the standard ones for anything that touches a live
third-party service. It needs a real `USDA_API_KEY`, which CI does not have. It
spends rate-limit budget — around 101 requests, more when the fallback chain
runs. And its result depends on USDA's data and ranking, both of which can change
without anything in this repo changing, so a failure would not indicate a
regression in our code. The fast, deterministic, network-free tests live in
`tests/test_m5_usda_client.py`; this is the periodic reality check beside them.

HOW TO READ THE OUTPUT
-------------------------
Each row is one class: the query that succeeded, and the food USDA returned with
its calories. What you are looking for:

- **`NO MATCH`** — USDA has no entry for this dish. Not necessarily a bug: for
  something like `sushi` it is the honest answer, and the app shows a clean 404
  rather than inventing a number. It is a bug when the food obviously *does*
  exist in USDA under another name, and the fix is an override table entry.
- **A description from the wrong food group** — "pizza" returning a pork product
  is the failure this script is for. Fix the override.
- **An implausible calorie figure** — a green vegetable at 500 kcal/100 g means
  the match is wrong even if the name looks vaguely right.

USAGE
--------
    python scripts/usda_smoke_check.py                 # all 101 classes
    python scripts/usda_smoke_check.py pizza sushi     # just these
    python scripts/usda_smoke_check.py --limit 20      # first 20, to save quota

Requires `USDA_API_KEY` in the environment or `.env` (see `08 §2`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow `python scripts/usda_smoke_check.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.core.errors import (  # noqa: E402
    NoNutritionMatchError,
    UsdaRateLimitedError,
    UsdaUnavailableError,
)
from app.integrations.usda_client import UsdaClient  # noqa: E402
from app.integrations.usda_query_map import build_query_chain  # noqa: E402

LABELS_PATH = Path(__file__).resolve().parent.parent / "models" / "labels_food101.json"

# A plausible default preparation per class, so the query this script sends is the
# one a real user would most likely produce. Anything unlisted uses `other`, which
# contributes no method words at all.
DEFAULT_METHODS: dict[str, str] = {
    "apple_pie": "baked",
    "baby_back_ribs": "roasted",
    "beef_carpaccio": "raw",
    "beef_tartare": "raw",
    "beet_salad": "raw",
    "beignets": "fried",
    "bruschetta": "baked",
    "caesar_salad": "raw",
    "caprese_salad": "raw",
    "ceviche": "raw",
    "chicken_wings": "fried",
    "chicken_curry": "boiled",
    "gnocchi": "boiled",
    "pulled_pork_sandwich": "roasted",
    "ramen": "boiled",
    "dumplings": "steamed",
    "gyoza": "fried",
    "chocolate_cake": "baked",
    "clam_chowder": "boiled",
    "edamame": "steamed",
    "filet_mignon": "grilled",
    "fish_and_chips": "fried",
    "french_fries": "fried",
    "french_onion_soup": "boiled",
    "fried_calamari": "fried",
    "fried_rice": "fried",
    "greek_salad": "raw",
    "grilled_salmon": "grilled",
    "guacamole": "raw",
    "hamburger": "grilled",
    "hot_and_sour_soup": "boiled",
    "hummus": "raw",
    "ice_cream": "raw",
    "lobster_bisque": "boiled",
    "miso_soup": "boiled",
    "mussels": "steamed",
    "omelette": "fried",
    "onion_rings": "fried",
    "oysters": "raw",
    "peking_duck": "roasted",
    "pho": "boiled",
    "pizza": "baked",
    "pork_chop": "grilled",
    "prime_rib": "roasted",
    "sashimi": "raw",
    "scallops": "grilled",
    "seaweed_salad": "raw",
    "spring_rolls": "fried",
    "steak": "grilled",
    "sushi": "raw",
    "tuna_tartare": "raw",
}


def load_class_names(path: Path) -> list[str]:
    """Read the labels file, accepting the index->class map or a flat list."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return [raw[key] for key in sorted(raw, key=int)]
    return list(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate USDA query mappings live.")
    parser.add_argument("classes", nargs="*", help="Specific Food-101 classes (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Only check the first N classes")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between calls")
    args = parser.parse_args()

    if not settings.usda_api_key:
        print("USDA_API_KEY is not set. Add it to .env (see 08 §2).", file=sys.stderr)
        return 2

    classes = args.classes or load_class_names(LABELS_PATH)
    if args.limit:
        classes = classes[: args.limit]

    client = UsdaClient(settings.usda_api_key, settings.usda_base_url, timeout=25.0)
    misses: list[str] = []
    matched = 0

    print(f"Checking {len(classes)} class(es) against {settings.usda_base_url}\n")
    print(f"{'class':<26} {'method':<9} {'kcal':>6}  description")
    print("-" * 100)

    try:
        for name in classes:
            method = DEFAULT_METHODS.get(name, "other")
            try:
                food = client.search(name, method)
            except NoNutritionMatchError:
                misses.append(name)
                query = build_query_chain(name, method)[0]
                print(f"{name:<26} {method:<9} {'--':>6}  NO MATCH   (tried {query!r})")
            except UsdaRateLimitedError:
                print(f"\nRate limited after {matched} lookups — stopping.", file=sys.stderr)
                break
            except UsdaUnavailableError as exc:
                print(f"{name:<26} {method:<9} {'--':>6}  UPSTREAM ERROR ({exc.message})")
            else:
                matched += 1
                print(f"{name:<26} {method:<9} {food.calories_per_100g:>6.0f}  {food.description}")

            time.sleep(args.delay)
    finally:
        client.close()

    print("-" * 100)
    print(f"\nmatched: {matched}/{len(classes)}   no match: {len(misses)}")
    if misses:
        print("\nNo USDA entry for:")
        for name in misses:
            print(f"  - {name}")
        print(
            "\nEach of these currently returns a clean 404 NO_NUTRITION_MATCH to the app.\n"
            "That is correct when USDA genuinely has no such food; add an override in\n"
            "app/integrations/usda_query_map.py when it has one under a different name."
        )

    # Always 0: a "miss" is information for a human, not a failing build. This
    # script reports, it does not gate (see the header note on why it is not a test).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
