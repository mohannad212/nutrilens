"""
scripts/seed_food101_classes.py
==================================

WHY THIS FILE EXISTS
---------------------
The `food_item` table doubles as a *USDA nutrition cache* keyed by
`(food101_class, cooking_method)` — but the classifier can only ever
predict one of the 101 Food-101 labels, so it's useful to pre-populate one
row per class *before* any USDA lookup has happened, with macros left
NULL. This script does that one-time (well, idempotent) seeding, reading
the canonical label list from `models/labels_food101.json`.

WHAT PROBLEM IT SOLVES
------------------------
Without this, the very first time a user's photo is classified as, say,
"waffles", the nutrition service would need special-case logic for
"this food_id doesn't exist yet, create it inline". Seeding all 101
classes up front means every classifier prediction always maps to an
*existing* `food_item` row — the nutrition service (`08`, built in a
later milestone) only ever needs to check "does this row have macros
cached yet?", never "does this row exist at all?".

HOW IT WORKS INTERNALLY
--------------------------
Reads the JSON array of 101 class name strings, and for each one runs an
"insert if not exists" — checking first whether a `food_item` row with
that `food101_class` already exists (using the class name itself as a
placeholder `food_name` and `usda_fdc_id=NULL`, since the real USDA name
and id are only known once nutrition_service does its first lookup for
that class). Wrapping every insert in this existence check is what makes
the script *idempotent*: running it five times leaves exactly 101 rows,
not 505.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Idempotent "check-then-insert" seed script run manually/via CI (chosen):
  safe to re-run in any environment (fresh dev DB, CI, staging) without
  duplicate-key errors or duplicate rows.
- A raw SQL `INSERT INTO food_item (...) VALUES (...), (...), ...` script:
  would need `ON CONFLICT DO NOTHING` to be idempotent, and loses the
  benefit of going through the ORM's session/transaction handling — using
  the ORM here keeps this script consistent with everywhere else in the
  app that touches food_item.
- Baking the 101 rows into the initial Alembic migration itself: schema
  migrations should describe *structure* (tables/columns/types), not seed
  *data* — mixing the two makes migrations harder to reason about and
  couples a one-time data load to schema versioning.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why NULL macros at seed time instead of looking up USDA for every
    class right now?"
A: That would mean 101 USDA API calls before the app even starts, several
   of them for foods (multiplied by cooking method) never actually
   photographed by a real user. Lazily filling macros in on first lookup
   (per the cache design in `02 §2.3` / `08`) avoids wasted API calls
   against the 3,600-request/hour limit.

Q: "Why is this idempotent, and why does that matter?"
A: Because the script is meant to be safe to run every time a fresh
   environment (a new developer's machine, CI, staging) is stood up —
   idempotency means "seed" never means "risk of a duplicate-key crash on
   the second run."

COMMON STUDENT MISTAKES
--------------------------
- Running raw `INSERT` without checking for existing rows first — crashes
  on the unique `(food101_class, cooking_method)` index on any re-run.
- Forgetting to commit the session — SQLAlchemy won't persist anything
  until `.commit()` is called.
- Hard-coding the class list inline instead of reading
  `labels_food101.json` — creates two sources of truth that can drift.
"""

import json
import sys
from pathlib import Path

# Allow `python scripts/seed_food101_classes.py` from backend/ (adds backend/
# to the import path even when the package isn't editable-installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import SessionLocal  # noqa: E402
from app.db.models import FoodItem  # noqa: E402

LABELS_PATH = Path(__file__).resolve().parent.parent / "models" / "labels_food101.json"


def seed_food101_classes() -> None:
    with open(LABELS_PATH) as f:
        classes: list[str] = json.load(f)

    db = SessionLocal()
    try:
        inserted = 0
        for class_name in classes:
            exists = (
                db.query(FoodItem)
                .filter(
                    FoodItem.food101_class == class_name,
                    FoodItem.cooking_method.is_(None),
                )
                .first()
            )
            if exists is not None:
                continue

            db.add(
                FoodItem(
                    usda_fdc_id=None,
                    food_name=class_name.replace("_", " ").title(),
                    food101_class=class_name,
                    cooking_method=None,
                    calories_per_100g=None,
                    protein_per_100g=None,
                    carbs_per_100g=None,
                    fat_per_100g=None,
                )
            )
            inserted += 1

        db.commit()
        print(
            f"seed_food101_classes: inserted {inserted} new row(s), "
            f"{len(classes) - inserted} already present."
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed_food101_classes()
