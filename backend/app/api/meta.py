"""
app/api/meta.py
==================

WHY THIS FILE EXISTS
---------------------
`03 §8` specifies `GET /meta/cooking-methods` with a one-line rationale that is
the entire justification for the endpoint: "so the app dropdown stays in sync".
M5 adds the cooking-method picker to the mobile app, and this is where its options
come from.

WHAT PROBLEM IT SOLVES
------------------------
The alternative is hard-coding the eight values in the React Native picker. That
works until the enum changes on one side only, and then the picker offers a value
the request schema rejects with a 422 — a bug that appears only for the most
recently added option, only on a device, and only after a release. Serving the
list from the same `CookingMethod` enum the validator uses makes that
disagreement impossible to express.

It also removes a class of app release: adding a cooking method becomes a backend
deploy plus a cache row, with no store submission for the picker to catch up.

WHY IT IS PUBLIC
-------------------
`03 §8` groups it with `/health` under "Health & meta" and marks neither with
"**Auth.**", unlike every endpoint in §2-§7. The content is a static list of eight
English words with no user data in it, so requiring a token would buy nothing and
would stop the app pre-loading the picker on the login screen.

`GET /meta/food-classes`
---------------------------
Deferred until the screen that needed it existed, which is now: the results
screen lets the user overrule a wrong prediction by picking from the full 101.
`/recognise`'s `alternatives` cannot serve that case — they are only populated
*below* the confidence threshold, so a prediction that is confidently wrong (a
plate of pasta returned as "French Fries" at 0.71) offers the user nothing to
correct it with, and the whole downstream chain inherits the error.

The open question was what to serve when the labels are missing. It resolved
itself: `labels_food101.json` is **tracked in git** — only the `.pth` weights ship
through a Release (`backend/models/README.md`). So unlike the classifier, the
labels are always present in a checkout, and their absence is a broken deployment
rather than the expected "weights not downloaded yet" state. The endpoint reads
the file and lets a genuine IO error surface as a 500, instead of inventing a
degraded mode for a case that cannot occur in a valid install.

It stays public for the same reason as the cooking methods: 101 English food
names, no user data.

COMMON STUDENT MISTAKES
--------------------------
- Hard-coding the list here as well, which recreates the drift this endpoint
  exists to prevent. It is derived from the enum, so there is exactly one source.
- Putting the list behind auth and then wondering why the picker is empty until
  after login.
"""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.core.enums import COOKING_METHODS

router = APIRouter()


@router.get("/meta/cooking-methods")
def cooking_methods() -> dict[str, list[str]]:
    """`03 §8` / `03 §10` — the cooking-method enum, in contract order."""
    return {"cooking_methods": COOKING_METHODS}


def _display_name(food101_class: str) -> str:
    """`grilled_salmon` -> `Grilled Salmon`, matching `recognition_service`."""
    return food101_class.replace("_", " ").title()


@lru_cache(maxsize=1)
def _load_food_classes() -> list[dict[str, str]]:
    """The 101 labels, alphabetical, read once per process.

    Cached because the file never changes while the process lives, and this is a
    public endpoint — an uncached read would be free disk IO for any caller.
    """
    path = Path(settings.model_dir) / settings.classifier_labels
    raw: dict[str, str] = json.loads(path.read_text())
    return [
        {"food101_class": cls, "display_name": _display_name(cls)}
        for cls in sorted(raw.values())
    ]


@router.get("/meta/food-classes")
def food_classes() -> dict[str, list[dict[str, str]]]:
    """`03 §8` — every class the model can predict, for the manual-correction list.

    Sorted alphabetically by class rather than kept in label-index order: the
    consumer is a human scrolling a list, and the index order is an artefact of
    how the model was trained.
    """
    return {"food_classes": _load_food_classes()}
