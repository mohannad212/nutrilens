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

WHY NOT `GET /meta/food-classes` TOO
---------------------------------------
`03 §8` lists it, and it is genuinely deferred rather than forgotten. It returns
the 101 labels, which live in `models/labels_food101.json` — an artefact the ML
track ships through a GitHub Release and which may legitimately be absent
(`04 §7`; the lifespan in `main.py` already tolerates that). Serving it therefore
needs a decision about what to return when the file is missing, and the screen
that consumes it (a manual "wrong prediction, pick the right food" list) is not
part of M5's flow — `/recognise` already returns the alternatives the user chooses
between (`03 §4`). It belongs with whichever milestone builds that screen.

COMMON STUDENT MISTAKES
--------------------------
- Hard-coding the list here as well, which recreates the drift this endpoint
  exists to prevent. It is derived from the enum, so there is exactly one source.
- Putting the list behind auth and then wondering why the picker is empty until
  after login.
"""

from fastapi import APIRouter

from app.core.enums import COOKING_METHODS

router = APIRouter()


@router.get("/meta/cooking-methods")
def cooking_methods() -> dict[str, list[str]]:
    """`03 §8` / `03 §10` — the cooking-method enum, in contract order."""
    return {"cooking_methods": COOKING_METHODS}
