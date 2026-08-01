"""
app/services/user_service.py
===============================

WHY THIS FILE EXISTS
---------------------
`08 §4` specifies a `user_service` for FR13 (health goal) and FR14 (account
deletion). M6 builds the FR13 half plus the body-metrics editor that M7's calorie
insight depends on; FR14's `delete_account` is M8's and is deliberately absent
rather than half-written.

WHAT PROBLEM IT SOLVES
------------------------
Very little logic, and that is the honest description — these are updates to
columns on a row the caller already holds. What the module buys is a place for
the two rules that are easy to get wrong at the router level, and a seam that
lets both be tested without HTTP.

THE PARTIAL-UPDATE RULE
--------------------------
`03 §3`: "Omitted keys are left unchanged; an explicit `null` clears a field."
`update_metrics` receives an already-filtered dict from
`MetricsUpdateRequest.provided_fields()`, containing only keys the client
actually sent. Every key in it is written, including one whose value is `None`.

The failure this prevents is destructive and silent. A Profile screen that
PATCHes just the weight, against an implementation that writes all five fields
from the parsed model, sets the other four to `NULL` — quietly deleting the
height, birth year, sex and activity level the user entered at registration, and
downgrading their calorie insight back to a descriptive note with nothing on
screen to explain why.

WHY UPDATING THE GOAL DOES NOT REGENERATE INSIGHTS
-----------------------------------------------------
`03 §3` describes the side effect as "next recommendation refresh uses the new
target distribution" — *next*, not now. That is the behaviour implemented here:
the goal is written and nothing else happens.

Regenerating eagerly was considered and rejected. It would put a multi-query
engine run inside a request whose job is to set one enum, and `06 §3`'s cache
already makes the next `GET /recommendations` do the work at the moment the user
actually looks at the Insights screen. The one visible consequence is that a user
who changes their goal and immediately opens Insights may see insights computed
against the old target until the cache expires — which is why `06 §3`'s cache is
keyed on the newest meal rather than on the goal, and why `POST
/recommendations/refresh` exists for a client that wants them current now.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why does `update_goal` take a `User` object rather than a user id?"
A: `get_current_user` has already loaded and authenticated the row. Passing the
   id would mean a second query for data the caller is holding, and — more
   importantly — it would create a signature that *permits* acting on a user
   other than the authenticated one. Taking the object makes that impossible to
   express.

Q: "Where is the authorisation check?"
A: There isn't one, and there does not need to be: every function here acts on
   the `User` the token resolved to. There is no path by which a user id from a
   request body could reach these functions.

COMMON STUDENT MISTAKES
--------------------------
- Writing all fields on a PATCH, wiping the ones the client did not mention.
- Accepting a `user_id` parameter from the request so one account can edit
  another's profile.
- Committing in the service *and* the router, producing two transactions where
  the second one has nothing to write.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import HealthGoal, User

logger = logging.getLogger(__name__)

# The only columns `PATCH /users/me/metrics` may touch. An allowlist rather than
# "whatever the schema produced", so that adding a field to the schema is never
# by itself enough to make it writable here.
EDITABLE_METRIC_FIELDS: frozenset[str] = frozenset(
    {"height_cm", "weight_kg", "birth_year", "sex", "activity_level"}
)


def update_goal(user: User, goal: HealthGoal, db: Session) -> User:
    """FR13 — set the user's health goal (`03 §3`).

    Takes effect on the next recommendation refresh, not immediately; see the
    module docstring for why that is the specified and intended behaviour.
    """
    user.health_goal = goal
    db.commit()
    db.refresh(user)
    return user


def update_metrics(user: User, provided: dict[str, Any], db: Session) -> User:
    """Update any subset of the five optional body metrics (`03 §3`).

    `provided` must contain *only* the keys the client actually sent — see
    `MetricsUpdateRequest.provided_fields()`. A key mapped to `None` clears that
    field; a key that is absent leaves it untouched.
    """
    for field, value in provided.items():
        if field not in EDITABLE_METRIC_FIELDS:
            # Unreachable through the schema; the allowlist is the guard that
            # keeps it unreachable if the schema later grows a field.
            logger.warning("Ignoring non-editable field %r in metrics update", field)
            continue
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return user
