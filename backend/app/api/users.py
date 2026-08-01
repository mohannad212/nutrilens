"""
app/api/users.py
===================

WHY THIS FILE EXISTS
---------------------
The `/users` endpoints of `03 §3`: read the current profile, change the health
goal (FR13), and edit the optional body metrics M7's calorie insight uses. These
back the Profile & Goals screen (`07 §2`, Screen 8).

WHY EVERY ROUTE IS `/users/me` AND NEVER `/users/{id}`
---------------------------------------------------------
This is the security property of the whole file, and it is worth stating as a
design rule rather than as three separate checks. The user id comes only from the
authenticated token, via `get_current_user`. There is no path parameter and no
body field carrying a user id, so there is no request a client can construct that
reads or writes another account's profile — the failure mode is *unrepresentable*
rather than defended against.

An endpoint of the form `PATCH /users/{user_id}/goal` would need an ownership
comparison on every route, and the day someone adds a fourth route and forgets
it is the day one user can edit another's data. `03 §3` specifies `/me`, and this
is why.

WHY `PATCH` AND NOT `PUT`
----------------------------
`PUT` means "replace the resource with this representation", which for the
metrics endpoint would mean an omitted field is *deleted*. That is the opposite
of the specified behaviour — `03 §3` says omitted keys are left unchanged — and
it is the behaviour that silently destroys data when a screen submits one field.
`PATCH` is the correct verb for a partial update and the one the contract names.

WHERE `DELETE /users/me` IS
------------------------------
`03 §3` also specifies it, for FR14. `09` assigns it to M8, and it is genuinely
absent rather than stubbed: it performs an irreversible cascade across every table
the user owns, needs its `confirm` flag guard, and deserves to be built with the
double-confirmation flow in the app that protects it, not ahead of it.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "`GET /users/me` returns data already in the JWT. Why not read it from the token?"
A: The token carries an id and an expiry, not a profile, and it is a *snapshot*
   from up to an hour ago. A goal changed five minutes ago would not be in it.
   Profile data has to come from the row.

Q: "Why does changing the goal return the whole profile rather than just the goal?"
A: `03 §3` specifies "returns updated profile", and it saves the client a
   follow-up GET to reconcile its cached copy after a mutation.

COMMON STUDENT MISTAKES
--------------------------
- Adding a `user_id` path parameter "for REST purity", which converts a property
  the type system guarantees into one a reviewer has to verify per route.
- Using `PUT` for the metrics endpoint and wiping unmentioned fields.
- Returning the ORM object rather than the response model, publishing
  `password_hash` to the client.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import User
from app.deps import get_current_user, get_db
from app.schemas.user import GoalUpdateRequest, MetricsUpdateRequest, UserProfile
from app.services import user_service

router = APIRouter()


@router.get("/users/me", response_model=UserProfile)
def read_profile(current_user: User = Depends(get_current_user)) -> User:
    """The authenticated user's profile (`03 §3`).

    No query needed: `get_current_user` already loaded the row while
    authenticating the token.
    """
    return current_user


@router.patch("/users/me/goal", response_model=UserProfile)
def update_goal(
    payload: GoalUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """
    FR13 — update the health goal (`03 §3`).

    Takes effect on the next recommendation refresh (`06`), not immediately.
    """
    return user_service.update_goal(current_user, payload.health_goal, db)


@router.patch("/users/me/metrics", response_model=UserProfile)
def update_metrics(
    payload: MetricsUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """
    Update any subset of the five optional body metrics (`03 §3`).

    Omitted keys are left unchanged; an explicit `null` clears a field. That
    distinction is carried by `provided_fields()`, which reports only the keys
    actually present in the request body.
    """
    return user_service.update_metrics(current_user, payload.provided_fields(), db)
