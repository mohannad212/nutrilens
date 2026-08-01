"""
app/api/meals.py
===================

WHY THIS FILE EXISTS
---------------------
`POST /meals` (`03 §6`) — the last step of the core loop and the endpoint whose
existence M5's Definition of Done is written around: "a user photographs a meal and
completes photo->log in <= 3 interactions". This is the third interaction.

WHAT PROBLEM IT SOLVES
------------------------
Almost nothing, deliberately. The interesting decisions — recompute totals, one
transaction, refresh insights after the commit — all live in `meal_service`, where
they are testable against a `Session` without HTTP. What is left here is the part
that genuinely belongs at the transport boundary: authentication, the `201`, and
the mapping to the response contract.

WHY `201` AND NOT `200`
--------------------------
`03 §6` specifies `201`, and it is the correct code for once: unlike
`/recommendations/refresh`, this call really does create a new resource with a new
identifier (`log_id`), and it is genuinely non-idempotent — posting the same body
twice logs two meals, because a user may legitimately eat the same thing twice.

WHERE THE USER ID COMES FROM
-------------------------------
`get_current_user`, and nowhere else. There is no `user_id` in the path or the
body, so there is no code path by which one account can write a meal into
another's history. This is the same rule the recommendations router follows and
the one `11` tests for per-user isolation depend on.

WHY `from`/`to` ARE DECLARED WITH AN ALIAS
--------------------------------------------
`03 §6` names the date filters `?from=` and `?to=`. `from` is a Python keyword, so
the parameter cannot be called that; `Query(alias="from")` keeps the wire name the
contract specifies while the Python identifier is `date_from`. Renaming the query
parameter to dodge the keyword would have been a silent breach of the contract —
the app would send `from` and the server would ignore it, filtering nothing and
returning a full history where the user asked for one day.

WHY 403 AND 404 ARE BOTH POSSIBLE ON A SINGLE MEAL
-----------------------------------------------------
`03 §1`'s split: 404 when no meal with that id exists, 403 when one does but
belongs to another user. The service raises both; the shared `AppError` handler
renders them. See `meal_service.get_meal` for the disclosure trade-off that
distinction carries.

WHY `DELETE` RETURNS 200 WITH A BODY RATHER THAN 204
-------------------------------------------------------
204 No Content is the tidier REST answer for a delete. `03 §6` specifies
`200 {"deleted": true}`, and the reason to follow it is the mobile client: a
response with a parseable body goes through the same success path as every other
call, whereas 204 is the one status where reading `response.data` yields nothing
and needs a special case in the shared axios layer (`07 §5`).

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "The response can take a second or two on a cold cache. Why not return `202`
    and finish in the background?"
A: Because the response carries the persisted totals, and the whole point of this
   screen is confirming what was recorded. An asynchronous log would show the user
   a success message for a meal that might still fail its USDA lookup, and the
   `03 §6` contract would have nothing to put in `total_calories`.

Q: "What happens to the recommendation refresh if it is slow?"
A: It runs after the commit, inside the same request, so it adds to the response
   time but cannot affect whether the meal was saved. Moving it to a background
   task would be the natural next step under load; at GP2's scale it is a handful
   of indexed queries over one user's own rows (`06 §3`).

COMMON STUDENT MISTAKES
--------------------------
- Returning `200` for a create, so the client cannot distinguish "logged" from
  "already existed".
- Taking `user_id` from the request body, which is how one user writes into
  another's history.
- Building the response from the request body instead of from the committed rows,
  so what the user sees is what they sent rather than what was stored.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.deps import get_current_user, get_db, get_usda_client
from app.integrations.usda_client import UsdaClient
from app.schemas.meal import (
    MealCreateRequest,
    MealCreateResponse,
    MealDeleteResponse,
    MealListResponse,
    MealSummary,
)
from app.services import meal_service
from app.services.meal_service import MealItemInput

router = APIRouter()

# `03 §6` defaults, with an upper bound: an unbounded `limit` lets one request ask
# for a user's entire history and serialise every item of every meal.
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


@router.post("/meals", response_model=MealCreateResponse, status_code=status.HTTP_201_CREATED)
def create_meal(
    payload: MealCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    usda: UsdaClient = Depends(get_usda_client),
) -> MealCreateResponse:
    """
    FR10 — persist a confirmed meal.

    Every nutrition value is looked up and computed server-side (`02 §6`); the body
    carries only what the user confirmed. A USDA failure with nothing cached
    surfaces as the documented 429/502/404 and writes nothing.
    """
    items = [
        MealItemInput(
            food101_class=item.food101_class,
            cooking_method=item.cooking_method.value,
            portion_confirmed_g=item.portion_confirmed_g,
            portion_min_g=item.portion_min_g,
            portion_max_g=item.portion_max_g,
            confidence_score=item.confidence_score,
        )
        for item in payload.items
    ]

    meal, total, unlocked = meal_service.log_meal(current_user, payload.meal_type, items, db, usda)
    return MealCreateResponse.from_meal(meal, total, unlocked)


@router.get("/meals", response_model=MealListResponse)
def list_meals(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    # `from` is a Python keyword, so the identifier differs from the wire name.
    # The alias is what keeps the contract's `?from=` working (see the header).
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealListResponse:
    """
    FR11 — the user's meal history, newest first, paginated (`03 §6`).

    Confirmed meals only. The user id comes from the token, never the request, so
    there is no way to page through another account's history.
    """
    meals, total = meal_service.list_meals(
        current_user.user_id,
        db,
        limit=limit,
        offset=offset,
        date_from=date_from,
        date_to=date_to,
    )
    return MealListResponse(
        total=total,
        limit=limit,
        offset=offset,
        meals=[MealSummary.from_meal(meal) for meal in meals],
    )


@router.get("/meals/{log_id}", response_model=MealSummary)
def get_meal(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealSummary:
    """
    One meal in full (`03 §6`). 404 if no such meal, 403 if it is someone else's —
    both raised by the service and rendered by the shared error handler.
    """
    return MealSummary.from_meal(meal_service.get_meal(current_user.user_id, log_id, db))


@router.delete("/meals/{log_id}", response_model=MealDeleteResponse)
def delete_meal(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MealDeleteResponse:
    """
    Delete one meal; its items cascade (`03 §6`). Ownership-checked, so a user can
    only ever delete their own history.
    """
    meal_service.delete_meal(current_user.user_id, log_id, db)
    return MealDeleteResponse()
