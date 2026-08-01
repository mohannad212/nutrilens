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

WHY THE ONLY REMAINING M6/M7 ENDPOINTS ARE NOT HERE
------------------------------------------------------
`GET /meals`, `GET /meals/{id}` and `DELETE /meals/{id}` are `03 §6` too, but `09`
assigns them to M6 (History). They are left out rather than built ahead because
each needs ownership checks and pagination semantics that the History screen's
design drives. `POST` is the one M5 needs to close the loop.

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

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.deps import get_current_user, get_db, get_usda_client
from app.integrations.usda_client import UsdaClient
from app.schemas.meal import MealCreateRequest, MealCreateResponse
from app.services import meal_service
from app.services.meal_service import MealItemInput

router = APIRouter()


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
