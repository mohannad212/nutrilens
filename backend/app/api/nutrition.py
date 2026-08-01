"""
app/api/nutrition.py
=======================

WHY THIS FILE EXISTS
---------------------
The HTTP surface for FR07/FR08 — `POST /nutrition` (`03 §5`), the call the mobile
Nutrition Summary screen makes once the user has confirmed *what* the food is, *how*
it was cooked, and *how much* of it there is. Like every router in this project it
is deliberately thin: validate, delegate, return.

WHAT PROBLEM IT SOLVES
------------------------
It keeps the cache-versus-USDA policy out of the transport layer. Everything
interesting about this endpoint — which source to trust, when a stale value beats
an error, how to scale — lives in `nutrition_service`, where it can be tested with
a `Session` and a stub and no TestClient. What is left here is the part that can
only be got right at the HTTP boundary: the auth dependency, the response model,
and the status code.

WHY IT IS A `POST` FOR WHAT IS ARGUABLY A READ
-------------------------------------------------
`03 §5` specifies POST, and the shape of the request is why: three parameters, one
of them a float, that together identify a *computation* rather than a resource. A
GET with a query string would work, but it would also invite intermediaries to
cache `?food101_class=pizza&cooking_method=baked&portion_g=210` as though the
response were a stable document — and it is not, because the underlying cache row
can be refreshed from USDA between calls. There is also a real side effect on the
first call for a food: the USDA result is written into `food_item`. A GET that
writes rows is exactly the thing HTTP semantics tell you not to build.

WHY THE ERRORS ARE NOT CAUGHT HERE
-------------------------------------
`UsdaRateLimitedError`, `UsdaUnavailableError` and `NoNutritionMatchError` are
`AppError` subclasses carrying their own status codes (429/502/404 per `03 §5`),
and `main.py` registers one handler that renders them into the `03 §9` envelope.
Catching them in the route to build a response would produce a second, subtly
different error format for the mobile client's single interceptor to trip over.

WHY THE SESSION IS COMMITTED HERE AND NOT IN THE SERVICE
-----------------------------------------------------------
A USDA miss writes a `food_item` row, and that write should persist — it is the
whole point of the cache. But the *service* must not commit, because
`meal_service` calls the same code inside a larger transaction where a mid-way
commit would leave a half-written meal (`08 §6`). So the commit lives at the
request boundary, which is the layer that knows the unit of work is finished. This
endpoint is that boundary; `POST /meals` is its own.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is this a separate endpoint at all, if `POST /meals` recomputes nutrition
    anyway?"
A: Because the user sees the numbers *before* deciding to log (`01`'s core loop:
   recognise -> portion -> nutrition -> log). They may look at 640 kcal and change
   the portion, or the cooking method, or abandon the meal. Folding it into
   `/meals` would mean the only way to see nutrition is to have already committed
   to logging it.

Q: "Could a caller pass a portion that disagrees with what `/recognise` estimated?"
A: Yes, and that is FR06 working as intended — the user is the authority on their
   own portion, and the slider exists precisely so they can override the estimate.
   The server validates plausibility (`schemas/nutrition.py`), not agreement.

COMMON STUDENT MISTAKES
--------------------------
- Committing inside the service, which makes it unusable from `POST /meals`.
- Putting the cache-versus-USDA decision in the route, which makes it reachable
  only through a TestClient.
- Forgetting `Depends(get_current_user)` on an endpoint that spends a shared API
  quota — an unauthenticated caller could drain the hourly budget for everyone.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import User
from app.deps import get_current_user, get_db, get_usda_client
from app.integrations.usda_client import UsdaClient
from app.schemas.nutrition import NutritionRequest, NutritionResponse
from app.services import nutrition_service

router = APIRouter()


@router.post("/nutrition", response_model=NutritionResponse)
def get_nutrition(
    payload: NutritionRequest,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    usda: UsdaClient = Depends(get_usda_client),
) -> NutritionResponse:
    """
    FR07/FR08 — macros for one confirmed item, scaled to its confirmed portion.

    Cache-first (`08 §6`). A miss calls USDA and caches the result, which is why
    the session is committed here even though nothing user-owned was written.
    """
    result = nutrition_service.get_nutrition(
        db,
        payload.food101_class,
        payload.cooking_method.value,
        payload.portion_g,
        usda,
    )
    db.commit()
    return NutritionResponse.from_result(result)
