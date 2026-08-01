"""
app/api/recommendations.py
=============================

WHY THIS FILE EXISTS
---------------------
The three HTTP endpoints of FR12, exactly as `03 §7` specifies them: read the
current insights, force a regeneration, and mark one read. Every route here is a
thin translation between HTTP and `recommendation_service` — no algorithm, no
queries, no policy.

WHAT PROBLEM IT SOLVES
------------------------
Keeping the routes thin is what makes the rest of M7 testable without HTTP: the
engine is tested with plain values, the service with a `Session`, and this file
only has to be right about status codes, auth, and response shape. If a route
started deciding *when* to regenerate or *what* counts as unlocked, that logic
would only be reachable through a TestClient.

HOW IT WORKS INTERNALLY
--------------------------
`get_current_user` (from `deps.py`) resolves the bearer token to a `User` and
raises `UnauthorizedError` otherwise, so every route below is authenticated by
construction — there is no code path that reads another user's insights, because
the user id never comes from the request. `RecommendationsResponse.from_result`
does the ORM-to-contract mapping.

The only difference between `GET /recommendations` and
`POST /recommendations/refresh` is the `force` flag: GET honours the `06 §3`
idempotency cache, refresh bypasses it.

WHY THESE STATUS CODES
-------------------------
- **`GET` returns 200 even when locked.** `03 §7` is explicit: a user with fewer
  than 7 meals is not an error, it is a normal state with progress information. A
  403 or 404 here would make the mobile client treat "keep logging" as a failure.
- **`refresh` returns 200, not 201.** It is idempotent from the client's point of
  view — "make my insights current" — and the response is the insight set, not a
  pointer to a newly created resource. It is also no-op safe while locked, per
  `03 §7`.
- **`PATCH .../read` returns 200 with the updated row**, so the client reconciles
  from the response rather than firing a follow-up GET.
- **404 vs 403 on `PATCH`** comes from the service and maps to `03 §1`'s
  distinction: 404 when no such insight exists, 403 when it exists but belongs to
  someone else.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is `refresh` a POST if it does not create anything?"
A: It has a side effect — it writes a new batch of `recommendation` rows and does
   real work. GET must stay safe and cacheable; a GET that regenerates rows would
   be surprising to any proxy or client that retries. `03 §7` specifies POST.

Q: "Could a user regenerate insights in a loop and hammer the database?"
A: They could call refresh repeatedly, yes. Each call is a handful of indexed
   queries against their own rows, and the endpoint is authenticated, so the blast
   radius is one account. Rate limiting sits at the deployment tier (`08 §10`); the
   `06 §3` cache is what protects the common path, since the app calls GET.

Q: "Why does the route pass the whole `User` object to the service?"
A: Generation needs the health goal and body metrics. `get_current_user` has
   already loaded the row, so passing the id instead would mean a second query for
   data we are holding.

COMMON STUDENT MISTAKES
--------------------------
- Taking `user_id` from the path or body ("/recommendations/{user_id}") — that is
  how one user ends up reading another's data. The authenticated token is the only
  acceptable source.
- Catching the service's `NotFoundError`/`ForbiddenError` here to build a response.
  The registered `AppError` handler in `main.py` already renders the `03 §9`
  envelope; catching them locally would produce a second, inconsistent format.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.models import User
from app.deps import get_current_user, get_db
from app.schemas.recommendation import (
    MarkReadResponse,
    RecommendationsResponse,
)
from app.services import recommendation_service as svc

router = APIRouter()


@router.get("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationsResponse:
    """
    FR12 — the user's current insights (<= 5).

    200 in both states: `unlocked:false` with progress below 7 confirmed meals,
    `unlocked:true` with insights above it (`03 §7`). Honours the `06 §3` cache, so
    repeatedly opening the Insights screen does not regenerate the text.
    """
    result = svc.get_recommendations(current_user, db)
    return RecommendationsResponse.from_result(result)


@router.post("/recommendations/refresh", response_model=RecommendationsResponse)
def refresh_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationsResponse:
    """
    FR12 — force regeneration from the latest window, bypassing the cache.

    No-op safe while locked: returns the same progress payload as GET rather than
    erroring (`03 §7`).
    """
    result = svc.get_recommendations(current_user, db, force=True)
    return RecommendationsResponse.from_result(result)


@router.patch("/recommendations/{rec_id}/read", response_model=MarkReadResponse)
def mark_recommendation_read(
    rec_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MarkReadResponse:
    """
    Mark one insight read (`03 §7`).

    404 if no such insight exists, 403 if it belongs to another user — both raised
    by the service and rendered by the shared `AppError` handler.
    """
    row = svc.mark_read(current_user.user_id, rec_id, db)
    return MarkReadResponse.model_validate(row)
