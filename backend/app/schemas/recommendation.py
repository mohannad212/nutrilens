"""
app/schemas/recommendation.py
================================

WHY THIS FILE EXISTS
---------------------
`03-api-specification.md` is a contract, and `§7` pins the exact JSON the
`/recommendations` endpoints return — including a field the mobile app relies on
to decide which screen to draw (`unlocked`) and a field that is a stated NFR
(`disclaimer`). This file is that contract expressed as Pydantic models, so the
shape is enforced by the framework rather than assembled by hand in each route.

WHAT PROBLEM IT SOLVES
------------------------
Two specific risks, both of which a hand-built dict invites:

1. **A missing disclaimer.** `06 §5` makes it non-negotiable on every insight
   response. Here it is a non-optional field with a constant default, so a
   response object *cannot be constructed without it*. No route has to remember.
2. **A leaked ORM column.** `Recommendation` rows carry `user_id`; returning them
   directly would publish it. `InsightOut` lists exactly the four fields `03 §7`
   specifies, so there is no code path that can widen the payload by accident.

HOW IT WORKS INTERNALLY
--------------------------
One response model covers both states rather than two models behind a union. The
locked case populates `meals_logged` / `meals_required` / `message` and leaves
`insights` empty; the unlocked case populates `insights` / `generated_at`. The
`unlocked` boolean is the discriminator the client switches on, exactly as in
`03 §7`.

`from_recommendations()` maps a `RecommendationResult` (the service's dataclass,
which holds ORM rows) into this shape. Keeping that conversion here rather than in
the route means the routes stay two lines each and the mapping is tested once.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- One model with an `unlocked` flag (chosen): matches `03 §7`, where both states
  are documented as the same object with different fields populated. A client
  parses one type and branches on a boolean.
- Two models in a `Union[LockedResponse, UnlockedResponse]`: arguably cleaner
  Python, but it changes the generated OpenAPI schema into a `oneOf` that the
  mobile TypeScript types would then have to narrow — more work at the boundary
  that matters, to tidy an internal detail.
- `progress` as a nested object: would read nicely, but `03 §7` puts
  `meals_logged` and `meals_required` at the top level and the contract wins.
- Returning `insights` as `None` when locked rather than `[]`: an empty list lets
  the client iterate unconditionally. `03 §7` shows `"insights": []` in the locked
  example, so this also matches.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is the disclaimer a field on every response instead of documentation?"
A: Because it must reach the user, not the developer. `07` renders it on the
   Insights screen, and a response that omitted it would let the app display
   dietary suggestions with no qualifier — the exact NFR failure `06 §5` forbids.
   Making it a required field with a constant default means the only way to ship a
   response without one is to delete the field, which breaks the tests in `11`.

Q: "Why does the locked response carry a `message` the server wrote?"
A: So the progress copy lives in one place. The alternative is the app computing
   "log 3 more meals" from the two counts, which duplicates the logic and drifts
   the moment the threshold changes.

COMMON STUDENT MISTAKES
--------------------------
- Adding `response_model=` to the route but building the dict by hand anyway —
  FastAPI then silently drops unexpected keys and the mismatch goes unnoticed
  until the app breaks.
- Using `datetime.now()` in a default for `generated_at`: it would be evaluated at
  import time, stamping every response with the moment the process started.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import RECOMMENDATION_DISCLAIMER
from app.services.recommendation_service import RecommendationResult


class InsightOut(BaseModel):
    """One insight as `03 §7` defines it — note the absence of `user_id`."""

    model_config = ConfigDict(from_attributes=True)

    rec_id: int
    insight_text: str
    nutrient_focus: str | None
    was_read: bool


class RecommendationsResponse(BaseModel):
    """
    The body of `GET /recommendations` and `POST /recommendations/refresh` (`03 §7`).

    Both the locked and unlocked states share this shape; `unlocked` is the
    discriminator the mobile client branches on.
    """

    unlocked: bool
    insights: list[InsightOut] = Field(default_factory=list)
    generated_at: datetime | None = None
    # Locked-state progress (03 §7). Present in both states so the client can show
    # "7 of 7 meals" once unlocked without a second request.
    meals_logged: int
    meals_required: int
    # Locked progress copy, or the "no meals in the window" notice. Absent when
    # there are insights to show instead.
    message: str | None = None
    # 06 §5 / NFR: never optional, never empty.
    disclaimer: str = RECOMMENDATION_DISCLAIMER

    @classmethod
    def from_result(cls, result: RecommendationResult) -> "RecommendationsResponse":
        """Map the service's dataclass (which holds ORM rows) onto the contract."""
        return cls(
            unlocked=result.unlocked,
            insights=[InsightOut.model_validate(row) for row in result.insights],
            generated_at=result.generated_at,
            meals_logged=result.meals_logged,
            meals_required=result.meals_required,
            message=result.message,
            disclaimer=result.disclaimer,
        )


class MarkReadResponse(BaseModel):
    """
    The body of `PATCH /recommendations/{rec_id}/read` (`03 §7`).

    Returns the updated insight so the client can reconcile its local state from
    the response instead of firing a follow-up GET.
    """

    model_config = ConfigDict(from_attributes=True)

    rec_id: int
    was_read: bool
