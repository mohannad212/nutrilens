"""
app/schemas/recognition.py
=============================

WHY THIS FILE EXISTS
---------------------
`POST /recognise` returns a specific JSON shape the mobile Recognition-Results
screen is built against (`03-api-specification.md §4`). This file is the
authoritative Pydantic contract for that response: the predicted item(s), the
per-prediction confidence, and — when the model is unsure — the alternative
candidates the user can pick from (FR04).

WHAT PROBLEM IT SOLVES
------------------------
Without a declared `response_model`, the route could return any dict and the
mobile client's TypeScript types would silently drift from what the server
actually sends. Declaring these schemas makes the shape a checked contract:
FastAPI validates every response against it and publishes it in `/openapi.json`.

SCOPE NOTE (M3 vs M4)
-----------------------
The full `/recognise` contract in `03 §4` includes a `portion` object per item
(min/max/point grams + method). Portion estimation is **M4** and is not built
yet, so `portion` is deliberately omitted from `Prediction` for now — this
milestone ships the *classification* half (FR03/FR04). When M4 lands, a
`portion` field is added here (and to the mobile types) without changing any of
the fields below.

HOW IT WORKS INTERNALLY
--------------------------
- `Prediction` is one recognised item: `rank` (1 = top-1), the raw Food-101
  class string (`grilled_salmon`), a human-friendly `display_name`
  (`Grilled Salmon`), and `confidence` (0..1, already rounded by the service).
- `Alternative` is a lighter candidate (no rank) used only to fill
  `alternatives` when the top-1 is below the confidence threshold (FR04).
- `RecognitionResponse` is the envelope: a `request_id` (for log correlation),
  the `items`, the `low_confidence` flag, and the `alternatives` list (empty
  when the model was confident).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Separate `Prediction` and `Alternative` types (chosen): the top item carries
  a `rank`, the pick-list candidates don't — modelling them distinctly keeps
  each field meaningful rather than stuffing everything into one optional-heavy
  model.
- Building the response in the service and validating it against these schemas,
  rather than letting the route hand-assemble a dict: guarantees the wire shape
  can't drift from `03 §4` without a schema change a reviewer would notice.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is `portion` missing when the API doc shows it?"
A: Scope. M3 delivers classification; M4 delivers portion estimation. The field
   is added in M4 — shipping the classifier first keeps milestones independently
   demonstrable, exactly as `09-build-phases.md` sequences them.

Q: "What is `request_id` for?"
A: A per-request correlation id echoed in the response and (later) the logs, so
   a specific recognition can be traced end-to-end without persisting the image.

COMMON STUDENT MISTAKES
--------------------------
- Returning the raw class string as the user-facing label (`grilled_salmon`)
  instead of a display name — the `display_name` field exists so the app never
  has to prettify snake_case itself.
- Letting `confidence` be an unrounded float with 15 decimals — the service
  rounds it; the schema just types it.
"""

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """One recognised food item (top-1 today; more once multi-item lands)."""

    rank: int = Field(ge=1, description="1 = the model's top prediction")
    food101_class: str = Field(description="Raw Food-101 class, e.g. 'grilled_salmon'")
    display_name: str = Field(description="Human-friendly label, e.g. 'Grilled Salmon'")
    confidence: float = Field(ge=0.0, le=1.0)


class Alternative(BaseModel):
    """A candidate offered when the top-1 is below the confidence threshold (FR04)."""

    food101_class: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class RecognitionResponse(BaseModel):
    """200 body for POST /recognise (03 §4) — classification half (M3)."""

    request_id: str
    items: list[Prediction]
    low_confidence: bool
    alternatives: list[Alternative]
