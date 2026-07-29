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

SCOPE NOTE (M3 → M4)
-----------------------
The full `/recognise` contract in `03 §4` includes a `portion` object per item
(min/max/point grams + method + reference). M3 shipped the *classification* half
(FR03/FR04) with `portion` omitted; **M4** adds it. The `portion` field on
`Prediction` is `Portion | None` — it is populated when the portion pipeline runs
(`05`) and is `None` when portion estimation is disabled (`settings.portion_enabled`
= false) or could not run, so the classification half never depends on it.

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

from typing import Literal

from pydantic import BaseModel, Field


class Portion(BaseModel):
    """Calibrated mass-range estimate for one item (FR05; `03 §4`, `05`).

    Always a range — `point_g` is only the slider's default midpoint, never a
    claim of precision. `method` records which scale stage produced it and
    `reference` names the cue (a container/reference-object class) or is null.
    """

    min_g: float = Field(ge=0.0, description="Lower bound of the plausible mass")
    max_g: float = Field(ge=0.0, description="Upper bound of the plausible mass")
    point_g: float = Field(ge=0.0, description="Midpoint; the slider's default")
    method: Literal["reference_object", "container_prior", "shape_only"]
    reference: str | None = Field(
        default=None, description="Scale cue used, e.g. 'dinner_plate' or 'fork'; null if none"
    )
    scale_source: str = Field(
        description="Human-readable explanation of how the scale was derived, "
        "e.g. 'measured from a fork in the photo'"
    )


class Prediction(BaseModel):
    """One recognised food item (top-1 today; more once multi-item lands)."""

    rank: int = Field(ge=1, description="1 = the model's top prediction")
    food101_class: str = Field(description="Raw Food-101 class, e.g. 'grilled_salmon'")
    display_name: str = Field(description="Human-friendly label, e.g. 'Grilled Salmon'")
    confidence: float = Field(ge=0.0, le=1.0)
    portion: Portion | None = Field(
        default=None,
        description="Portion estimate (FR05); null if portion estimation is disabled/unavailable",
    )


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
