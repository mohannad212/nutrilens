"""
app/services/recognition_service.py
======================================

WHY THIS FILE EXISTS
---------------------
The `/recognise` route should stay thin — parse the upload, return JSON. The
actual recognition logic (validate the image, run the model, apply the FR04
confidence rule, shape the response) lives here so it is testable without an
HTTP client and so the route file reads as a one-line map of "what the endpoint
does".

WHAT PROBLEM IT SOLVES
------------------------
It is the single place that turns "raw uploaded bytes + a loaded classifier"
into the exact `RecognitionResponse` contract from `03 §4`, including the FR04
decision: *is the model confident enough to commit to a top-1, or should it hand
the user a short pick-list of alternatives?* Keeping that decision here (not in
the route, not in the model) means the threshold rule is unit-testable with a
stub classifier and no real weights.

HOW IT WORKS INTERNALLY
--------------------------
`recognise(raw, content_type, classifier)`:
  1. `validate_and_load_image` (in `preprocess`) enforces FR03 and returns a
     decoded in-memory RGB image — or raises the specific 4xx.
  2. `classifier.predict(image, top_k=3)` returns the top-3 `(class, confidence)`
     pairs, highest first.
  3. Out-of-distribution gate: if even the top-1 falls below
     `settings.no_food_confidence_floor` (default 0.25), raise
     `NoFoodDetectedError` (400). Food-101 has no "not food" class, so the model
     always names *some* food; a near-uniform distribution is the only available
     evidence that the subject is not a meal.
  4. The top-1 becomes the single `items[0]` (rank 1). If a portion estimator is
     wired and `settings.portion_enabled`, the M4 pipeline (`05`) estimates its
     mass range and attaches a `portion` object; otherwise `portion` is null and
     the classification half is unaffected.
  5. FR04 rule: if the top-1 confidence is below `settings.confidence_threshold`
     (default 0.60), set `low_confidence = True` and fill `alternatives` with the
     top-3 candidates for the user to choose from; otherwise `alternatives` is
     empty.
  6. A fresh `request_id` (uuid4 hex) is attached for log correlation.
The image exists only as bytes -> PIL -> tensor in memory and is never written
to disk (M3 DoD).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Threshold logic in the service (chosen) rather than in the classifier: the
  model's job is to output probabilities; deciding what confidence is "good
  enough" is a product rule (`03 §4`, FR04) that may change independently of the
  model, so it lives in application code.
- Returning up to three candidates as `alternatives` (including the top-1) when
  unsure: matches the contract's "populate alternatives with up to 3 candidates
  for the user to choose from" — the app shows a short pick-list instead of
  silently committing to a shaky guess.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why return the top-1 in `items` even when low_confidence is true?"
A: The contract always returns a best guess in `items`; `low_confidence` +
   `alternatives` tell the app to *also* offer a pick-list. The app decides
   whether to auto-select the top-1 or ask the user (`07`).

Q: "How does the app know a photo isn't food at all? Food-101 has no such class."
A: It infers it from the *shape* of the output. A closed-set softmax over 101
   food classes must assign its mass to some food, so an unrelated subject (a
   laptop, a face) produces a diffuse distribution — top-1 around 0.1-0.2 rather
   than the 0.6-0.95 a real dish scores. `no_food_confidence_floor` turns that
   into a rejection. It is a heuristic, not a true open-set method: the rigorous
   alternatives are an explicit background class trained on non-food images, or
   an OOD detector (ODIN / energy-based scoring). Both need data or training
   changes beyond this build, and are the honest answer to "how would you improve
   it?".

Q: "Why not just raise `confidence_threshold` instead of adding a second one?"
A: They answer different questions. `confidence_threshold` (0.60) asks "am I sure
   *which* food this is?" and its failure mode is a pick-list. The floor asks "is
   this food at all?" and its failure mode is a retake. Collapsing them would
   force one response to two different problems: raise 0.60 and hard food photos
   get refused; lower the floor toward it and non-food gets a pick-list of foods.

Q: "Where does display_name come from?"
A: Derived from the Food-101 class string here (`grilled_salmon` ->
   'Grilled Salmon') so the client never has to prettify snake_case.

COMMON STUDENT MISTAKES
--------------------------
- Comparing confidence against a hard-coded 0.6 instead of
  `settings.confidence_threshold` — the threshold is configurable per
  environment and must be read from settings.
- Writing the upload to a temp file "to open it with PIL" — decode from the
  in-memory bytes instead; the DoD forbids touching disk.
"""

from __future__ import annotations

import uuid

from PIL import Image

from app.config import settings
from app.core.errors import NoFoodDetectedError
from app.ml.classifier import FoodClassifier
from app.ml.food_gate import FoodGate
from app.ml.portion import PortionEstimator, describe_scale_source
from app.ml.preprocess import validate_and_load_image
from app.schemas.recognition import Alternative, Portion, Prediction, RecognitionResponse

# How many candidates to offer when the model is unsure (FR04: "up to 3").
_TOP_K = 3


def _display_name(food101_class: str) -> str:
    """`grilled_salmon` -> `Grilled Salmon` for a user-facing label."""
    return food101_class.replace("_", " ").title()


def _estimate_portion(
    image: Image.Image,
    food_class: str,
    estimator: PortionEstimator | None,
) -> Portion | None:
    """Run the M4 portion pipeline for the top-1 item, or return None (FR05).

    Portion is deliberately optional: it is skipped when disabled
    (`settings.portion_enabled`) or when no estimator is wired, so the
    classification half (FR03/FR04) never depends on it. The estimator itself
    degrades gracefully (`05 §10`) — a `shape_only` range is a valid result, not
    a failure.
    """
    if estimator is None or not settings.portion_enabled:
        return None
    est = estimator.estimate(image, food_class)
    return Portion(
        min_g=est.min_g,
        max_g=est.max_g,
        point_g=est.point_g,
        method=est.method,
        reference=est.reference,
        scale_source=describe_scale_source(est.method, est.reference),
    )


def recognise(
    raw: bytes,
    content_type: str | None,
    classifier: FoodClassifier,
    portion_estimator: PortionEstimator | None = None,
    food_gate: FoodGate | None = None,
) -> RecognitionResponse:
    """Validate an uploaded image, classify it, estimate the portion, apply FR04.

    Raises the image-validation `AppError`s (413/400) from `preprocess` and
    `InferenceFailedError` (500) from the classifier; all become the shared
    error envelope. The image (and any mask) is processed entirely in memory.
    """
    image = validate_and_load_image(raw, content_type)

    # Open-set gate, ahead of the specialist: Food-101 has no "not food" class,
    # so this is the only stage that can answer "is this a meal at all?". Skipped
    # when the gate isn't loaded (fail open — see `deps.get_food_gate`).
    if food_gate is not None:
        verdict = food_gate.inspect(image)
        if verdict.food_probability < settings.food_gate_min_prob:
            raise NoFoodDetectedError(
                details={
                    "food_probability": verdict.food_probability,
                    "threshold": settings.food_gate_min_prob,
                    "looks_like": verdict.top_label,
                }
            )

    predictions = classifier.predict(image, top_k=_TOP_K)  # [(class, conf), ...]

    top_class, top_conf = predictions[0]

    # Backstop for anything the gate let through: a distribution this flat means
    # the specialist has no real opinion either. Checked before portion
    # estimation — no point measuring the mass of something we don't call food.
    if top_conf < settings.no_food_confidence_floor:
        raise NoFoodDetectedError(
            details={
                "top_confidence": round(top_conf, 4),
                "floor": settings.no_food_confidence_floor,
            }
        )
    items = [
        Prediction(
            rank=1,
            food101_class=top_class,
            display_name=_display_name(top_class),
            confidence=top_conf,
            portion=_estimate_portion(image, top_class, portion_estimator),
        )
    ]

    # FR04: below the threshold, offer the top-k as a pick-list instead of
    # silently committing to a low-confidence guess.
    low_confidence = top_conf < settings.confidence_threshold
    alternatives = (
        [
            Alternative(
                food101_class=cls,
                display_name=_display_name(cls),
                confidence=conf,
            )
            for cls, conf in predictions
        ]
        if low_confidence
        else []
    )

    return RecognitionResponse(
        request_id=uuid.uuid4().hex,
        items=items,
        low_confidence=low_confidence,
        alternatives=alternatives,
    )
