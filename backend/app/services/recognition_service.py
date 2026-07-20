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
  3. The top-1 becomes the single `items[0]` (rank 1).
  4. FR04 rule: if the top-1 confidence is below `settings.confidence_threshold`
     (default 0.60), set `low_confidence = True` and fill `alternatives` with the
     top-3 candidates for the user to choose from; otherwise `alternatives` is
     empty.
  5. A fresh `request_id` (uuid4 hex) is attached for log correlation.
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

from app.config import settings
from app.ml.classifier import FoodClassifier
from app.ml.preprocess import validate_and_load_image
from app.schemas.recognition import Alternative, Prediction, RecognitionResponse

# How many candidates to offer when the model is unsure (FR04: "up to 3").
_TOP_K = 3


def _display_name(food101_class: str) -> str:
    """`grilled_salmon` -> `Grilled Salmon` for a user-facing label."""
    return food101_class.replace("_", " ").title()


def recognise(
    raw: bytes,
    content_type: str | None,
    classifier: FoodClassifier,
) -> RecognitionResponse:
    """Validate an uploaded image, classify it, and apply the FR04 rule.

    Raises the image-validation `AppError`s (413/400) from `preprocess` and
    `InferenceFailedError` (500) from the classifier; all become the shared
    error envelope. The image is processed entirely in memory.
    """
    image = validate_and_load_image(raw, content_type)
    predictions = classifier.predict(image, top_k=_TOP_K)  # [(class, conf), ...]

    top_class, top_conf = predictions[0]
    items = [
        Prediction(
            rank=1,
            food101_class=top_class,
            display_name=_display_name(top_class),
            confidence=top_conf,
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
