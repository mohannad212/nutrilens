"""
app/api/recognise.py
=======================

WHY THIS FILE EXISTS
---------------------
This is the HTTP surface for FR03/FR04 — the single `POST /recognise` endpoint
the mobile Camera-Capture screen uploads a photo to. Like the auth router, it is
deliberately thin: accept the multipart upload, hand the bytes to
`recognition_service`, return the `RecognitionResponse`.

WHAT PROBLEM IT SOLVES
------------------------
`/recognise` is the one endpoint that takes `multipart/form-data` (a file) rather
than JSON (`03 §1/§4`). This file owns that upload plumbing — declaring the
`image` form field, reading its bytes, and requiring authentication — while
keeping all recognition logic in the service so the route stays a one-liner of
orchestration.

HOW IT WORKS INTERNALLY
--------------------------
- `image: UploadFile = File(...)` tells FastAPI to parse a multipart file field
  named `image`; a request with no file auto-returns 422 (missing required field)
  through the shared validation envelope — the service never runs.
- `user = Depends(get_current_user)` enforces auth: a missing/invalid token is a
  401 before any image work happens. The user isn't used further in M3, but the
  endpoint is protected per `03 §4` ("**Auth.**").
- `classifier = Depends(get_classifier)` injects the model loaded once at startup
  (or a stub in tests).
- We `await image.read()` to get the raw bytes in memory and pass them, with the
  declared `content_type`, to `recognition_service.recognise`. The `UploadFile`
  is spooled by Starlette in memory/temp and released when the request ends; we
  never persist it ourselves (M3 DoD: no image written to disk).

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- `UploadFile` (chosen) over `bytes = File(...)`: `UploadFile` streams and
  exposes `content_type`/`filename`, and doesn't force the whole body into memory
  before the handler runs — better for 10 MB photos.
- All validation/inference in the service: the route can't accidentally diverge
  from the contract, and the logic is testable without a TestClient.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why multipart here when every other endpoint is JSON?"
A: You can't put a raw binary image in a JSON body without base64-bloating it by
   ~33%. `multipart/form-data` is the standard, efficient way to upload files
   (`03 §1`).

Q: "Is the endpoint authenticated?"
A: Yes — `Depends(get_current_user)`. `03 §4` marks `/recognise` as auth-only;
   an anonymous caller gets 401 before the image is even read.

COMMON STUDENT MISTAKES
--------------------------
- Reading the file synchronously (`image.file.read()`) inside an async route,
  blocking the event loop — use `await image.read()`.
- Forgetting the auth dependency, leaving an expensive ML endpoint open to
  unauthenticated callers.
"""

from fastapi import APIRouter, Depends, File, UploadFile

from app.db.models import User
from app.deps import get_classifier, get_current_user, get_portion_estimator
from app.ml.classifier import FoodClassifier
from app.ml.portion import PortionEstimator
from app.schemas.recognition import RecognitionResponse
from app.services.recognition_service import recognise as recognise_service

router = APIRouter()


@router.post("/recognise", response_model=RecognitionResponse)
async def recognise(
    image: UploadFile = File(...),
    _user: User = Depends(get_current_user),
    classifier: FoodClassifier = Depends(get_classifier),
    portion_estimator: PortionEstimator | None = Depends(get_portion_estimator),
) -> RecognitionResponse:
    raw = await image.read()
    return recognise_service(raw, image.content_type, classifier, portion_estimator)
