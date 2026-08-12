"""
tests/test_m3_recognise.py
=============================

WHY THIS FILE EXISTS
---------------------
M3's Definition of Done (`09-build-phases.md`) requires: a photographed known
food returns the correct top-1 (or a top-3 including it), oversized/low-res
images return the correct 4xx, and **no image is written to disk** (verified in
test). This file verifies each of those over the real `POST /recognise` HTTP
route, plus the FR04 low-confidence rule and the endpoint's auth requirement.

HOW IT AVOIDS THE 69 MB MODEL
-------------------------------
`/recognise` depends on a loaded `FoodClassifier`. Loading the real TorchScript
weights in a unit test would be slow and would couple the test to a gitignored
artifact, so these tests override the `get_classifier` dependency with a tiny
`StubClassifier` that returns canned `(class, confidence)` predictions. That
lets us drive every branch (confident vs unsure) deterministically and test the
service/route wiring — not PyTorch, which is validated separately by the
training pipeline's own eval (91.4% top-1).

WHAT EACH TEST VALIDATES
--------------------------
- test_recognise_confident_returns_top1: happy path — a valid photo yields 200,
  the correct top-1 in `items`, `low_confidence=false`, and no alternatives.
- test_recognise_low_confidence_returns_alternatives: FR04 — when the top-1 is
  below the threshold, `low_confidence=true` and `alternatives` is populated.
- test_oversized_image_returns_413 / _low_res_returns_413... : FR03 validation
  maps to the exact 4xx + canonical error code.
- test_bad_format_returns_400, test_missing_image_returns_422: the remaining
  bad-input paths.
- test_requires_auth_401: an anonymous caller is rejected before any image work.
- test_no_image_written_to_disk: the DoD's disk guarantee — the request path
  never calls the obvious image-persistence APIs.
"""

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from app.deps import get_classifier, get_current_user, get_food_gate
from app.ml.food_gate import GateVerdict
from app.main import create_app

RECOGNISE_URL = "/api/v1/recognise"

CONFIDENT_PREDS = [("grilled_salmon", 0.91), ("sashimi", 0.05), ("tuna_tartare", 0.02)]
UNSURE_PREDS = [("caesar_salad", 0.42), ("greek_salad", 0.31), ("bruschetta", 0.10)]


class StubClassifier:
    """Stands in for the real FoodClassifier — returns canned predictions."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, image, top_k=3):
        return self.predictions[:top_k]


def _make_client(predictions=CONFIDENT_PREDS, authed=True) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: StubClassifier(predictions)
    if authed:
        # Bypass JWT/DB — the route only needs *a* user to exist.
        app.dependency_overrides[get_current_user] = lambda: object()
    return TestClient(app)


def _encode(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _jpeg_bytes(width=640, height=480) -> bytes:
    """A realistic photo: textured (sharp) and evenly lit — passes every gate."""
    rng = np.random.default_rng(0)
    arr = rng.integers(60, 200, (height, width, 3), dtype=np.uint8)
    return _encode(Image.fromarray(arr, "RGB"))


def _dark_jpeg(width=640, height=480) -> bytes:
    """Under-exposed photo — mean luminance well below the floor."""
    return _encode(Image.new("RGB", (width, height), (8, 8, 8)))


def _blurry_jpeg(width=640, height=480) -> bytes:
    """A textured image driven out of focus — low Laplacian variance."""
    rng = np.random.default_rng(1)
    arr = rng.integers(60, 200, (height, width, 3), dtype=np.uint8)
    blurred = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(8))
    return _encode(blurred)


def _upload(client, data: bytes, filename="food.jpg", content_type="image/jpeg"):
    return client.post(RECOGNISE_URL, files={"image": (filename, data, content_type)})


# --------------------------- happy paths (FR03/FR04) --------------------------
def test_recognise_confident_returns_top1():
    client = _make_client(CONFIDENT_PREDS)
    resp = _upload(client, _jpeg_bytes())
    assert resp.status_code == 200

    body = resp.json()
    assert body["items"][0]["rank"] == 1
    assert body["items"][0]["food101_class"] == "grilled_salmon"
    assert body["items"][0]["display_name"] == "Grilled Salmon"
    assert body["items"][0]["confidence"] == 0.91
    assert body["low_confidence"] is False
    assert body["alternatives"] == []
    assert "request_id" in body


def test_recognise_low_confidence_returns_alternatives():
    client = _make_client(UNSURE_PREDS)
    resp = _upload(client, _jpeg_bytes())
    assert resp.status_code == 200

    body = resp.json()
    # FR04: top-1 (0.42) is below the 0.60 default threshold.
    assert body["low_confidence"] is True
    assert body["items"][0]["food101_class"] == "caesar_salad"
    alt_classes = [a["food101_class"] for a in body["alternatives"]]
    assert alt_classes == ["caesar_salad", "greek_salad", "bruschetta"]


# ------------------------------ bad input (FR03) ------------------------------
def test_oversized_image_returns_413():
    client = _make_client()
    resp = _upload(client, b"x" * (11 * 1024 * 1024))
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_low_res_image_returns_400():
    client = _make_client()
    resp = _upload(client, _jpeg_bytes(width=320, height=240))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "IMAGE_TOO_SMALL"


def test_bad_format_returns_400():
    client = _make_client()
    resp = _upload(client, b"this is not an image", content_type="image/jpeg")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "IMAGE_BAD_FORMAT"


def test_missing_image_returns_422():
    client = _make_client()
    resp = client.post(RECOGNISE_URL)  # no file field
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


# ---------------------- quality gate (FR03 extension) -------------------------
def test_too_dark_image_returns_400():
    client = _make_client()
    resp = _upload(client, _dark_jpeg())
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "IMAGE_POOR_LIGHTING"


def test_blurry_image_returns_400():
    client = _make_client()
    resp = _upload(client, _blurry_jpeg())
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "IMAGE_TOO_BLURRY"


# --------------------------------- auth (03 §4) -------------------------------
def test_requires_auth_401():
    client = _make_client(authed=False)
    resp = _upload(client, _jpeg_bytes())
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


# ------------------------------ DoD: no disk write ----------------------------
def test_no_image_written_to_disk(monkeypatch):
    """The request path must never persist the image (M3 DoD)."""
    from pathlib import Path

    # Build the fixture bytes FIRST (this legitimately saves to an in-memory
    # BytesIO), then install the guards so only the request path is watched.
    data = _jpeg_bytes()
    client = _make_client()

    def _fail(*args, **kwargs):
        raise AssertionError("image was written to disk during /recognise")

    # The two obvious ways to persist an image — guard both.
    monkeypatch.setattr(Image.Image, "save", _fail)
    monkeypatch.setattr(Path, "write_bytes", _fail)

    resp = _upload(client, data)
    assert resp.status_code == 200  # succeeds without ever calling the guards


# ------------------------- open-set gate vs. the floor -------------------------
# Neither the `FoodGate` nor `no_food_confidence_floor` had any coverage before
# this block, which is how a false "No food detected" on a real plate of food
# reached the app. The rule under test: rejection is the *gate's* call, and a low
# Food-101 score is a pick-list (FR04), not a refusal.

FLAT_PREDS = [("carrot_cake", 0.18), ("ravioli", 0.11), ("deviled_eggs", 0.09)]


class StubGate:
    """Stands in for the ImageNet FoodGate with a fixed food probability."""

    def __init__(self, food_probability: float, top_label: str = "plate"):
        self._verdict = GateVerdict(
            food_probability=food_probability,
            top_label=top_label,
            top_probability=0.7,
        )

    def inspect(self, image):
        return self._verdict


def _make_gated_client(predictions, gate):
    app = create_app()
    app.dependency_overrides[get_classifier] = lambda: StubClassifier(predictions)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_food_gate] = lambda: gate
    return TestClient(app)


def test_gate_rejects_non_food():
    """Below `food_gate_min_prob` the upload is refused — the gate's own job."""
    client = _make_gated_client(CONFIDENT_PREDS, StubGate(0.01, top_label="laptop"))
    resp = _upload(client, _jpeg_bytes())

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NO_FOOD_DETECTED"


def test_flat_prediction_past_the_gate_is_a_picklist_not_a_rejection():
    """A real plate whose dish Food-101 lacks must not return NO_FOOD_DETECTED.

    This is the scrambled-eggs case: the gate is confident it is food (0.96),
    while Food-101 has no matching class and spreads its mass thin. Rejecting
    here told the user "no food detected" about a photo of food.
    """
    client = _make_gated_client(FLAT_PREDS, StubGate(0.96))
    resp = _upload(client, _jpeg_bytes())

    assert resp.status_code == 200
    body = resp.json()
    assert body["low_confidence"] is True
    assert [a["food101_class"] for a in body["alternatives"]] == [
        "carrot_cake",
        "ravioli",
        "deviled_eggs",
    ]


def test_floor_still_applies_when_the_gate_is_unavailable():
    """With no gate loaded the floor is the only open-set check left — keep it."""
    client = _make_gated_client(FLAT_PREDS, None)
    resp = _upload(client, _jpeg_bytes())

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NO_FOOD_DETECTED"
