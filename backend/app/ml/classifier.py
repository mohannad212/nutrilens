"""
app/ml/classifier.py
=======================

WHY THIS FILE EXISTS
---------------------
This is the serving side of the ML track. The offline training pipeline
(`04-ml-model.md`) produced a TorchScript file, `food101_efficientnet_b4.pth`,
that reached 91.4% top-1 on the Food-101 test split. This file wraps that
artifact in a small `FoodClassifier` class the backend can call: give it a
photo, get back the top-k `(class, confidence)` predictions. It is the only
place in the app that touches PyTorch.

WHAT PROBLEM IT SOLVES
------------------------
Loading a 69 MB model and running a neural network is expensive and stateful —
it must happen exactly once, at process startup, not per request. It also needs
the *same* preprocessing the model saw at evaluation time or accuracy collapses
(`preprocess.py` owns that). This class encapsulates both: load-once weights +
labels, and a `predict()` that funnels every image through the pinned eval
transform before inference. Everything downstream (`recognition_service`) deals
in plain `(class, confidence)` tuples and never imports torch.

HOW IT WORKS INTERNALLY
--------------------------
- `__init__` loads the TorchScript module with `torch.jit.load(..., map_location
  =device).eval()` and the `labels_food101.json` index->class map, and builds the
  eval transform once. TorchScript means we don't even need the Python model-
  definition code at serve time — the architecture travels inside the file.
- `predict(pil_image, top_k)` applies the transform, adds a batch dimension,
  runs a softmax over the logits under `torch.inference_mode()` (no autograd
  graph, less memory, faster), and returns the top-k classes with rounded
  probabilities, highest first.
- `load_classifier(...)` is an `lru_cache`d factory so the heavy load happens
  once per (weights, labels, device) per process — the FastAPI lifespan calls it
  at startup, and repeated calls (e.g. across tests in one process) are free.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- TorchScript (`torch.jit.load`) as chosen in `04 §7`: portable and self-
  contained (no need to re-declare the EfficientNet-B4 architecture at serve
  time, no torchvision-version-mismatch risk on the class definition), which is
  exactly what a deploy target wants.
- `torch.inference_mode()` over `torch.no_grad()`: strictly faster/lighter for
  pure inference — it also disables view tracking, and we never train here.
- A cached module-level factory over constructing the classifier inline in the
  lifespan: lets tests build the app repeatedly without paying the load cost
  each time, and keeps "how the model is loaded" in one testable function.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why load at startup instead of per request?"
A: Loading is ~1–2 s and allocates the whole network; doing it per request would
   add that latency to every call and thrash memory. Load once, reuse the warm
   model for the life of the process (`04 §8`).

Q: "CPU or GPU?"
A: `settings.device` — CPU by default (the eval VM has no GPU, and B4 CPU
   inference is ~0.3–1.0 s/image, within the 5 s NFR). Set `DEVICE=cuda` to use a
   GPU with no code change.

Q: "What if inference itself throws?"
A: `predict` wraps the forward pass and raises `InferenceFailedError` (500,
   INFERENCE_FAILED) so `/recognise` returns the contract's error envelope
   instead of a bare stack trace (`03 §4/§9`).

COMMON STUDENT MISTAKES
--------------------------
- Forgetting `.eval()` — leaves dropout/BN in training mode and predictions
  become noisy and non-deterministic.
- Rebuilding the transform or reloading the model inside `predict` — silently
  turns a fast call into a slow one.
- Indexing the label map with an int when the JSON keys are strings
  (`{"0": "apple_pie"}`) — `idx_to_class[str(i)]`, not `[i]`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from app.core.errors import InferenceFailedError
from app.ml.preprocess import build_eval_transform


class FoodClassifier:
    """Loads the trained Food-101 model once and predicts top-k classes."""

    def __init__(self, weights_path: str, labels_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        # TorchScript carries the architecture, so no model definition is needed.
        self.model = torch.jit.load(weights_path, map_location=self.device).eval()
        # {"0": "apple_pie", "1": "baby_back_ribs", ...} — index -> class name.
        self.idx_to_class: dict[str, str] = json.loads(Path(labels_path).read_text())
        self.transform = build_eval_transform()

    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k `(food101_class, confidence)` pairs, highest first.

        `image` must already be a decoded RGB `PIL.Image` (see
        `preprocess.validate_and_load_image`). Confidences are softmax
        probabilities rounded to 4 dp. Any failure in the forward pass is
        surfaced as `InferenceFailedError` (500) for the shared error envelope.
        """
        k = min(top_k, len(self.idx_to_class))
        try:
            # inference_mode: no autograd graph, less memory, faster than no_grad.
            with torch.inference_mode():
                x = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, 380, 380)
                probs = torch.softmax(self.model(x), dim=1)[0]
                confidences, indices = probs.topk(k)
        except Exception as exc:  # torch/runtime failure -> contract's 500
            raise InferenceFailedError() from exc

        return [
            (self.idx_to_class[str(idx.item())], round(conf.item(), 4))
            for conf, idx in zip(confidences, indices, strict=True)
        ]


@lru_cache(maxsize=1)
def load_classifier(weights_path: str, labels_path: str, device: str) -> FoodClassifier:
    """Build (once per process) and return a ready `FoodClassifier`.

    Cached so the FastAPI lifespan's startup load isn't repeated when the app is
    reconstructed (e.g. per-test). Raises whatever `torch.jit.load` raises if the
    weights are missing/corrupt — the caller (lifespan) decides how to handle a
    model that won't load.
    """
    return FoodClassifier(weights_path, labels_path, device)
