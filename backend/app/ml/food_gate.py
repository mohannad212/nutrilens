"""
app/ml/food_gate.py
======================

WHY THIS FILE EXISTS
---------------------
`classifier.py` serves a **closed-set** model: EfficientNet-B4 fine-tuned on
Food-101, whose softmax is defined over exactly 101 food classes. Such a model
cannot express "this is not food" — the probabilities must sum to 1 across the
101 foods no matter what the camera is pointed at. Photograph a laptop and it
answers "apple pie", because apple pie is the closest thing it has a word for.

This module is the missing half: an **open-set pre-filter** that decides whether
the photo is of food *at all*, before the specialist is asked which food it is.

WHAT PROBLEM IT SOLVES
------------------------
The obvious guard — reject when the top-1 confidence is low — does not work, and
we have the counter-example to prove it: one laptop photo scored 0.05 (correctly
rejected by a floor) while a second, from a closer angle, scored **0.46** on
"apple pie" with the runner-up at 0.09. A wide margin and a high-ish score, on an
object that is unambiguously not food. Confidence measures *which of the 101*,
never *whether any of the 101*; no threshold on it can separate the two.

The signal has to come from a model that has actually seen non-food. ImageNet-1k
has 1000 general classes — including `laptop`, `computer keyboard`, `space bar`
and `notebook` — alongside ~45 food classes and the tableware that surrounds a
meal. Asking it "how much probability mass lands on food-or-dining?" answers the
open-set question directly. On the 0.46 laptop photo it puts 0.79 on `space bar`
and effectively nothing on food.

HOW IT WORKS INTERNALLY
--------------------------
- `mobilenet_v3_large` with ImageNet-1k weights: small (~22 MB, already in the
  torch hub cache as a dependency of the portion pipeline) and a few milliseconds
  per image, so the gate is cheap next to B4 inference.
- `food_probability(image)` sums the softmax mass over two index sets:
  `_FOOD_CLASSES`   — actual edibles (guacamole … eggnog).
  `_DINING_CLASSES` — the context a meal is photographed in (plate, soup bowl,
                      dining table, restaurant, wok, frying pan …). Included
                      because many real dishes are not themselves ImageNet
                      classes — a plate of shawarma has no `shawarma` label, but
                      `plate` and `dining table` still fire.
- The caller compares that sum against `settings.food_gate_min_prob` and raises
  `NoFoodDetectedError` below it.
- Indices are verified against the shipped category names at load time, so a
  future torchvision reordering fails loudly here instead of silently gating on
  the wrong classes.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- A second pretrained classifier (chosen): needs no new data, no training, and no
  weights we do not already have. It is a genuine open-set signal because the
  model was trained on the non-food world.
- A trained binary food/not-food head: stronger, but needs a labelled non-food
  dataset and a training run — outside this build's scope.
- A dedicated OOD detector (ODIN, energy-based scoring, Mahalanobis): the
  academically rigorous answer, and the right citation for "how would you improve
  this?". Each needs calibration data and tuning that a two-model pipeline does
  not.
- Reusing the portion segmenter's mask coverage: rejected — it is trained to find
  food-shaped regions in food photos and was never shown non-food, so it has the
  same closed-set blind spot as the classifier.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why not just raise the confidence threshold instead of adding a model?"
A: See the 0.46 laptop above. Raising the threshold to catch it would sit at or
   above `confidence_threshold` (0.60), which would delete the FR04 pick-list
   entirely and start refusing legitimate hard food photos. The two questions are
   independent, so they need independent evidence.

Q: "Isn't ImageNet also a closed set?"
A: Yes, but a closed set that *contains the negative class*. We are not asking it
   to name the dish; we are asking whether the scene resembles food-or-dining at
   all, and its 1000 classes cover enough of the non-food world to answer that.

Q: "What if a dish isn't in ImageNet's food classes?"
A: That is what `_DINING_CLASSES` is for — the plate, bowl, table and restaurant
   context fires even when the dish itself has no ImageNet label. The threshold
   is deliberately permissive (default 0.10) because a false rejection of a real
   meal is far worse for the user than occasionally admitting a non-meal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

# ImageNet-1k class indices. The ordering is fixed by the dataset, and
# `load_food_gate` asserts the anchors below still hold before trusting them.
#
# Edibles: the contiguous food block, minus 958 ("hay"), which sits inside it but
# is not food in any sense this app cares about.
_FOOD_CLASSES = frozenset(range(924, 958)) | frozenset(range(959, 970))

# The furniture of a meal. A dish absent from ImageNet still gets photographed on
# one of these, which is what keeps the gate from refusing unfamiliar cuisines.
_DINING_CLASSES = frozenset(
    {
        923,  # plate
        809,  # soup bowl
        659,  # mixing bowl
        532,  # dining table
        762,  # restaurant
        868,  # tray
        504,  # coffee mug
        505,  # coffeepot
        647,  # measuring cup
        909,  # wok
        567,  # frying pan
        738,  # pot
        521,  # Crock Pot
        849,  # teapot
    }
)

# Sanity anchors: (index, expected category name). Guards against a torchvision
# release renumbering the categories under us.
_ANCHORS = ((924, "guacamole"), (963, "pizza"), (923, "plate"), (620, "laptop"))

_ACCEPT_CLASSES = _FOOD_CLASSES | _DINING_CLASSES


@dataclass
class GateVerdict:
    """What the gate saw, kept for the error `details` and for logs."""

    food_probability: float
    top_label: str
    top_probability: float


class FoodGate:
    """Answers "is this a photo of food?" independently of Food-101."""

    def __init__(self, model: Any, transform: Any, categories: list[str], device: str):
        self._model = model
        self._transform = transform
        self._categories = categories
        self._device = torch.device(device)

    def inspect(self, image: Image.Image) -> GateVerdict:
        """Summed food-or-dining probability, plus the single best label."""
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor)[0], dim=0)

        food_mass = float(sum(probs[i].item() for i in _ACCEPT_CLASSES))
        top_idx = int(torch.argmax(probs).item())
        return GateVerdict(
            food_probability=round(food_mass, 4),
            top_label=self._categories[top_idx],
            top_probability=round(float(probs[top_idx].item()), 4),
        )


def load_food_gate(device: str = "cpu") -> FoodGate:
    """Build the gate from torchvision's cached ImageNet weights.

    Raises `AssertionError` if the ImageNet category ordering no longer matches
    the indices above — better to fail at startup than to gate on wrong classes.
    """
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    categories: list[str] = list(weights.meta["categories"])

    for index, expected in _ANCHORS:
        if categories[index] != expected:
            raise AssertionError(
                f"ImageNet category {index} is {categories[index]!r}, expected {expected!r} — "
                "the food-gate index sets need revisiting."
            )

    model = mobilenet_v3_large(weights=weights)
    model.eval()
    model.to(torch.device(device))
    return FoodGate(model, weights.transforms(), categories, device)
