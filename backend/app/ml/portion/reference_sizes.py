"""
app/ml/portion/reference_sizes.py
====================================

WHY THIS FILE EXISTS
---------------------
Stage 2 of the portion pipeline (`05 §4`) is the *strongest* scale signal: find a
known-size everyday object in the frame (a fork, a coin, a credit card...) and,
because we know its real dimension in centimetres, one measurement in pixels
gives a precise `cm_per_px`. This file is the lookup from a detected reference
object to its real-world size, plus the priority order that decides which
reference to trust when several are visible.

WHAT PROBLEM IT SOLVES
------------------------
`cm_per_px = real_size_cm / detected_pixel_length`. That formula needs one number
per reference class — the real size — and it must be the *right* dimension
(a fork's length, a credit card's long edge, a coin's diameter). Getting the
dimension or the value wrong silently scales every portion estimate. Centralising
these here (with the exact ISO figure for a credit card, region note for coins)
makes the physical assumptions auditable in one place, exactly like the container
priors do for Stage 1.

HOW IT WORKS INTERNALLY
--------------------------
- `REFERENCE_SIZES` maps each reference class from `05 §4` to a
  `ReferenceSize(real_cm, dimension, priority)`. `dimension` is a human label of
  *which* measurement `real_cm` refers to ("long edge", "length", "diameter") so
  the detector/estimator measures the matching pixel span.
- `priority` encodes the reliability order from `05 §4`
  (`credit_card > smartphone > coin > fork/spoon/knife > chopsticks > hand`);
  lower number = more reliable. `best_reference()` picks the most reliable
  reference among those actually detected.
- `size_for()` is a non-raising accessor (returns `None` for unknown labels) so a
  detector that emits a class we don't model can't crash the pipeline.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Explicit priority integers (chosen) over relying on dict insertion order: the
  reliability ranking is a deliberate product/physics decision from `05 §4` that a
  reviewer should see stated numerically, not inferred from how the dict happens
  to be written.
- A `dimension` label stored alongside the value: prevents the classic bug of
  measuring a fork's *width* against its real *length* — the label documents the
  contract the detector must satisfy.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is the credit card the most reliable reference?"
A: Its long edge is standardised worldwide to 8.56 cm (ISO/IEC 7810 ID-1), so
   there is essentially zero size variance — unlike a hand (varies per person) or
   a coin (varies per region/denomination).

Q: "Two references are in frame and disagree — what happens?"
A: `05 §4`: use the highest-priority one; if the next agrees within 15% average
   them, otherwise trust the more reliable one alone. The averaging lives in the
   estimator; this file just supplies the sizes and the priority order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceSize:
    """Real-world size of a known reference object used to calibrate scale."""

    real_cm: float
    dimension: str  # which measurement real_cm refers to, e.g. "long edge"
    priority: int  # lower = more reliable (see 05 §4 ordering)


# Known real-world dimensions from `05 §4`. `priority` encodes the reliability
# order stated there: credit_card > smartphone > coin > fork/spoon/knife >
# chopsticks > hand (lower number = prefer it when several are detected).
REFERENCE_SIZES: dict[str, ReferenceSize] = {
    "credit_card": ReferenceSize(8.56, "long edge", 1),  # ISO/IEC 7810 ID-1
    "smartphone": ReferenceSize(14.7, "long edge", 2),  # configurable per device
    "coin": ReferenceSize(2.4, "diameter", 3),  # default US quarter
    "fork": ReferenceSize(19.0, "length", 4),  # Western table fork
    "spoon": ReferenceSize(18.0, "length", 4),  # tablespoon
    "knife": ReferenceSize(21.0, "length", 4),  # table knife
    "chopsticks": ReferenceSize(24.0, "length", 5),
    "hand": ReferenceSize(8.4, "width across knuckles", 6),  # weakest
}


def size_for(reference_class: str) -> ReferenceSize | None:
    """Return the real size for a reference class, or None if we don't model it."""
    return REFERENCE_SIZES.get(reference_class)


def best_reference(detected_classes: list[str]) -> str | None:
    """Pick the most reliable reference among those detected (lowest priority int).

    Returns None if none of the detected classes is a known reference — the
    estimator then falls back to the container prior (`05 §4`).
    """
    known = [c for c in detected_classes if c in REFERENCE_SIZES]
    if not known:
        return None
    return min(known, key=lambda c: REFERENCE_SIZES[c].priority)
