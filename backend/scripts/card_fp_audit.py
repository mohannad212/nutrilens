"""False-positive audit for the credit-card reference detector (backlog §1).

WHY THIS SCRIPT EXISTS
A wrongly-detected card is a *confidently wrong ruler* — strictly worse than no
ruler, because the pipeline's honest fallback is a wider band, not a bad number.
The card detector's thresholds are therefore tuned for zero false positives, and
the backlog's explicit instruction is: do not ship, or loosen, without a
false-positive check over the whole weighed set. This is that check, kept as a
script so it can be re-run after any threshold change rather than remembered.

Run:  .venv/bin/python scripts/card_fp_audit.py
Every photo in data/weighed/images is a NEGATIVE — none contains a card — so any
detection at the shipped settings is a regression and the script exits non-zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.ml.portion.reference_detector import CardReferenceDetector  # noqa: E402

IMAGES = Path(__file__).resolve().parents[1] / "data" / "weighed" / "images"


def main() -> int:
    shipped = CardReferenceDetector(
        ratio_tol=settings.card_ratio_tol, min_long_px=settings.card_min_long_px
    )
    # The setting that was measured to FAIL, kept as the contrast case: at 12%
    # tolerance the kitchen scale's display panel (ratio 1.42) reads as a card.
    loose = CardReferenceDetector(ratio_tol=0.12, min_long_px=12.0)

    photos = sorted(IMAGES.glob("*.jpg"))
    if not photos:
        print(f"No images in {IMAGES}")
        return 1

    print(f"Card false-positive audit over {len(photos)} weighed photos (all negatives)")
    print(f"shipped: ratio_tol={settings.card_ratio_tol} min_long_px={settings.card_min_long_px}")
    print(f"{'photo':26s} {'shipped':>10s} {'loose(12%)':>12s}")
    failures = 0
    for p in photos:
        img = Image.open(p)
        s_hits, l_hits = shipped.find_cards(img), loose.find_cards(img)
        if s_hits:
            failures += 1
        note = f" <- FALSE POSITIVE ratio={s_hits[0]['ratio']:.3f}" if s_hits else ""
        print(f"{p.name:26s} {len(s_hits):>10d} {len(l_hits):>12d}{note}")

    if failures:
        print(f"\nFAIL: {failures} false positive(s) at the shipped thresholds.")
        return 1
    print("\nPASS: zero false positives at the shipped thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
