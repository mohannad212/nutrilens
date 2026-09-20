"""Prototype: find an ISO/IEC 7810 ID-1 card (85.60 x 53.98 mm) by geometry alone.

No training. A card is a rigid rectangle whose aspect ratio is fixed at 1.5857,
so the detector is: find quadrilaterals -> keep the ones whose rotated bounding
box has that ratio -> report the long edge in pixels.

The whole risk is FALSE POSITIVES: a wrong "card" is a confidently wrong ruler,
which is worse than no ruler. So this is tuned to be strict and is measured
against photos containing no card at all.
"""
import cv2
import numpy as np

CARD_LONG_MM, CARD_SHORT_MM = 85.60, 53.98
CARD_RATIO = CARD_LONG_MM / CARD_SHORT_MM  # 1.5857


def find_cards(bgr, ratio_tol=0.12, min_frac=0.0008, max_frac=0.08, solidity_min=0.90):
    h, w = bgr.shape[:2]
    frame = float(h * w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    out = []
    # Multiple thresholds: a card may be lighter or darker than the surface, and
    # a single Canny setting misses low-contrast cards.
    edge_sets = [cv2.Canny(gray, lo, hi) for lo, hi in ((30, 90), (50, 150), (75, 200))]
    for edges in edge_sets:
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_frac * frame <= area <= max_frac * frame):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            # Solidity guards against ragged edge fragments that happen to have
            # four vertices.
            hull_area = cv2.contourArea(cv2.convexHull(approx))
            if hull_area <= 0 or area / hull_area < solidity_min:
                continue
            (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(approx)
            if min(rw, rh) < 12:
                continue
            ratio = max(rw, rh) / max(min(rw, rh), 1e-6)
            if abs(ratio - CARD_RATIO) / CARD_RATIO > ratio_tol:
                continue
            out.append(
                {
                    "long_px": float(max(rw, rh)),
                    "ratio": float(ratio),
                    "area_px": float(area),
                    "box": np.int32(cv2.boxPoints(((_cx, _cy), (rw, rh), _ang))),
                }
            )

    # De-duplicate across the three edge settings (same card found repeatedly).
    kept = []
    for d in sorted(out, key=lambda d: -d["area_px"]):
        if all(abs(d["long_px"] - k["long_px"]) / k["long_px"] > 0.15 for k in kept):
            kept.append(d)
    return kept
