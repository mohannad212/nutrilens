"""
scripts/eval_portion_mape.py
===============================

WHY THIS FILE EXISTS
---------------------
The portion pipeline's headline result for the GP2 report is its **MAPE** — Mean
Absolute Percentage Error of the estimated grams vs. real kitchen-scale weights
(`05 §7, §9`). This script is the measuring tool: point it at a folder of weighed
food photos + a manifest of true weights, and it runs the real `PortionEstimator`
over them and prints the MAPE (plus a per-stage breakdown and range coverage).

WHAT PROBLEM IT SOLVES
------------------------
"How accurate is the portion estimator?" has to be answered with a number on real
data, not vibes. This turns a set of (photo, true grams) pairs into that number,
reproducibly. It deliberately uses the **ground-truth food class from the manifest**
(not the classifier) so the metric measures *portion* error alone — a
misclassification shouldn't pollute the portion score.

HOW IT WORKS INTERNALLY
--------------------------
- Reads a CSV manifest with columns: `image, food_class, true_grams`.
- Builds a `PortionEstimator` loading whatever aux models are in `models/`
  (container classifier, food segmenter); the COCO reference detector is opt-in via
  `--use-reference` since it downloads a model.
- For each row: estimate the portion, compute the absolute percentage error of
  `point_g` vs `true_grams`, and whether the true weight fell inside the predicted
  `[min_g, max_g]` range (**coverage** — the honesty check for a range estimate).
- Aggregates overall MAPE, MAPE per scale `method`, median error, and coverage,
  writes a per-item CSV, and prints a summary.

MANIFEST FORMAT (what you collect)
-----------------------------------
    image,food_class,true_grams
    plate_rice_01.jpg,fried_rice,182
    salmon_fork_02.jpg,grilled_salmon,204
Weigh each single-food dish on a kitchen scale, photograph it (optionally with a
fork/spoon/knife in frame to exercise Stage 2), and record the grams. ~150 rows
give a stable number.

USAGE
------
    python scripts/eval_portion_mape.py \
        --manifest data/weighed/manifest.csv \
        --images-dir data/weighed/images \
        --out artifacts/mape_report.csv [--use-reference]

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Ground-truth food class from the manifest (chosen) over running the classifier:
  isolates portion error from classification error, so the number means what the
  report claims it means.
- Reporting **coverage** alongside MAPE: for a *range* estimate, "how often does the
  band contain the truth" matters as much as the midpoint error — a wide honest
  band with high coverage is the design goal (`05 §9`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow `python scripts/eval_portion_mape.py` from backend/ (see seed scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.config import settings  # noqa: E402
from app.ml.portion import PortionEstimator  # noqa: E402


# ------------------------------ pure metric helpers --------------------------
def absolute_percentage_error(pred_g: float, true_g: float) -> float:
    """|pred − true| / true × 100. Requires true_g > 0."""
    return abs(pred_g - true_g) / true_g * 100.0


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-item results into MAPE, coverage, and a per-method breakdown.

    Each row needs: `ape` (abs % error), `covered` (bool), `method` (str).
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "mape": 0.0, "median_ape": 0.0, "coverage": 0.0, "by_method": {}}
    apes = sorted(r["ape"] for r in rows)
    mape = sum(apes) / n
    median = apes[n // 2] if n % 2 else (apes[n // 2 - 1] + apes[n // 2]) / 2
    coverage = sum(1 for r in rows if r["covered"]) / n * 100.0

    by_method: dict[str, dict] = {}
    for method in sorted({r["method"] for r in rows}):
        sub = [r["ape"] for r in rows if r["method"] == method]
        by_method[method] = {"n": len(sub), "mape": sum(sub) / len(sub)}
    return {
        "n": n,
        "mape": mape,
        "median_ape": median,
        "coverage": coverage,
        "by_method": by_method,
    }


# ------------------------------ estimator wiring -----------------------------
def build_estimator(use_reference: bool) -> PortionEstimator:
    """Construct the estimator with whatever aux models are available (like the app)."""
    from app.ml.portion.container_model import load_container_model
    from app.ml.portion.segmenter import load_food_segmenter

    model_dir = Path(settings.model_dir)
    container_model = None
    cw, cl = model_dir / settings.container_weights, model_dir / settings.container_labels
    if cw.exists() and cl.exists():
        container_model = load_container_model(str(cw), str(cl), settings.device)
        print(f"  container model: {cw.name}")

    segmenter = None
    sw = model_dir / settings.segmenter_weights
    if sw.exists():
        segmenter = load_food_segmenter(str(sw), settings.device)
        print(f"  segmenter: {sw.name}")

    reference_detector = None
    if use_reference:
        from app.ml.portion.reference_detector import load_coco_reference_detector

        reference_detector = load_coco_reference_detector(
            settings.device, settings.reference_score_threshold
        )
        print("  reference detector: COCO (fork/spoon/knife)")

    return PortionEstimator(
        container_model=container_model,
        reference_detector=reference_detector,
        segmenter=segmenter,
    )


# ---------------------------------- run --------------------------------------
def evaluate(manifest: Path, images_dir: Path, out: Path, use_reference: bool) -> dict:
    estimator = build_estimator(use_reference)
    rows: list[dict] = []
    detailed: list[dict] = []

    with open(manifest, newline="") as f:
        for entry in csv.DictReader(f):
            true_g = float(entry["true_grams"])
            if true_g <= 0:
                print(f"  skip {entry['image']}: true_grams must be > 0")
                continue
            path = images_dir / entry["image"] if images_dir else Path(entry["image"])
            if not path.exists():
                print(f"  skip {entry['image']}: file not found ({path})")
                continue

            est = estimator.estimate(Image.open(path), entry["food_class"])
            ape = absolute_percentage_error(est.point_g, true_g)
            covered = est.min_g <= true_g <= est.max_g
            rows.append({"ape": ape, "covered": covered, "method": est.method})
            detailed.append(
                {
                    "image": entry["image"],
                    "food_class": entry["food_class"],
                    "true_grams": true_g,
                    "point_g": est.point_g,
                    "min_g": est.min_g,
                    "max_g": est.max_g,
                    "method": est.method,
                    "reference": est.reference or "",
                    "ape_pct": round(ape, 1),
                    "covered": covered,
                }
            )

    if detailed:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detailed[0]))
            w.writeheader()
            w.writerows(detailed)

    return summarize(rows)


def print_summary(s: dict, out: Path) -> None:
    print("\n" + "=" * 44)
    print(f"Portion MAPE report (N={s['n']})")
    if s["n"] == 0:
        print("  no valid rows — check the manifest/paths.")
        return
    print(f"  Overall MAPE:    {s['mape']:.1f}%")
    print(f"  Median APE:      {s['median_ape']:.1f}%")
    print(f"  Range coverage:  {s['coverage']:.1f}%   (true weight inside [min,max])")
    print("  By scale method:")
    for method, m in s["by_method"].items():
        print(f"    {method:17s} n={m['n']:<4d} MAPE={m['mape']:.1f}%")
    print(f"  Per-item CSV:    {out}")
    print("=" * 44)


def main() -> None:
    ap = argparse.ArgumentParser(description="Portion-estimation MAPE (05 §7, §9)")
    ap.add_argument("--manifest", required=True, help="CSV: image,food_class,true_grams")
    ap.add_argument("--images-dir", default="", help="prefix for relative image paths")
    ap.add_argument("--out", default="artifacts/mape_report.csv")
    ap.add_argument("--use-reference", action="store_true", help="enable COCO detector")
    args = ap.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else Path()
    summary = evaluate(Path(args.manifest), images_dir, Path(args.out), args.use_reference)
    print_summary(summary, Path(args.out))


if __name__ == "__main__":
    main()
