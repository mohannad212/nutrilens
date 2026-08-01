"""
tests/test_m4_mape.py
========================

Verifies the pure metric math in `scripts/eval_portion_mape.py` (the MAPE harness,
`05 §7, §9`). The end-to-end run (loading models, reading images) is exercised
separately; here we pin the arithmetic — MAPE, median, coverage, per-method — so
the reported headline number can't silently be computed wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_portion_mape as mape  # noqa: E402


def test_absolute_percentage_error():
    assert mape.absolute_percentage_error(240, 200) == 20.0
    assert mape.absolute_percentage_error(150, 200) == 25.0
    assert mape.absolute_percentage_error(200, 200) == 0.0


def test_summarize_mape_median_coverage():
    rows = [
        {"ape": 20.0, "covered": True, "method": "container_prior"},
        {"ape": 10.0, "covered": True, "method": "reference_object"},
        {"ape": 50.0, "covered": False, "method": "shape_only"},
    ]
    s = mape.summarize(rows)
    assert s["n"] == 3
    assert round(s["mape"], 1) == 26.7  # (20+10+50)/3
    assert s["median_ape"] == 20.0  # middle of [10,20,50]
    assert round(s["coverage"], 1) == 66.7  # 2 of 3 covered
    assert s["by_method"]["reference_object"] == {"n": 1, "mape": 10.0}


def test_summarize_even_count_median_is_averaged():
    rows = [
        {"ape": 10.0, "covered": True, "method": "shape_only"},
        {"ape": 30.0, "covered": True, "method": "shape_only"},
    ]
    assert mape.summarize(rows)["median_ape"] == 20.0  # (10+30)/2


def test_summarize_empty():
    s = mape.summarize([])
    assert s["n"] == 0 and s["mape"] == 0.0 and s["by_method"] == {}
