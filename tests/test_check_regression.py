"""Phase 7: the eval-as-CI regression gate. Tests the pure check() function
against synthetic before/after dicts -- deliberately no dependency on
Chroma, a real pipeline, or the checked-in baseline file, so these verify
the GATE's logic (what counts as a regression, tolerance handling, missing
values) independent of whatever the actual pipeline numbers happen to be
today."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_regression import check  # noqa: E402


def _result(recall=1.0, mrr=0.9, coverage=0.9, vs_coverage=0.8, refusal=0.83):
    return {
        "retrieval": {
            "recall_at_k": recall, "mrr": mrr,
            "citation_coverage": coverage, "citation_coverage_version_sensitive": vs_coverage,
        },
        "refusal": {"recall": refusal},
    }


def test_identical_results_pass():
    baseline = _result()
    current = _result()
    assert check(baseline, current) == []


def test_improvement_passes():
    baseline = _result(recall=0.90, refusal=0.70)
    current = _result(recall=1.00, refusal=0.90)
    assert check(baseline, current) == []


def test_small_drop_within_tolerance_passes():
    baseline = _result(recall=1.00)
    current = _result(recall=0.99)  # 1 point drop, tolerance is 2 points
    assert check(baseline, current) == []


def test_large_drop_fails():
    baseline = _result(recall=1.00)
    current = _result(recall=0.80)
    failures = check(baseline, current)
    assert len(failures) == 1
    assert "Recall@k" in failures[0]


def test_refusal_regression_is_caught_independently_of_recall():
    baseline = _result(refusal=0.833)
    current = _result(refusal=0.50)  # exactly the Phase 5 fan-out regression, replayed
    failures = check(baseline, current)
    assert any("Refusal recall" in f for f in failures)


def test_multiple_simultaneous_regressions_all_reported():
    baseline = _result(recall=1.0, refusal=0.833, coverage=0.90)
    current = _result(recall=0.80, refusal=0.50, coverage=0.60)
    failures = check(baseline, current)
    assert len(failures) == 3


def test_missing_baseline_metric_is_not_a_regression():
    baseline = {"retrieval": {}, "refusal": {}}
    current = _result()
    assert check(baseline, current) == []


def test_missing_current_value_is_reported():
    baseline = _result()
    current = {"retrieval": {}, "refusal": {}}
    failures = check(baseline, current)
    # every tracked metric that had a baseline value but no current value
    # should be flagged
    assert len(failures) == len(baseline["retrieval"]) + len(baseline["refusal"])


def test_version_sensitive_coverage_has_wider_tolerance():
    # 4 points is within this metric's 5-point tolerance but would fail the
    # 2-point tolerance every other retrieval metric uses
    baseline = _result(vs_coverage=0.80)
    current = _result(vs_coverage=0.76)
    assert check(baseline, current) == []
