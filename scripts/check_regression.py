#!/usr/bin/env python3
"""Phase 7: eval-as-CI's actual gate. Compares a freshly-produced
scripts/run_eval.py result against the frozen data/eval/results/
ci_baseline.json and exits non-zero if a tracked metric regressed past its
tolerance -- this is the piece that makes "eval-as-CI" mean something
more than "the eval script ran and printed some numbers."

Deliberately simple and dependency-free (stdlib only) so it's trivial to
run both in CI and locally with no extra setup, and so its own logic is
easy to unit-test (see tests/test_check_regression.py) with synthetic
before/after JSON -- no Chroma, no retrieval pipeline, no network needed to
verify the *gate itself* is correct.

Tolerances are absolute-percentage-point drops, not relative, and
deliberately non-zero: local_tfidf/BM25/RRF are all fully deterministic
given the same corpus and config, so in principle a regression check could
demand an exact match -- but exact-match gates are exactly the kind of
brittle CI that gets `--force`-merged past on a bad day. A couple of points
of slack catches a real regression without being a nuisance over
noise-level float differences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import REPO_ROOT  # noqa: E402

# (dotted path into the results JSON, human label, max allowed drop in
# absolute percentage points, higher-is-better)
TRACKED_METRICS: list[tuple[tuple[str, ...], str, float]] = [
    (("retrieval", "recall_at_k"), "Recall@k", 0.02),
    (("retrieval", "mrr"), "MRR", 0.02),
    (("retrieval", "citation_coverage"), "Citation coverage (all)", 0.02),
    (("retrieval", "citation_coverage_version_sensitive"), "Citation coverage (version_sensitive)", 0.05),
    (("refusal", "recall"), "Refusal recall", 0.02),
]


def _get(d: dict, path: tuple[str, ...]):
    node = d
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def check(baseline: dict, current: dict) -> list[str]:
    """Returns a list of human-readable failure messages -- empty means the
    gate passes. Pure function, no I/O, so it's directly unit-testable."""
    failures = []
    for path, label, tolerance in TRACKED_METRICS:
        base_val = _get(baseline, path)
        cur_val = _get(current, path)
        if base_val is None:
            continue  # nothing to compare against -- not a regression, just untracked
        if cur_val is None:
            failures.append(f"{label}: baseline had {base_val:.2%}, current run produced no value at all")
            continue
        drop = base_val - cur_val
        if drop > tolerance:
            failures.append(
                f"{label} regressed: {base_val:.2%} -> {cur_val:.2%} "
                f"(dropped {drop:.2%}, tolerance is {tolerance:.2%})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/eval/results/ci_baseline.json")
    parser.add_argument("--current", default="data/eval/results/ci.json")
    args = parser.parse_args()

    baseline_path = REPO_ROOT / args.baseline
    current_path = REPO_ROOT / args.current

    if not baseline_path.exists():
        print(f"No baseline file at {baseline_path.relative_to(REPO_ROOT)} -- nothing to compare against.")
        print("(Run scripts/run_eval.py with config/ci.yaml and copy the output there first.)")
        return 1
    if not current_path.exists():
        print(f"No current result file at {current_path.relative_to(REPO_ROOT)} -- run scripts/run_eval.py first.")
        return 1

    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())

    failures = check(baseline, current)

    print(f"=== Regression check: {current_path.name} vs. {baseline_path.name} ===")
    for path, label, tolerance in TRACKED_METRICS:
        base_val, cur_val = _get(baseline, path), _get(current, path)
        if base_val is None:
            continue
        cur_str = f"{cur_val:.2%}" if cur_val is not None else "n/a"
        delta_str = f"{(cur_val - base_val):+.2%}" if cur_val is not None else "n/a"
        print(f"  {label:<42} baseline {base_val:.2%}  current {cur_str}  ({delta_str})")

    if failures:
        print(f"\nREGRESSION DETECTED ({len(failures)} metric(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nNo regression beyond tolerance. Gate passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
