#!/usr/bin/env python3
"""Phase 6: config & experiment infrastructure. One place to see every
recorded eval run, the config and commit it came from, and whether it's
still actually reproducible -- rather than trusting docs/ABLATIONS.md's
prose to have transcribed every number correctly by hand.

Two checks this does that a spreadsheet of results wouldn't catch on its
own:
- **Config drift**: hashes the config file this run says it used and
  compares it to the file's current on-disk content. A mismatch means the
  config was edited after this run without producing a new run -- the
  recorded numbers no longer describe what that file does today.
- **Dirty-tree runs**: flags any run recorded while the git working tree
  had uncommitted changes -- `git show <commit>:config/...` won't
  reconstruct that exact run for those, only the config file's own bytes
  (still hashed and checked, just not commit-pinned).

Neither check requires touching Chroma, an embedding model, or the
network -- it only reads the JSON files already written to
data/eval/results/ and the config files already in the repo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import REPO_ROOT  # noqa: E402
from k8s_rag.repro import config_content_hash  # noqa: E402


def load_results() -> list[dict]:
    results_dir = REPO_ROOT / "data" / "eval" / "results"
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("-calibration.json"):
            continue  # calibration sweeps are a different shape -- listed separately below
        try:
            rows.append({"_path": path, **json.loads(path.read_text())})
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def fmt_pct(x) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"


def main() -> None:
    rows = load_results()
    if not rows:
        print("No eval results found under data/eval/results/. Run scripts/run_eval.py first.")
        return

    header = f"{'run_name':<32} {'recall@k':>9} {'mrr':>7} {'refusal':>8} {'commit':>10} {'status'}"
    print(header)
    print("-" * len(header))

    for r in rows:
        run_name = r.get("run_name", r["_path"].stem)
        recall = fmt_pct(r.get("retrieval", {}).get("recall_at_k"))
        mrr_val = r.get("retrieval", {}).get("mrr")
        mrr_str = f"{mrr_val:.3f}" if isinstance(mrr_val, (int, float)) else "n/a"
        refusal = fmt_pct(r.get("refusal", {}).get("recall"))

        repro = r.get("repro")
        config_path = r.get("config_path")
        status_flags = []

        if repro is None:
            status_flags.append("no repro metadata (pre-Phase-6 run)")
            commit_str = "n/a"
        else:
            commit = repro.get("git_commit")
            commit_str = commit[:10] if commit else "n/a"
            if repro.get("git_dirty"):
                status_flags.append("recorded on a DIRTY tree")

            if config_path:
                config_file = REPO_ROOT / config_path
                if not config_file.exists():
                    status_flags.append(f"config file missing: {config_path}")
                else:
                    current_hash = config_content_hash(config_file)
                    recorded_hash = repro.get("config_content_hash")
                    if recorded_hash and current_hash != recorded_hash:
                        status_flags.append(
                            f"CONFIG DRIFT: {config_path} has changed since this run "
                            f"({recorded_hash} -> {current_hash})"
                        )

        status = "; ".join(status_flags) if status_flags else "OK, reproducible"
        print(f"{run_name:<32} {recall:>9} {mrr_str:>7} {refusal:>8} {commit_str:>10} {status}")


if __name__ == "__main__":
    main()
