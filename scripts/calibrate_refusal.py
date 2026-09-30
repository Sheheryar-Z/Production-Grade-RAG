#!/usr/bin/env python3
"""Phase 4, the half of it that doesn't need a new model: calibrate a
refusal confidence threshold directly from retrieval scores already being
produced, rather than picking `min_relevance_score` by guesswork.

Why there's no ML cross-encoder reranker in this repo yet: investigated
`sentence-transformers` CrossEncoder models and the `flashrank` library as
network-free-ish options -- both download their model weights from
huggingface.co (confirmed by reading flashrank's own source:
`model_url = 'https://huggingface.co/prithivida/flashrank/resolve/main/{}.zip'`
in its Config.py), the same host this sandbox's egress policy blocks. Unlike
Phase 1's embedding model, no GitHub-hosted alternative was found for a real
reranker. That's a documented gap, not a silent skip -- see docs/ABLATIONS.md.

What IS possible without a new model: the retriever already produces a
relevance score per result (RRF fused rank score, in hybrid mode). Whether
that score is a *good* refusal signal -- i.e. whether unanswerable questions
reliably score lower than answerable ones -- is an empirical question, so
this script answers it empirically: run retrieval (not generation) over the
golden set, collect each question's top-1 score, and sweep threshold
candidates to see how well score alone separates "should answer" from
"should refuse".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import REPO_ROOT, RunConfig  # noqa: E402
from k8s_rag.graph import build_retriever  # noqa: E402
from k8s_rag.query_understanding import fanout_search, needs_version_fanout  # noqa: E402
from k8s_rag.repro import repro_metadata  # noqa: E402
from k8s_rag.vectorstore import VectorStore  # noqa: E402


def _search(retriever, question: str, target_version: str | None, cfg: RunConfig, top_k: int):
    """Mirrors graph.py's retrieve_node exactly -- calibrating against plain
    retriever.search() while query_understanding.enabled routes real
    traffic through fanout_search() would calibrate a threshold against a
    score distribution the pipeline never actually produces. (This mismatch
    is exactly what the first version of this script got wrong -- see
    docs/ABLATIONS.md Phase 5.)"""
    if needs_version_fanout(question, target_version, cfg):
        return fanout_search(retriever, question, cfg, top_k)
    return retriever.search(question, top_k=top_k, version_filter=target_version)


def load_golden_set() -> list[dict]:
    path = REPO_ROOT / "data" / "eval" / "golden_qa.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    cfg = RunConfig.load()
    store = VectorStore(cfg)
    retriever = build_retriever(cfg, store)
    top_k = cfg.get("retrieval", "top_k", default=8)
    questions = load_golden_set()

    rows = []
    for q in questions:
        chunks = _search(retriever, q["question"], q.get("target_version"), cfg, top_k)
        top_score = chunks[0].score if chunks else 0.0
        rows.append({
            "id": q["id"], "category": q["category"], "top_score": top_score,
            "should_refuse": q["category"] == "unanswerable",
        })

    print(f"{'id':10} {'category':18} {'top_score':>10}  should_refuse")
    for r in sorted(rows, key=lambda r: r["top_score"]):
        print(f"{r['id']:10} {r['category']:18} {r['top_score']:>10.5f}  {r['should_refuse']}")

    scores = sorted({round(r["top_score"], 6) for r in rows})
    # candidate thresholds: midpoints between consecutive distinct scores,
    # plus the extremes, so every possible split point gets tried
    candidates = [scores[0] - 1e-6] + [
        (a + b) / 2 for a, b in zip(scores, scores[1:])
    ] + [scores[-1] + 1e-6]

    should_refuse = [r for r in rows if r["should_refuse"]]
    should_answer = [r for r in rows if not r["should_refuse"]]

    best = None
    results = []
    for t in candidates:
        refusal_recall = sum(r["top_score"] < t for r in should_refuse) / len(should_refuse)
        answerable_pass_rate = sum(r["top_score"] >= t for r in should_answer) / len(should_answer)
        # Youden's J: how well this single threshold separates the two
        # groups, independent of how many questions are in each group
        j = refusal_recall + answerable_pass_rate - 1
        results.append((t, refusal_recall, answerable_pass_rate, j))
        if best is None or j > best[3]:
            best = (t, refusal_recall, answerable_pass_rate, j)

    print(f"\n{'threshold':>10} {'refusal_recall':>15} {'answerable_pass_rate':>22} {'youden_j':>10}")
    for t, rr, apr, j in results:
        marker = "  <-- best" if best is not None and t == best[0] else ""
        print(f"{t:>10.5f} {rr:>15.2%} {apr:>22.2%} {j:>10.3f}{marker}")

    print(f"\nBest threshold: {best[0]:.5f} -> refusal_recall={best[1]:.1%}, "
          f"answerable_pass_rate={best[2]:.1%} (n_refuse={len(should_refuse)}, n_answer={len(should_answer)})")

    out_path = REPO_ROOT / "data" / "eval" / "results" / f"{cfg.get('run_name')}-calibration.json"
    out_path.write_text(json.dumps({
        "config_path": str(cfg.path.relative_to(REPO_ROOT)),
        "repro": repro_metadata(REPO_ROOT, cfg.path),
        "rows": rows,
        "best_threshold": best[0], "best_refusal_recall": best[1], "best_answerable_pass_rate": best[2],
    }, indent=2))
    print(f"\nWritten to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
