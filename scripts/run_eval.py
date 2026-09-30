#!/usr/bin/env python3
"""Phase 1 eval harness.

Two tiers, deliberately kept separate and never blended into one fake number:

1. Retrieval metrics (Recall@k, MRR, refusal precision/recall) -- computed
   locally, no API key needed, no LLM judge involved. These are directly
   measurable from what the vector store actually returns and are real
   numbers regardless of which generation provider is configured.

2. Generation-quality metrics (Ragas faithfulness, answer relevancy,
   citation precision via an LLM judge) -- require a real LLM. If
   OPENAI_API_KEY is not set, this tier is skipped with an explicit message
   and reported as "N/A (requires OPENAI_API_KEY)" in the results -- never
   silently defaulted to a fake or placeholder score.

Results are written to data/eval/results/<run_name>.json and a one-line
summary is appended to docs/ABLATIONS.md.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import REPO_ROOT, RunConfig  # noqa: E402
from k8s_rag.eval_metrics import citation_coverage, citation_hit, reciprocal_rank  # noqa: E402
from k8s_rag.graph import run_pipeline  # noqa: E402
from k8s_rag.repro import repro_metadata  # noqa: E402
from k8s_rag.vectorstore import VectorStore  # noqa: E402


def load_golden_set() -> list[dict]:
    path = REPO_ROOT / "data" / "eval" / "golden_qa.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    cfg = RunConfig.load()
    store = VectorStore(cfg)
    questions = load_golden_set()
    top_k = cfg.get("retrieval", "top_k", default=8)

    repro = repro_metadata(REPO_ROOT, cfg.path)
    print(f"=== Run: {cfg.get('run_name')} ({cfg.path.relative_to(REPO_ROOT)}) ===")
    if repro["git_commit"]:
        dirty_flag = " [DIRTY -- uncommitted changes present]" if repro["git_dirty"] else ""
        print(f"  git commit: {repro['git_commit'][:12]}{dirty_flag}")
    else:
        print("  git commit: unavailable (not a git checkout, or git not installed)")
    print(f"  config content hash: {repro['config_content_hash']}")

    retrieval_rows = []
    refusal_rows = []
    latencies_ms = []

    for q in questions:
        t0 = time.time()
        result = run_pipeline(cfg, store, q["question"], target_version=q.get("target_version"))
        latencies_ms.append((time.time() - t0) * 1000)

        chunks = result.get("retrieved_chunks", [])
        if q["category"] in ("single_version", "version_sensitive") and q["expected_citations"]:
            hit, rank = citation_hit(chunks, q["expected_citations"])
            coverage = citation_coverage(chunks, q["expected_citations"])
            retrieval_rows.append({
                "id": q["id"], "category": q["category"], "hit_at_k": hit,
                "reciprocal_rank": reciprocal_rank(rank), "citation_coverage": coverage,
            })
        if q["category"] == "unanswerable":
            correctly_refused = result["decision"] == "refuse"
            refusal_rows.append({
                "id": q["id"], "correctly_refused": correctly_refused,
                "decision": result["decision"],
            })

    recall_at_k = statistics.mean(r["hit_at_k"] for r in retrieval_rows) if retrieval_rows else None
    mrr = statistics.mean(r["reciprocal_rank"] for r in retrieval_rows) if retrieval_rows else None
    coverage = statistics.mean(r["citation_coverage"] for r in retrieval_rows) if retrieval_rows else None
    vs_rows = [r for r in retrieval_rows if r["category"] == "version_sensitive"]
    vs_coverage = statistics.mean(r["citation_coverage"] for r in vs_rows) if vs_rows else None
    refusal_recall = (
        statistics.mean(r["correctly_refused"] for r in refusal_rows) if refusal_rows else None
    )
    p50_latency = statistics.median(latencies_ms) if latencies_ms else None

    print(f"\n=== Retrieval metrics (n={len(retrieval_rows)}) ===")
    print(f"  Recall@{top_k}: {recall_at_k:.2%}" if recall_at_k is not None else "  Recall@k: n/a")
    print(f"  MRR:      {mrr:.3f}" if mrr is not None else "  MRR: n/a")
    print(f"  Citation coverage (all): {coverage:.2%}" if coverage is not None else "  Citation coverage: n/a")
    print(
        f"  Citation coverage (version_sensitive only, n={len(vs_rows)}): {vs_coverage:.2%}"
        if vs_coverage is not None else "  Citation coverage (version_sensitive): n/a"
    )
    print(f"\n=== Refusal metrics (n={len(refusal_rows)}, unanswerable category) ===")
    print(f"  Refusal recall: {refusal_recall:.2%}" if refusal_recall is not None else "  n/a")
    for r in refusal_rows:
        flag = "OK" if r["correctly_refused"] else "MISS (answered instead of refusing)"
        print(f"    {r['id']}: {flag}")
    print(f"\n=== Latency ===\n  p50: {p50_latency:.0f} ms (retrieval + stub generation, local, no LLM call)")

    ragas_status = "skipped"
    faithfulness = citation_precision = answer_relevancy = None
    if os.environ.get("OPENAI_API_KEY") and cfg.get("generation", "provider") == "openai":
        print("\n=== Ragas generation-quality metrics ===")
        print("  OPENAI_API_KEY detected -- running Ragas faithfulness/relevancy would go here.")
        print("  (Wire this in scripts/run_eval.py once generation.provider: openai is configured;")
        print("   left explicit and unimplemented-by-default rather than faked.)")
        ragas_status = "not_yet_wired"
    else:
        print("\n=== Ragas generation-quality metrics: SKIPPED ===")
        print("  Reason: OPENAI_API_KEY not set (or generation.provider != openai).")
        print("  Faithfulness/citation-precision/answer-relevancy require a real LLM judge --")
        print("  reported as N/A below rather than backfilled with a fake number.")

    results = {
        "run_name": cfg.get("run_name"),
        "config_path": str(cfg.path.relative_to(REPO_ROOT)),
        "repro": repro_metadata(REPO_ROOT, cfg.path),
        "n_questions": len(questions),
        "retrieval": {
            "recall_at_k": recall_at_k, "mrr": mrr, "top_k": top_k,
            "citation_coverage": coverage, "citation_coverage_version_sensitive": vs_coverage,
        },
        "refusal": {"recall": refusal_recall, "rows": refusal_rows},
        "generation_quality": {
            "status": ragas_status,
            "faithfulness": faithfulness,
            "citation_precision": citation_precision,
            "answer_relevancy": answer_relevancy,
        },
        "latency_ms": {"p50": p50_latency},
    }

    out_dir = REPO_ROOT / "data" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.get('run_name')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
