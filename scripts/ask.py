#!/usr/bin/env python3
"""Ask the pipeline a single question from the command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.config import RunConfig  # noqa: E402
from k8s_rag.graph import run_pipeline  # noqa: E402
from k8s_rag.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--version", default=None, help="filter retrieval to one k8s version")
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig.load()
    store = VectorStore(cfg)
    result = run_pipeline(cfg, store, args.question, target_version=args.version)

    if args.show_chunks:
        print("--- retrieved chunks ---")
        for c in result.get("retrieved_chunks", []):
            print(f"  [{c.score:.3f}] v{c.version} {c.source_page} :: {c.text[:100].strip()}...")
        print()

    print(f"--- decision: {result['decision']} ---")
    print(result["answer"])


if __name__ == "__main__":
    main()
