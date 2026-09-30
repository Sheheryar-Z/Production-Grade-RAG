#!/usr/bin/env python3
"""Phase 8: run the FastAPI service. Respects K8S_RAG_CONFIG like every
other script in this repo (RunConfig.load()) so the exact same config that
produced a recorded eval result is what the live service runs -- no
separate "serving config" that could silently drift from what was measured.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.api import create_app  # noqa: E402
from k8s_rag.config import RunConfig  # noqa: E402


def main() -> None:
    import uvicorn

    cfg = RunConfig.load()
    rate_limit = int(os.environ.get("K8S_RAG_RATE_LIMIT_PER_MINUTE", "60"))
    app = create_app(cfg, rate_limit_per_minute=rate_limit)

    host = os.environ.get("K8S_RAG_HOST", "0.0.0.0")
    port = int(os.environ.get("K8S_RAG_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
