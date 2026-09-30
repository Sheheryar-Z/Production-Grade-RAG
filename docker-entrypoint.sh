#!/bin/sh
# Phase 8: first-run ingestion. A fresh image (or a fresh volume mounted at
# data/chroma) has no vector store yet -- `docker run` should work
# end-to-end from a clean checkout, not require a separate manual "now go
# run scripts/ingest.py" step the README has to explain. Idempotent: once
# data/chroma is populated (e.g. a named volume reused across restarts),
# this is a fast no-op check, not a re-fetch/re-ingest every start.
set -e

CHUNK_COUNT=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from k8s_rag.config import RunConfig
from k8s_rag.vectorstore import VectorStore
try:
    cfg = RunConfig.load()
    print(VectorStore(cfg).count())
except Exception:
    print(0)
" 2>/dev/null || echo 0)

if [ "$CHUNK_COUNT" -eq "0" ]; then
    echo "[entrypoint] No ingested data found (config: ${K8S_RAG_CONFIG:-config/default.yaml}) -- fetching corpus and ingesting."
    echo "[entrypoint] This needs network access to github.com; if it fails, mount a pre-built data/chroma volume instead."
    python3 scripts/fetch_corpus.py
    python3 scripts/ingest.py
else
    echo "[entrypoint] Found $CHUNK_COUNT ingested chunks -- skipping fetch/ingest."
fi

exec "$@"
