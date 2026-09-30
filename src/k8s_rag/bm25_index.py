"""BM25 keyword search -- the other half of Phase 3's hybrid retrieval.
Deliberately kept fully separate from VectorStore/embeddings: BM25 scores
raw term overlap, vector search scores embedding similarity, and they're
fused at the *ranking* level in retrieval.py (Reciprocal Rank Fusion), never
blended into one score space. That separation is the point of "hybrid" --
each retriever's failure mode is covered by the other's strength (Phase 1's
ABLATIONS.md has a concrete example of keyword matching winning outright on
this jargon-dense corpus, which is the whole reason BM25 belongs here at
all, not just embeddings alone).

rank_bm25's BM25Okapi wants the whole corpus in memory as tokenized
documents -- fine at this project's scale (hundreds to a couple thousand
chunks) and exactly what the library expects, not a shortcut.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from k8s_rag.config import REPO_ROOT, RunConfig

_TOKEN_RE = re.compile(r"[a-zA-Z0-9/._-]+")


def bm25_persist_path(cfg: RunConfig) -> Path:
    persist_dir = REPO_ROOT / cfg.get("vector_store", "persist_dir", default="data/chroma")
    return persist_dir / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """Keeps slashes and dots inside a token (`policy/v1beta1`, `1.25`)
    rather than splitting on them -- those punctuation marks are load-bearing
    in Kubernetes API identifiers, and splitting them apart would turn an
    exact-match signal into a weaker multi-token partial match."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self) -> None:
        self._model: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._versions: list[str] = []

    def fit_and_persist(self, chunks: list[dict], path: Path) -> None:
        """chunks: [{id, text, version, ...}]. `text` is what gets indexed --
        same field the vector store embeds, so both retrievers see the same
        unit of content."""
        chunk_ids = [c["id"] for c in chunks]
        versions = [c["version"] for c in chunks]
        tokenized = [tokenize(c["text"]) for c in chunks]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": chunk_ids, "versions": versions, "tokenized": tokenized}, f)
        self._chunk_ids, self._versions = chunk_ids, versions
        self._model = BM25Okapi(tokenized)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(
                f"No fitted BM25 index at {path}. Run scripts/ingest.py first -- "
                "it fits BM25 on the corpus and persists it here."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._versions = data["versions"]
        self._model = BM25Okapi(data["tokenized"])

    def search(self, query: str, top_k: int, version_filter: str | None = None) -> list[BM25Hit]:
        if self._model is None:
            raise RuntimeError("BM25Index used before load()/fit_and_persist()")
        scores = self._model.get_scores(tokenize(query))
        pairs = zip(self._chunk_ids, self._versions, scores)
        if version_filter:
            pairs = (p for p in pairs if p[1] == version_filter)
        ranked = sorted(pairs, key=lambda p: p[2], reverse=True)
        return [BM25Hit(chunk_id=cid, score=float(s)) for cid, _v, s in ranked[:top_k]]
