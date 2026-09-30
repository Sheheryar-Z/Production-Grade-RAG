"""Thin wrapper around ChromaDB. Kept deliberately small: Chroma is doing the
ANN search, this module just owns the schema (what metadata rides along with
every chunk) so retrieval, citation formatting, and eval all agree on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from k8s_rag.config import REPO_ROOT, RunConfig
from k8s_rag.embeddings import get_embedding_provider


@dataclass
class RetrievedChunk:
    text: str             # the retrieval-matched text (child chunk, for parent_document strategy)
    context_text: str     # what generation should actually read (== text unless parent_document)
    version: str
    source_page: str
    source_url: str
    chunk_id: str
    score: float           # similarity/relevance, higher = more relevant. For a hybrid-mode
                            # result this is the fused RRF score, not raw cosine similarity --
                            # see retrieval.py. Don't compare scores across retrieval modes.
    parent_id: str | None = None


def dedupe_by_parent(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Several child chunks of a parent_document strategy can legitimately
    all rank highly for one query; only the first (highest-ranked) one
    should reach generation, or the same parent context gets repeated.
    Chunks with no parent_id (fixed/recursive strategies) are never deduped
    against each other. Shared by vector-only and hybrid retrieval so both
    apply this rule identically."""
    out: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    for c in chunks:
        if c.parent_id and c.parent_id in seen_parents:
            continue
        if c.parent_id:
            seen_parents.add(c.parent_id)
        out.append(c)
        if len(out) >= top_k:
            break
    return out


class VectorStore:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        persist_dir = REPO_ROOT / cfg.get("vector_store", "persist_dir", default="data/chroma")
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = cfg.get("vector_store", "collection_name", default="k8s_docs")
        self._embedder = get_embedding_provider(cfg)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Each chunk dict: {id, text, context_text, version, source_page,
        source_url, parent_id?}. `context_text` and `parent_id` are optional
        and default to `text` / "" -- only `parent_document` chunking sets
        them to something different from the retrieval text itself."""
        if not chunks:
            return
        embeddings = self._embedder.embed([c["text"] for c in chunks])
        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "version": c["version"],
                    "source_page": c["source_page"],
                    "source_url": c["source_url"],
                    "context_text": c.get("context_text") or c["text"],
                    "parent_id": c.get("parent_id") or "",
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        return self._collection.count()

    def _to_chunk(self, cid: str, doc: str, meta: dict, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            text=doc, context_text=meta.get("context_text") or doc,
            version=meta["version"], source_page=meta["source_page"],
            source_url=meta["source_url"], chunk_id=cid, score=score,
            parent_id=meta.get("parent_id") or None,
        )

    def raw_search(self, query: str, top_k: int, version_filter: str | None = None) -> list[RetrievedChunk]:
        """Ranked vector hits, no parent dedup applied -- used directly by
        vector-only mode's `search()` (which dedups after overfetching) and
        by the hybrid retriever (which dedups only after fusing with BM25,
        so a chunk isn't penalized just because a vector-only pass would
        have deduped it before BM25 got a say)."""
        query_embedding = self._embedder.embed_one(query)
        where = {"version": version_filter} if version_filter else None
        n = min(top_k, max(self._collection.count(), 1))
        result = self._collection.query(query_embeddings=[query_embedding], n_results=n, where=where)
        ids, docs, metas, dists = result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        return [self._to_chunk(cid, doc, meta, 1.0 - dist) for cid, doc, meta, dist in zip(ids, docs, metas, dists)]

    def search(self, query: str, top_k: int, version_filter: str | None = None) -> list[RetrievedChunk]:
        # overfetch: when parent_document chunking is in play, several
        # child chunks can share one parent -- fetch extra raw hits so
        # dedup below still leaves top_k *distinct* results, not fewer.
        overfetch_k = top_k * 4
        raw = self.raw_search(query, overfetch_k, version_filter)
        return dedupe_by_parent(raw, top_k)

    def get_by_ids(self, ids: list[str]) -> dict[str, RetrievedChunk]:
        """Materialize full RetrievedChunk records for a set of chunk ids
        whose scores came from somewhere other than a vector query (BM25,
        in Phase 3's hybrid retriever). The returned score is a placeholder
        (0.0) -- callers overwrite it with the real fused score."""
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["documents", "metadatas"])
        out = {}
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            out[cid] = self._to_chunk(cid, doc, meta, 0.0)
        return out
