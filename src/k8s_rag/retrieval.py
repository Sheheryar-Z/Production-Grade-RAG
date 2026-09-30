"""Phase 3: hybrid retrieval. Fuses vector search and BM25 keyword search
with Reciprocal Rank Fusion (RRF) rather than a weighted score blend --
RRF works on *rank position*, not raw score magnitude, which sidesteps the
problem that cosine similarity and BM25 scores live on completely different,
incomparable scales (there's no principled way to pick a blend weight
between a 0-1 cosine value and an unbounded BM25 score without RRF's
rank-based normalization doing that work instead).

    RRF(d) = sum over each ranker r where d appears: 1 / (k_rrf + rank_r(d))

A chunk that ranks well in *either* list scores well; a chunk that ranks
well in *both* scores best of all. k_rrf=60 is the standard constant from
the original RRF paper (Cormack et al.) -- it flattens the curve so rank 1
vs. rank 2 isn't a huge score swing, which matters here because BM25 and
vector rankings can legitimately disagree by a couple of positions on a
genuinely relevant chunk.
"""
from __future__ import annotations

from k8s_rag.bm25_index import BM25Index
from k8s_rag.config import RunConfig
from k8s_rag.vectorstore import RetrievedChunk, VectorStore, dedupe_by_parent

RRF_K = 60


def rrf_fuse(ranked_id_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Pure RRF math, kept separate from any store/index so it's directly
    unit-testable: `ranked_id_lists` is one ranked list of ids per
    retriever, best match first. Returns {id: fused_score}, not yet sorted."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, cid in enumerate(ranked_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, store: VectorStore, bm25: BM25Index, cfg: RunConfig):
        self._store = store
        self._bm25 = bm25
        self._overfetch = cfg.get("retrieval", "top_k", default=8) * 4

    def search(self, query: str, top_k: int, version_filter: str | None = None) -> list[RetrievedChunk]:
        vector_hits = self._store.raw_search(query, self._overfetch, version_filter)
        bm25_hits = self._bm25.search(query, self._overfetch, version_filter)

        rrf_scores = rrf_fuse([
            [h.chunk_id for h in vector_hits],
            [h.chunk_id for h in bm25_hits],
        ])

        # materialize every fused candidate as a full RetrievedChunk. Chunks
        # that came from the vector pass already have one; BM25-only chunks
        # need a lookup (they only carried an id + score so far).
        by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in vector_hits}
        missing_ids = [cid for cid in rrf_scores if cid not in by_id]
        by_id.update(self._store.get_by_ids(missing_ids))

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered: list[RetrievedChunk] = []
        for cid, rrf_score in fused:
            chunk = by_id.get(cid)
            if chunk is None:
                continue  # shouldn't happen, but never crash retrieval over a missing lookup
            # overwrite with the fused score -- see RetrievedChunk.score's docstring:
            # downstream code (refusal thresholding, eval) should read this, not the
            # original single-retriever score, once fusion has happened.
            chunk.score = rrf_score
            ordered.append(chunk)

        return dedupe_by_parent(ordered, top_k)
