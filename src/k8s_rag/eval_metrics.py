"""Retrieval-quality metrics that don't need an LLM judge: computed directly
from what the vector store returned, so they're real numbers on every run
regardless of which generation provider is configured. Kept separate from
run_eval.py so they're unit-testable without spinning up Chroma or a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ScoredChunk(Protocol):
    version: str
    source_page: str


def citation_hit(retrieved_chunks: list[Any], expected_citations: list[dict]) -> tuple[bool, int | None]:
    """Did any retrieved chunk match an expected citation (same k8s version +
    source doc page)? Returns (hit, 1-indexed rank of the first hit)."""
    expected_pairs = {(c["version"], Path(c["file"]).stem) for c in expected_citations}
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_slug = chunk.source_page.replace("/", "__")
        if (chunk.version, chunk_slug) in expected_pairs:
            return True, rank
    return False, None


def reciprocal_rank(rank: int | None) -> float:
    return (1.0 / rank) if rank else 0.0


def citation_coverage(retrieved_chunks: list[Any], expected_citations: list[dict]) -> float:
    """What fraction of the expected citations were actually retrieved --
    not just "was at least one of them" (that's citation_hit). For a
    single-citation question the two are identical; for a version_sensitive
    question with citations spanning several k8s versions, coverage is the
    metric that actually catches "found the newest version's doc, missed
    the older one's" -- exactly the version-blending failure mode Phase 5
    targets. See docs/ABLATIONS.md Phase 5 for why citation_hit alone was
    hiding this."""
    if not expected_citations:
        return 1.0
    expected_pairs = {(c["version"], Path(c["file"]).stem) for c in expected_citations}
    got_pairs = {(c.version, c.source_page.replace("/", "__")) for c in retrieved_chunks}
    covered = expected_pairs & got_pairs
    return len(covered) / len(expected_pairs)
