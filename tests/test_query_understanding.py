"""Phase 5: per-version fan-out. Unit tests for the heuristic (what counts
as 'the question already named a version') and the merge logic (every
version's #1 hit is guaranteed a slot), using fakes -- no Chroma, no real
retriever, so these run in milliseconds like the rest of the suite."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.config import RunConfig  # noqa: E402
from k8s_rag.query_understanding import (  # noqa: E402
    fanout_search,
    needs_version_fanout,
    question_names_a_version,
)


def _cfg(enabled: bool, versions: list[str] | None = None) -> RunConfig:
    return RunConfig(
        raw={
            "query_understanding": {"enabled": enabled},
            "corpus": {"versions": versions if versions is not None else ["1.21", "1.24", "1.25"]},
        },
        path=None,  # not read by these tests
    )


def test_question_names_a_version_true_for_explicit_version():
    assert question_names_a_version("What apiVersion in Kubernetes 1.31?")
    assert question_names_a_version("Is this available as of v1.28?")
    assert question_names_a_version("On 1.9 was this present?")


def test_question_names_a_version_false_for_unscoped_question():
    assert not question_names_a_version("What apiVersion should I use for Ingress?")
    assert not question_names_a_version("Is PodSecurityPolicy something I should use?")


def test_needs_version_fanout_requires_enabled():
    cfg = _cfg(enabled=False)
    assert not needs_version_fanout("What apiVersion for Ingress?", None, cfg)


def test_needs_version_fanout_false_when_target_version_given():
    cfg = _cfg(enabled=True)
    assert not needs_version_fanout("What apiVersion for Ingress?", "1.31", cfg)


def test_needs_version_fanout_false_when_question_names_a_version():
    cfg = _cfg(enabled=True)
    assert not needs_version_fanout("What apiVersion in Kubernetes 1.31?", None, cfg)


def test_needs_version_fanout_true_for_unscoped_question():
    cfg = _cfg(enabled=True)
    assert needs_version_fanout("What apiVersion should I use for Ingress?", None, cfg)


@dataclass
class _FakeChunk:
    chunk_id: str
    version: str
    score: float
    parent_id: str | None = None


class _FakeRetriever:
    """Returns a fixed, per-version ranked list so the merge logic can be
    tested without a real vector store."""

    def __init__(self, by_version: dict[str, list[_FakeChunk]]):
        self.by_version = by_version
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query, top_k, version_filter=None):
        self.calls.append((query, top_k, version_filter))
        return self.by_version.get(version_filter, [])[:top_k]


def test_fanout_search_guarantees_one_hit_per_version():
    # version "1.25"'s best hit scores lower than version "1.21"'s *second*
    # hit -- a plain top-k-by-score merge would drop 1.25 entirely.
    by_version = {
        "1.21": [_FakeChunk("a1", "1.21", 0.90), _FakeChunk("a2", "1.21", 0.85)],
        "1.24": [_FakeChunk("b1", "1.24", 0.80)],
        "1.25": [_FakeChunk("c1", "1.25", 0.40)],
    }
    retriever = _FakeRetriever(by_version)
    cfg = _cfg(enabled=True, versions=["1.21", "1.24", "1.25"])

    result = fanout_search(retriever, "unscoped question", cfg, top_k=8)
    versions_present = {c.version for c in result}

    assert versions_present == {"1.21", "1.24", "1.25"}
    # one retrieval call per corpus version, not one call total
    assert len(retriever.calls) == 3
    assert all(v == vf for _, _, vf in retriever.calls for v in [vf] if vf)


def test_fanout_search_respects_top_k_budget():
    by_version = {
        "1.21": [_FakeChunk(f"a{i}", "1.21", 1.0 - i * 0.01) for i in range(5)],
        "1.24": [_FakeChunk(f"b{i}", "1.24", 0.9 - i * 0.01) for i in range(5)],
        "1.25": [_FakeChunk(f"c{i}", "1.25", 0.8 - i * 0.01) for i in range(5)],
    }
    retriever = _FakeRetriever(by_version)
    cfg = _cfg(enabled=True, versions=["1.21", "1.24", "1.25"])

    result = fanout_search(retriever, "unscoped question", cfg, top_k=4)
    assert len(result) <= 4


def test_fanout_search_empty_versions_falls_back_to_plain_search():
    cfg = _cfg(enabled=True, versions=[])
    retriever = _FakeRetriever({None: [_FakeChunk("x", "1.31", 0.5)]})
    result = fanout_search(retriever, "unscoped question", cfg, top_k=8)
    assert [c.chunk_id for c in result] == ["x"]
