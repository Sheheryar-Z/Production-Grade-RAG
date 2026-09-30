import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.eval_metrics import citation_hit, reciprocal_rank  # noqa: E402


@dataclass
class FakeChunk:
    version: str
    source_page: str


def test_citation_hit_finds_match_and_reports_rank():
    retrieved = [
        FakeChunk(version="1.25", source_page="concepts/workloads/controllers/deployment"),
        FakeChunk(version="1.25", source_page="reference/using-api/deprecation-guide"),
    ]
    expected = [{"version": "1.25", "file": "1.25/reference__using-api__deprecation-guide.md"}]
    hit, rank = citation_hit(retrieved, expected)
    assert hit is True
    assert rank == 2


def test_citation_hit_no_match_returns_false_and_none():
    retrieved = [FakeChunk(version="1.19", source_page="concepts/workloads/controllers/deployment")]
    expected = [{"version": "1.25", "file": "1.25/reference__using-api__deprecation-guide.md"}]
    hit, rank = citation_hit(retrieved, expected)
    assert hit is False
    assert rank is None


def test_citation_hit_requires_matching_version_not_just_page():
    # same page, wrong version -- must NOT count as a hit (this is the exact
    # failure mode the version-sensitive eval questions are designed to catch)
    retrieved = [FakeChunk(version="1.19", source_page="concepts/services-networking/ingress")]
    expected = [{"version": "1.31", "file": "1.31/concepts__services-networking__ingress.md"}]
    hit, rank = citation_hit(retrieved, expected)
    assert hit is False


def test_citation_hit_empty_expected_citations_never_hits():
    retrieved = [FakeChunk(version="1.25", source_page="concepts/workloads/controllers/deployment")]
    hit, rank = citation_hit(retrieved, [])
    assert hit is False
    assert rank is None


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0
