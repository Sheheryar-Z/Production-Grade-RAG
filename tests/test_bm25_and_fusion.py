import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.bm25_index import tokenize  # noqa: E402
from k8s_rag.retrieval import rrf_fuse  # noqa: E402


def test_tokenize_keeps_api_identifiers_intact():
    tokens = tokenize("The policy/v1beta1 API version is removed in v1.25.")
    assert "policy/v1beta1" in tokens
    assert "v1.25" in tokens or "1.25" in " ".join(tokens)


def test_tokenize_lowercases():
    assert tokenize("PodDisruptionBudget") == ["poddisruptionbudget"]


def test_rrf_fuse_rewards_items_in_both_lists():
    vector_ranked = ["a", "b", "c"]
    bm25_ranked = ["b", "a", "d"]
    scores = rrf_fuse([vector_ranked, bm25_ranked])
    # "a" and "b" appear in both lists -> should outscore "c" and "d", which
    # appear in only one
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]


def test_rrf_fuse_rewards_higher_rank():
    vector_ranked = ["x", "y"]
    scores = rrf_fuse([vector_ranked])
    assert scores["x"] > scores["y"]


def test_rrf_fuse_handles_disjoint_lists():
    scores = rrf_fuse([["a", "b"], ["c", "d"]])
    assert set(scores.keys()) == {"a", "b", "c", "d"}
    assert scores["a"] == scores["c"]  # both rank 1 in their own list


def test_rrf_fuse_empty_lists():
    assert rrf_fuse([[], []]) == {}
