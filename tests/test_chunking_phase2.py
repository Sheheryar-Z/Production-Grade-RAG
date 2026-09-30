import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.chunking import (  # noqa: E402
    chunk_document,
    parent_document_chunk,
    recursive_chunk,
    split_into_sections,
)

SAMPLE_DOC = """## Ingress

An Ingress routes external traffic to Services.

### Deprecated annotation

Before ingressClassName existed, an annotation was used instead.

## CronJob

CronJob runs Jobs on a schedule.
"""


def test_split_into_sections_builds_header_paths():
    sections = split_into_sections(SAMPLE_DOC)
    paths = [p for p, _ in sections]
    assert "Ingress" in paths
    assert "Ingress > Deprecated annotation" in paths
    assert "CronJob" in paths


def test_split_into_sections_handles_no_headers():
    sections = split_into_sections("Just a paragraph, no headers at all.")
    assert len(sections) == 1
    assert sections[0][0] == ""


def test_recursive_chunk_never_merges_across_top_level_sections_when_large_enough():
    chunks = recursive_chunk(SAMPLE_DOC, chunk_size_tokens=400, overlap_tokens=60, min_chunk_tokens=1)
    headers = [h for h, _ in chunks]
    # each section stays its own chunk since none exceeds chunk_size_tokens
    assert any(h.startswith("Ingress") for h in headers)
    assert any(h.startswith("CronJob") for h in headers)


def test_recursive_chunk_merges_undersized_chunks_forward():
    tiny_doc = "## A\n\nHi.\n\n## B\n\nHi again.\n\n## C\n\n" + " ".join(["word"] * 100)
    chunks = recursive_chunk(tiny_doc, chunk_size_tokens=400, overlap_tokens=0, min_chunk_tokens=30)
    # sections A and B are each ~2 tokens -- both should get merged forward
    # rather than surviving as their own near-empty chunks
    assert len(chunks) < 3


def test_recursive_chunk_falls_back_to_fixed_within_an_oversized_section():
    big_section = "## Big\n\n" + "\n\n".join([" ".join(["word"] * 150)] * 6)
    chunks = recursive_chunk(big_section, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(h == "Big" for h, _ in chunks)


def test_parent_document_chunk_children_smaller_than_parent():
    doc = "## Section\n\n" + "\n\n".join([" ".join(["word"] * 100)] * 8)
    children = parent_document_chunk(doc, parent_size_tokens=1200, child_size_tokens=100, child_overlap_tokens=20)
    assert len(children) > 1
    for c in children:
        assert len(c["child_text"]) < len(c["parent_text"])
        assert c["child_text"] in c["parent_text"] or c["child_text"].strip() in c["parent_text"]


def test_parent_document_chunk_shares_parent_id_within_one_section():
    doc = "## Section\n\n" + "\n\n".join([" ".join(["word"] * 100)] * 8)
    children = parent_document_chunk(doc, parent_size_tokens=1200, child_size_tokens=100, child_overlap_tokens=20)
    parent_indices = {c["parent_index"] for c in children}
    assert parent_indices == {0}  # small doc -> one parent section


def test_chunk_document_parent_document_strategy_sets_context_text_and_parent_id():
    doc = (
        '---\nk8s_version: "1.25"\nsource_page: "concepts/ingress"\n---\n\n'
        "## Section\n\n" + "\n\n".join([" ".join(["word"] * 100)] * 8)
    )
    chunks = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=20,
                             strategy="parent_document", parent_size_tokens=1200)
    assert len(chunks) > 1
    for c in chunks:
        assert c.parent_id is not None
        assert len(c.text) <= len(c.context_text)
        assert c.metadata["k8s_version"] == "1.25"


def test_chunk_document_fixed_strategy_context_text_equals_text():
    doc = '---\nk8s_version: "1.19"\nsource_page: "x"\n---\n\nJust one short paragraph.'
    chunks = chunk_document(doc, chunk_size_tokens=400, overlap_tokens=60, strategy="fixed")
    assert len(chunks) == 1
    assert chunks[0].context_text == chunks[0].text
    assert chunks[0].parent_id is None
