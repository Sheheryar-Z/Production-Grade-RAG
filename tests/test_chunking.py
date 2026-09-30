import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.chunking import chunk_document, fixed_chunk, strip_front_matter  # noqa: E402


def test_strip_front_matter_extracts_metadata():
    text = '---\nk8s_version: "1.25"\nsource_page: "concepts/ingress"\n---\n\nBody text here.'
    meta, body = strip_front_matter(text)
    assert meta["k8s_version"] == "1.25"
    assert meta["source_page"] == "concepts/ingress"
    assert body.strip() == "Body text here."


def test_strip_front_matter_handles_no_front_matter():
    meta, body = strip_front_matter("Just a body, no front matter.")
    assert meta == {}
    assert body == "Just a body, no front matter."


def test_chunk_document_strips_both_front_matter_blocks():
    # ours, then the kubernetes/website page's own Hugo front matter
    text = (
        '---\nk8s_version: "1.25"\nsource_page: "concepts/ingress"\n---\n\n'
        '---\ntitle: Ingress\nreviewers:\n- bprashanth\n---\n\n'
        "## Ingress\n\nAn Ingress needs apiVersion, kind, metadata and spec fields.\n\n"
        "It routes external HTTP(S) traffic to Services within the cluster."
    )
    chunks = chunk_document(text, chunk_size_tokens=400, overlap_tokens=60)
    assert len(chunks) == 1
    assert "title: Ingress" not in chunks[0].text
    assert "reviewers" not in chunks[0].text
    assert "Ingress needs apiVersion" in chunks[0].text
    assert chunks[0].metadata["k8s_version"] == "1.25"


def test_fixed_chunk_splits_long_text_into_multiple_pieces():
    # ~150 words per paragraph, 6 paragraphs -> well over a 100-token chunk size
    paragraph = " ".join(["word"] * 150)
    text = "\n\n".join([paragraph] * 6)
    chunks = fixed_chunk(text, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # every chunk should be non-empty and roughly within a reasonable size
    for c in chunks:
        assert c.strip()


def test_fixed_chunk_keeps_short_text_as_one_chunk():
    text = "A short paragraph.\n\nAnother short paragraph."
    chunks = fixed_chunk(text, chunk_size_tokens=400, overlap_tokens=60)
    assert len(chunks) == 1
