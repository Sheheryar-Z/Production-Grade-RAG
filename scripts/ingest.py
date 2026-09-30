#!/usr/bin/env python3
"""Phase 1/2 — chunk the fetched corpus and load it into ChromaDB."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k8s_rag.bm25_index import BM25Index, bm25_persist_path  # noqa: E402
from k8s_rag.chunking import chunk_document  # noqa: E402
from k8s_rag.config import REPO_ROOT, RunConfig  # noqa: E402
from k8s_rag.embeddings import get_embedding_provider, tfidf_persist_path  # noqa: E402
from k8s_rag.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    cfg = RunConfig.load()
    corpus_dir = REPO_ROOT / "data" / "corpus"
    chunk_size = cfg.get("chunking", "chunk_size_tokens", default=400)
    overlap = cfg.get("chunking", "chunk_overlap_tokens", default=60)
    strategy = cfg.get("chunking", "strategy", default="fixed")
    parent_size = cfg.get("chunking", "parent_size_tokens", default=1200)
    if strategy not in ("fixed", "recursive", "parent_document"):
        raise NotImplementedError(
            f"chunking strategy '{strategy}' isn't implemented. "
            "Use one of: fixed, recursive, parent_document."
        )

    print(f"Chunking with strategy={strategy}, size={chunk_size} tokens, overlap={overlap} tokens"
          + (f", parent_size={parent_size} tokens" if strategy == "parent_document" else ""))

    all_chunks: list[dict] = []
    md_files = sorted(corpus_dir.glob("*/*.md"))
    for md_file in md_files:
        raw = md_file.read_text()
        doc_chunks = chunk_document(
            raw, chunk_size, overlap, strategy=strategy, parent_size_tokens=parent_size,
        )
        doc_prefix = f"{md_file.parent.name}__{md_file.stem}"
        for c in doc_chunks:
            version = c.metadata.get("k8s_version", md_file.parent.name)
            source_page = c.metadata.get("source_page", md_file.stem)
            source_url = c.metadata.get("source_url", "")
            chunk_id = f"{doc_prefix}__{c.chunk_index}"
            # namespace parent_id with the source doc so dedup in
            # VectorStore.search() never collides parents across documents
            parent_id = f"{doc_prefix}__{c.parent_id}" if c.parent_id else ""
            all_chunks.append({
                "id": chunk_id, "text": c.text, "context_text": c.context_text,
                "version": version, "source_page": source_page, "source_url": source_url,
                "parent_id": parent_id,
            })
        print(f"  {md_file.relative_to(REPO_ROOT)} -> {len(doc_chunks)} chunks")

    if cfg.get("embedding", "provider", default="sentence_transformers") == "local_tfidf":
        embedder = get_embedding_provider(cfg)
        path = tfidf_persist_path(cfg)
        embedder.fit_and_persist([c["text"] for c in all_chunks], path)
        print(f"\nFit TF-IDF vectorizer on {len(all_chunks)} chunks -> {path.relative_to(REPO_ROOT)}")

    bm25_enabled = cfg.get("retrieval", "bm25", "enabled", default=False) or cfg.get("retrieval", "mode") == "hybrid"
    if bm25_enabled:
        bm25 = BM25Index()
        path = bm25_persist_path(cfg)
        bm25.fit_and_persist(all_chunks, path)
        print(f"Fit BM25 index on {len(all_chunks)} chunks -> {path.relative_to(REPO_ROOT)}")

    store = VectorStore(cfg)
    store.reset()

    print(f"\nEmbedding and upserting {len(all_chunks)} chunks into Chroma "
          f"(collection={cfg.get('vector_store', 'collection_name')})...")
    # batch to keep memory/API-call size sane
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        store.add_chunks(all_chunks[i:i + batch_size])
        print(f"  upserted {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

    print(f"\nDone. Collection now has {store.count()} chunks.")


if __name__ == "__main__":
    main()
