"""Chunking strategies, ablated against each other on the golden eval set
before any one of them gets called "the" chunking strategy (see
docs/ABLATIONS.md Phase 2 for the actual numbers).

Token counts here are approximated as whitespace-split words * 1.3 (a rough
words-to-tokens ratio for English prose) to avoid a hard tokenizer dependency.
That approximation is a documented simplification, not a hidden one -- swap
in `tiktoken` if exact counts start to matter for a real generation model's
context window.

Three strategies:

- `fixed` (Phase 1): a plain sliding window over paragraphs, no awareness of
  document structure. The floor every other strategy has to beat.
- `recursive`: splits on markdown headers first (so a chunk never silently
  spans two different sections), then falls back to `fixed`'s paragraph
  packing *within* a section if that section is still too big. Small
  leftover sections get merged forward rather than shipped as noisy
  near-empty chunks.
- `parent_document`: embeds small, precise "child" chunks (good for
  matching a specific question closely) but returns their larger "parent"
  section as the context handed to generation (good for not starving the
  answer of surrounding detail). This is the one strategy here that isn't
  just a different way to cut the same text -- retrieval and generation
  read different-sized units on purpose.

`semantic` chunking (embedding-similarity-based split points) is
deliberately not implemented: it would make the chunking ablation circular
with the embedding-provider ablation already run in Phase 1 (which
embedding model decides where a *different* embedding model's chunks
break?). Documented as a real gap, not silently skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

WORDS_TO_TOKENS = 1.3


@dataclass
class Chunk:
    text: str                       # what gets embedded and matched against a query
    chunk_index: int
    approx_tokens: int
    metadata: dict
    context_text: str = ""          # what gets handed to generation; defaults to `text`
    parent_id: str | None = None    # shared by every child chunk of the same parent section

    def __post_init__(self) -> None:
        if not self.context_text:
            self.context_text = self.text


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_TO_TOKENS)


def strip_front_matter(text: str) -> tuple[dict, str]:
    """Corpus files start with a YAML front-matter block written by
    fetch_corpus.py. Pull it out as metadata and return the body separately."""
    m = re.match(r"^---\n(.*?)\n---\n\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


def _strip_both_front_matters(raw_text: str) -> tuple[dict, str]:
    # ours, then the kubernetes/website page's own Hugo front matter -- see
    # the note in chunk_document's history for why both need stripping.
    meta, after_ours = strip_front_matter(raw_text)
    hugo_meta, body = strip_front_matter(after_ours)
    return {**meta, **hugo_meta}, body


def fixed_chunk(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Word-based sliding window, sized in approximate tokens. Splits on
    paragraph boundaries where possible so a chunk doesn't start or end
    mid-sentence more often than it has to."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > chunk_size_tokens and current:
            chunks.append("\n\n".join(current))
            # carry the tail of the current chunk forward as overlap
            overlap_words: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                w = p.split()
                overlap_words = w + overlap_words if overlap_count == 0 else overlap_words
                overlap_count += _approx_tokens(p)
                if overlap_count >= overlap_tokens:
                    break
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = _approx_tokens(current[0]) if current else 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return [c.strip() for c in chunks if c.strip()]


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headers. Returns [(header_path, section_text)],
    where header_path is e.g. "Ingress > Deprecated annotation" and
    section_text includes that header line plus everything under it, up to
    the next header of the same or higher level. Text before the first
    header (if any) gets header_path ""."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)] if text.strip() else []

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    # track a stack of (level, title) to build a breadcrumb path
    path_stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[m.start():end].strip()

        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        header_path = " > ".join(t for _, t in path_stack)

        if section_text:
            sections.append((header_path, section_text))
    return sections


def recursive_chunk(text: str, chunk_size_tokens: int, overlap_tokens: int,
                     min_chunk_tokens: int = 40) -> list[tuple[str, str]]:
    """Returns [(header_path, chunk_text)]. Never lets a chunk span two
    top-level sections if it doesn't have to; falls back to fixed_chunk's
    paragraph packing inside a section that's still too big on its own;
    merges sections that end up too small forward into their neighbor
    rather than shipping noisy near-empty chunks."""
    sections = split_into_sections(text)
    raw_chunks: list[tuple[str, str]] = []
    for header_path, section_text in sections:
        if _approx_tokens(section_text) <= chunk_size_tokens:
            raw_chunks.append((header_path, section_text))
        else:
            for piece in fixed_chunk(section_text, chunk_size_tokens, overlap_tokens):
                raw_chunks.append((header_path, piece))

    # merge undersized chunks forward so a 15-token orphan heading doesn't
    # become its own chunk
    merged: list[tuple[str, str]] = []
    pending: tuple[str, str] | None = None
    for header_path, chunk_text in raw_chunks:
        if pending is None:
            pending = (header_path, chunk_text)
            continue
        if _approx_tokens(pending[1]) < min_chunk_tokens:
            pending = (pending[0], pending[1] + "\n\n" + chunk_text)
        else:
            merged.append(pending)
            pending = (header_path, chunk_text)
    if pending is not None:
        merged.append(pending)
    return merged


def parent_document_chunk(text: str, parent_size_tokens: int, child_size_tokens: int,
                           child_overlap_tokens: int) -> list[dict]:
    """Two passes: first cut the document into "parent" sections (large,
    header-aware -- reuses recursive_chunk's section logic at a bigger
    size), then cut each parent into smaller "child" pieces that are what
    actually get embedded. Returns a flat list of
    {child_text, parent_text, parent_index, header_path}."""
    parents = recursive_chunk(text, parent_size_tokens, overlap_tokens=0, min_chunk_tokens=80)
    out: list[dict] = []
    for parent_index, (header_path, parent_text) in enumerate(parents):
        children = fixed_chunk(parent_text, child_size_tokens, child_overlap_tokens)
        for child_text in children:
            out.append({
                "child_text": child_text,
                "parent_text": parent_text,
                "parent_index": parent_index,
                "header_path": header_path,
            })
    return out


def chunk_document(raw_text: str, chunk_size_tokens: int, overlap_tokens: int,
                    strategy: str = "fixed", parent_size_tokens: int | None = None) -> list[Chunk]:
    meta, body = _strip_both_front_matters(raw_text)

    if strategy == "fixed":
        pieces = fixed_chunk(body, chunk_size_tokens, overlap_tokens)
        return [
            Chunk(text=p, chunk_index=i, approx_tokens=_approx_tokens(p), metadata=meta)
            for i, p in enumerate(pieces)
        ]

    if strategy == "recursive":
        pieces = recursive_chunk(body, chunk_size_tokens, overlap_tokens)
        return [
            Chunk(
                text=chunk_text, chunk_index=i, approx_tokens=_approx_tokens(chunk_text),
                metadata={**meta, "section": header_path} if header_path else dict(meta),
            )
            for i, (header_path, chunk_text) in enumerate(pieces)
        ]

    if strategy == "parent_document":
        if not parent_size_tokens:
            raise ValueError("parent_document strategy requires parent_size_tokens")
        children = parent_document_chunk(body, parent_size_tokens, chunk_size_tokens, overlap_tokens)
        out = []
        for i, c in enumerate(children):
            child_meta = {**meta, "section": c["header_path"]} if c["header_path"] else dict(meta)
            out.append(Chunk(
                text=c["child_text"], context_text=c["parent_text"], chunk_index=i,
                approx_tokens=_approx_tokens(c["child_text"]), metadata=child_meta,
                parent_id=f"p{c['parent_index']}",
            ))
        return out

    raise ValueError(f"Unknown chunking strategy: {strategy}")
