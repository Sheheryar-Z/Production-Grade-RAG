"""Phase 5: query understanding, scoped to what's honestly buildable without
an LLM.

The roadmap's original Phase 5 goal was "rewriting/decomposition, optional
HyDE, for multi-hop and ambiguous questions." HyDE (generate a hypothetical
answer, embed *that*) and free-form query rewriting both need a real
generation model -- this repo's `stub` generator is extractive-only, not a
paraphraser, and OPENAI_API_KEY isn't set in this sandbox. Rather than fake
either technique with the stub, or skip the phase, this module implements
the one form of query decomposition that's genuinely structural and doesn't
need an LLM at all: **per-version fan-out**.

The problem it targets is real and was measured before writing any code
(see docs/ABLATIONS.md Phase 5): this corpus's `version_sensitive` questions
("What apiVersion should I use for Ingress?") don't name a version, and a
single query against the whole corpus doesn't reliably surface evidence
from *every* version whose docs actually differ -- whichever version's
phrasing happens to score highest crowds out the others. Measured baseline:
mean citation coverage 77.8% across the 6 version_sensitive golden
questions (vs. citation_hit, which only checks "hit at least one expected
citation" and was already near 100% -- it was hiding this).

The fix doesn't require understanding the query semantically: the corpus's
version list is already known (`corpus.versions` in config), so a question
that doesn't pin a version can simply be asked once per version, with the
retriever's own `version_filter` doing the isolation Chroma/BM25 already
support. No new model, no new embedding call shape -- just more retrieval
calls, which is a real, measured latency cost (see docs/ABLATIONS.md),
not a free lunch.
"""
from __future__ import annotations

import re

from k8s_rag.config import RunConfig
from k8s_rag.vectorstore import RetrievedChunk, dedupe_by_parent

# A bare "1.24", "v1.24", "1.24.0" etc. -- deliberately permissive (doesn't
# require it to be one of the corpus's actual versions) because the point is
# detecting *whether the user already scoped the question*, not validating
# the version. "1.40" (not in this corpus, see ua-003) still counts as
# scoped -- the pipeline should search 1.40 and correctly find nothing,
# not silently fan out across every version instead.
_VERSION_MENTION_RE = re.compile(r"\bv?1\.\d{1,2}(?:\.\d{1,2})?\b")


def question_names_a_version(question: str) -> bool:
    return bool(_VERSION_MENTION_RE.search(question))


def needs_version_fanout(question: str, target_version: str | None, cfg: RunConfig) -> bool:
    """True when the question is a version-sensitive candidate: no version
    was passed explicitly (--version / target_version), and none is named
    in the question text either. This is a text-pattern heuristic, not
    semantic understanding -- it will over-trigger on a question that's
    genuinely version-independent (e.g. "what is a Pod?") and under-trigger
    if someone phrases a version some other way ("the latest release").
    Both are honest limitations of a heuristic, not bugs to hide."""
    if not cfg.get("query_understanding", "enabled", default=False):
        return False
    if target_version:
        return False
    return not question_names_a_version(question)


def fanout_search(retriever, question: str, cfg: RunConfig, top_k: int) -> list[RetrievedChunk]:
    """One retrieval call per corpus version, then merge so every version
    gets a guaranteed shot at contributing its best hit before any single
    version's runner-up hits crowd out another version's top hit.

    Strategy: fetch a few candidates per version, take each version's #1
    hit first (guarantees coverage), then fill remaining top_k budget with
    the next-best hits pooled across all versions by score. Simple and
    auditable rather than a clever weighted merge -- the golden-set ablation
    (docs/ABLATIONS.md Phase 5) is what actually justifies this, not the
    strategy's elegance.
    """
    versions: list[str] = cfg.get("corpus", "versions", default=[])
    if not versions:
        return retriever.search(question, top_k=top_k, version_filter=None)

    per_version_k = max(2, (top_k // len(versions)) + 1)
    per_version_hits: list[list[RetrievedChunk]] = []
    for v in versions:
        hits = retriever.search(question, top_k=per_version_k, version_filter=v)
        if hits:
            per_version_hits.append(hits)

    if not per_version_hits:
        return []

    merged: list[RetrievedChunk] = [hits[0] for hits in per_version_hits]
    leftover = sorted(
        (c for hits in per_version_hits for c in hits[1:]),
        key=lambda c: c.score,
        reverse=True,
    )
    seen_ids = {c.chunk_id for c in merged}
    for c in leftover:
        if len(merged) >= top_k:
            break
        if c.chunk_id in seen_ids:
            continue
        merged.append(c)
        seen_ids.add(c.chunk_id)

    merged.sort(key=lambda c: c.score, reverse=True)
    return dedupe_by_parent(merged, top_k)
