# Golden evaluation set

`golden_qa.jsonl` is the Phase 0 seed set: 20 hand-authored questions, each
grounded in the actual fetched corpus (facts were checked against the real
doc text with `grep`, not written from memory and hoped to be right — see the
project roadmap doc's Phase 0 notes for why that matters).

## Schema

| Field | Meaning |
| --- | --- |
| `id` | stable id, prefixed by category (`sv-`, `vs-`, `ua-`) |
| `category` | `single_version` \| `version_sensitive` \| `unanswerable` |
| `target_version` | the Kubernetes version the question is about, or `null` when it deliberately doesn't specify one (that's the point of `version_sensitive` questions) |
| `question` | the question text |
| `expected_answer` | the ground-truth answer a correct system should produce |
| `expected_citations` | the corpus file(s) that should be cited |
| `notes` | why this question is in the set / what failure mode it's designed to catch |

## Category breakdown (seed set: 20 questions)

- **single_version (8)** — one unambiguous correct answer, pinned to one Kubernetes version. Scored on faithfulness, citation precision/recall, answer relevancy.
- **version_sensitive (6)** — the correct answer depends on which version is being asked about, and several don't specify a version at all. This is the hardest category and the one that actually separates a naive RAG system (which blends chunks from different versions into one confident-sounding but wrong answer) from a well-built one. Scored the same as single_version, plus a manual check that version-specific answers weren't blended.
- **unanswerable (6)** — out-of-scope questions, a future/past version not in the corpus, and one prompt-injection attempt. Scored on refusal precision/recall, not answer accuracy. `ua-004` is scored separately as an adversarial/injection test, not a plain coverage test.

## Growing this set

20 questions is enough to unblock Phase 1-5 development with a real signal,
but it's a seed, not the finished set. Before Phase 7 (eval-as-CI) is
meaningful as a regression gate, grow this to 60-100 by:

1. Adding more curated pages to `scripts/fetch_corpus.py`'s manifest (more
   API objects with real version history: StorageClass, NetworkPolicy,
   admission webhooks, feature gate promotions).
2. Writing 2-3 questions per new page the same way these were written:
   grep the actual fetched file for the fact, don't write from memory.
3. Keeping the category balance roughly the same — version_sensitive and
   unanswerable questions are more valuable per-question than single_version
   ones, since they're what a generic RAG demo gets wrong.
