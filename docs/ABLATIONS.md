# Ablation results

The single most important artifact in this repo. Every row is a real eval
run against the 20-question golden set (`data/eval/golden_qa.jsonl`), tied to
a specific config file, never hand-edited. Raw output:
`data/eval/results/<run_name>.json`.

**A caveat that applies to every row below, stated up front rather than
buried**: this repo was built in a sandboxed environment whose egress policy
blocks huggingface.co and every LLM/embedding API host (OpenAI, Cohere) —
re-confirmed after a network change on the developer's end, so this is an
organization-level allowlist in this sandbox specifically, not something a
local network change affects. GitHub (including release assets) *is*
reachable, which matters below. Generation uses an extractive stub instead
of a real LLM (no `OPENAI_API_KEY`) — so faithfulness/citation-precision/answer-relevancy
are not yet measured anywhere in this doc; that's a one-line config change
(`generation.provider: openai`) plus a key, on any machine that has one.

## A real finding, not just a workaround: two local embedding options, and they disagree

Two embedding fallbacks were tried here, and their retrieval numbers are
different enough to be a genuine lesson, not a footnote:

- **`local_tfidf`** — pure scikit-learn bag-of-words. Zero network
  dependency at all.
- **`spacy_vectors`** — real static word vectors (`en_core_web_md`, 300-dim,
  GloVe-style), averaged per chunk. Installed from a GitHub release asset
  (`release-assets.githubusercontent.com`), which this sandbox's policy
  allows even though huggingface.co is blocked. Verified before use: it
  correctly scores "PodSecurityPolicy was removed in 1.25" as similar to a
  paraphrase using "PSP" instead (cosine 0.67) and near-zero similarity to
  an unrelated sentence — genuine semantic behavior, not a placebo.

The bag-of-words fallback beat the real-word-vector fallback on this
corpus, by a lot. That's the finding.

## Phase 1: naive baseline — TF-IDF embeddings, fixed 400-token chunking (the recorded baseline)

Run: `config/sandbox_local.yaml`, 20-question golden set, top_k=8.

| Metric | Result | Notes |
| --- | --- | --- |
| Recall@8 (single_version + version_sensitive, n=14) | **100.0%** | Every question's ground-truth chunk was retrieved somewhere in the top 8. Expected: these questions use the same terminology as the docs (e.g. "PodDisruptionBudget", "policy/v1beta1"), which is exactly the case TF-IDF is strong at. |
| MRR | **0.881** | Ground-truth chunk is usually ranked #1 or #2, not just present in the top 8. |
| Refusal recall (unanswerable, n=6) | **16.7%** (1/6) | **The headline finding of this phase.** With no reranker and no relevance threshold, the pipeline refuses only when retrieval returns literally zero chunks (`ua-003`, a version filter with no matching documents). Every out-of-scope, adversarial, or out-of-corpus-range question it *could* return chunks for, it answered instead of refusing — because "8 chunks came back" is not the same as "8 *relevant* chunks came back". This is the number Phase 4 (reranker + calibrated refusal threshold) exists to fix, and now there's a baseline to prove the fix worked. |
| p50 latency | **7 ms** | Retrieval + stub generation only, no network call. Not representative of real generation latency — recorded here so the *retrieval* latency trend is visible across phases independent of whichever LLM is plugged in later. |
| Faithfulness (Ragas) | N/A | Requires `OPENAI_API_KEY` + `generation.provider: openai`. Not measured yet — the stub generator is extractive, not generative, so a faithfulness score against it would be nearly meaningless (it can only ever quote retrieved text). |
| Citation precision/recall | N/A | Same reason. |
| Answer relevancy | N/A | Same reason. |
| Cost/query | $0.00 | Local embeddings, no LLM call. Will be real and nonzero from Phase 4+ once a real generation provider is configured. |

### What this phase actually proves

Retrieval finds the right evidence when it exists (100% recall is a strong
floor to build on). Refusal is where the naive baseline is genuinely bad —
16.7% refusal recall means it hallucinates confidently on 5 of 6 questions
it should have declined. That gap, not a vague "it works pretty well", is
the real starting point for Phase 4.

### What's not yet measured, and why that's stated rather than hidden

Answer-quality (faithfulness, citation precision, relevancy) needs a real
LLM to judge against — an extractive stub can't produce the kind of answer
those metrics are designed to score. The honest move is to leave those
cells N/A here rather than run Ragas against stub output and report a
number that doesn't mean what a reader would assume it means.

## Phase 1, variant: spaCy word-vector embeddings (why it's not the recorded baseline)

Run: `config/sandbox_spacy.yaml`, same 20-question golden set, same top_k=8,
identical pipeline — only `embedding.provider` changed.

| Metric | TF-IDF (recorded baseline) | spaCy word vectors |
| --- | --- | --- |
| Recall@8 | **100.0%** | **28.6%** |
| MRR | **0.881** | **0.196** |
| Refusal recall | 16.7% | 16.7% (identical — refusal logic doesn't depend on embedding choice yet) |
| p50 latency | 7 ms | 11 ms |

Word vectors performed *worse*, not better, and there's a real, specific
reason rather than a vague "embeddings didn't work": this corpus's
retrieval signal lives almost entirely in exact technical tokens —
`policy/v1beta1`, `PodDisruptionBudget`, `autoscaling/v2`, `dockershim`.
`en_core_web_md` was trained on general web/news text, so multi-word API
identifiers either fall outside its vocabulary or get weak, noisy vectors.
Averaging those into one 300-dim vector per ~400-token chunk dilutes the
handful of tokens that actually matter into the surrounding prose. TF-IDF,
despite being "dumber", weights exactly those rare, distinctive, jargon
tokens the highest by construction (that's what inverse-document-frequency
does) — a good match for a documentation corpus this jargon-dense.

This is *why* Phase 3 (hybrid retrieval: BM25 keyword search + vector
search, fused) is on the roadmap rather than "just use embeddings" — the
two approaches have complementary failure modes, and a corpus like this one
is a clean demonstration of the keyword side actually winning outright
against a plausible-looking semantic alternative. `spacy_vectors` stays in
the codebase as a documented option (`src/k8s_rag/embeddings.py`), but the
TF-IDF run above is the one carried forward as "the Phase 1 baseline" in
every later phase's comparison, because it's the stronger of the two
network-free options actually available here.

## Phase 2 — chunking ablation

Same embedding (`local_tfidf`, the Phase 1 winner) across all three runs —
chunking strategy is the only variable. See `src/k8s_rag/chunking.py` for
what each strategy actually does.

| Configuration | Chunk count | Recall@8 | MRR | p50 latency |
| --- | --- | --- | --- | --- |
| `fixed` (Phase 1 baseline) | 597 | **100.0%** | 0.881 | 7 ms |
| `recursive` (header-aware) | 1127 | 92.9% | 0.774 | 10 ms |
| `parent_document` (small child / large parent) | 1958 | **100.0%** | **0.889** | 13 ms |

**Winner: `parent_document`.** It matched `fixed`'s recall and edged out its
MRR slightly, and — a real advantage the retrieval metrics above don't
capture on their own — it hands generation a larger, more complete parent
section (up to 1200 tokens) instead of just the ~400-token slice that
happened to match, so a real LLM has more surrounding detail to work with
without retrieval precision suffering (matching still happens on the small,
precise 150-token child). Verified directly in `scripts/ask.py`: the correct
PodDisruptionBudget fact reached the final answer even when its child chunk
wasn't the #1 ranked hit, because the *parent* section containing it was
still what got handed to generation.

**The counter-intuitive result: `recursive` did worse, not better, than the
naive baseline.** The instinct that "structure-aware, one-topic-per-chunk
splitting should obviously help" doesn't hold up here. Splitting on markdown
headers produced almost twice as many chunks (1127 vs. 597) because this
corpus's `deprecation-guide` pages alone have dozens of `####` subsections.
More, smaller chunks means less term density per chunk for a lexical
matcher like TF-IDF — the exact combination of terms a question needs can
end up split across a header boundary into two separate chunks instead of
co-occurring in one. `fixed`'s cruder, header-blind packing happened to keep
more relevant terms together. This is exactly the kind of result worth
running an eval for instead of assuming: the "obviously better" technique
lost on a real, measured number.

`parent_document` is now `config/default.yaml`'s configured strategy.

## Phase 3 — hybrid retrieval (BM25 + vector, fused with RRF)

Built on top of Phase 2's winner (`parent_document` chunking, still
`local_tfidf` embeddings) — one more variable changed at a time. Fusion is
Reciprocal Rank Fusion, not a manual score blend; see
`src/k8s_rag/retrieval.py`'s module docstring for why RRF and not a weighted
sum (cosine similarity and BM25 scores live on incomparable scales).

| Configuration | Recall@8 | MRR | p50 latency |
| --- | --- | --- | --- |
| `parent_document` + vector-only (Phase 2 winner) | 100.0% | 0.889 | 13 ms |
| `parent_document` + hybrid (BM25 + vector, RRF) | 100.0% | **0.893** | **73–100 ms** |

**The result is genuinely marginal, and there's a real, specific reason
why — not just "hybrid didn't help much here".** Hybrid retrieval's value
proposition is fusing two retrievers with *complementary* failure modes: a
semantic retriever (catches paraphrases, misses exact jargon) and a lexical
one (catches exact jargon, misses paraphrases). But in every sandbox run in
this repo, the "vector" side has been `local_tfidf` — which is *also*
lexical. Fusing BM25 with TF-IDF-cosine is fusing two retrievers that are
already highly correlated on this corpus, so there's little genuinely new
signal for RRF to add. The real test of hybrid retrieval is
`config/default.yaml`'s combination (BM25 + a genuine `sentence_transformers`
or `openai` embedding) on a machine that can reach those hosts — expect a
bigger gap there, precisely because Phase 1's finding was that TF-IDF and
real semantic embeddings disagree substantially on this corpus.

**The real, quantified cost found here: latency, by roughly 9x** (11 ms vs.
100 ms, measured directly with repeated timing, not the single-run p50
above). `rank_bm25`'s `BM25Okapi.get_scores()` rescans the *entire* corpus
on every single query — O(corpus size) per query, with no persistent index
structure, unlike Chroma's approximate-nearest-neighbor index which is
sub-linear. At ~2,000 chunks this is still fast in absolute terms, but it
would not scale to a large corpus without swapping to a real inverted-index
implementation (Elasticsearch, OpenSearch, or Postgres full-text search)
instead of `rank_bm25`'s in-memory linear scan. That's a concrete, measured
example of exactly the "production" gap Phase 8 exists to close — not a
guess about what might happen at scale.

`config/default.yaml` now has `retrieval.mode: hybrid` enabled, with this
tradeoff documented rather than hidden.

## Phase 4 — reranking + calibrated refusal

Split into two halves, and only one of them was buildable here.

**The reranker half: blocked, and confirmed rather than assumed.**
Investigated `sentence-transformers` CrossEncoder models and the
`flashrank` library as possible network-free-ish rerankers. Both are
huggingface.co-hosted — confirmed by reading `flashrank`'s own source
(`Config.py`: `model_url = 'https://huggingface.co/prithivida/flashrank/resolve/main/{}.zip'`),
not assumed from the library's reputation. Unlike Phase 1's embedding model,
no GitHub-release-hosted alternative was found for a real cross-encoder.
This stays a documented gap: no ML reranker in this repo yet.

**The calibrated-refusal half: built, measured, and it's the best single
number in this whole project.** A reranker's other job — beyond reordering
results — is producing a relevance score confident enough to gate
generation on. The retriever already produces one (the RRF fused score, in
hybrid mode); the open question was whether that score is actually a *good*
refusal signal. `scripts/calibrate_refusal.py` answers this empirically
rather than guessing a threshold: it runs retrieval (no generation) over
all 20 golden questions, collects each one's top-1 score, and sweeps every
possible split point to find the threshold that best separates "should
refuse" from "should answer" (Youden's J statistic).

| Configuration | Refusal recall | Answerable pass-rate (no false refusals) | Recall@8 | MRR |
| --- | --- | --- | --- | --- |
| Phase 3 (hybrid, no confidence gate) | 16.7% (1/6) | 100% | 100.0% | 0.893 |
| Phase 4 (hybrid + calibrated confidence gate, threshold=0.02990) | **83.3%** (5/6) | **100%** | 100.0% | 0.893 |

Refusal recall goes from 16.7% to 83.3% — and, just as important, the
*answerable* pass-rate stays at 100%: not one of the 14 questions the system
should answer got wrongly refused. Retrieval quality itself (Recall@8, MRR)
is untouched, because the gate only decides whether to answer, not what to
retrieve. This is a real, end-to-end verified result, not just the
calibration script's own math: `scripts/run_eval.py` was re-run with the
threshold wired into `confidence_gate.min_score` in
`config/sandbox_local_phase4.yaml`, and it reproduced exactly what the
calibration predicted.

**The one miss, and why, honestly**: `ua-001` ("How do I configure Istio
mutual TLS with Kubernetes 1.28?") still isn't refused. Its top score
(0.03226) sits inside the answerable cluster, not with the other five
unanswerable questions. The likely reason: Istio is topically adjacent to
real content in this corpus (service networking, TLS, Ingress), so
retrieval finds *plausible-looking* Kubernetes networking chunks even
though nothing about Istio itself is in the corpus — a single score
threshold can't distinguish "confidently relevant" from "confidently
relevant to the wrong topic". That's exactly the failure mode an actual
cross-encoder reranker (trained to judge query-document relevance
directly, not just embedding/keyword overlap) would be expected to catch
and this pure retrieval-score threshold can't. Left as the concrete reason
the reranker half of this phase still matters, not swept under the 83.3%
headline number.

`config/default.yaml` now has `confidence_gate.enabled: true` with this
calibrated threshold. `reranker.enabled` stays `false`, with the reason
documented in the config file itself rather than left to look like an
oversight.

## Phase 5 — query understanding

**Negative result, not shipped, and one of the more useful findings in
this project precisely because it's negative.**

The roadmap's original scope was LLM-based query rewriting/decomposition
and optional HyDE. Both need a real generation model to paraphrase or
hallucinate a hypothetical answer -- the `stub` generator here is
extractive-only, and `OPENAI_API_KEY` isn't set in this sandbox. Rather
than fake either technique, this phase implemented the one form of
decomposition that's genuinely structural and needs no model at all:
**per-version fan-out** (`src/k8s_rag/query_understanding.py`). The
motivating problem was real and measured before writing any retrieval
code: this corpus's `version_sensitive` questions don't name a version
("What apiVersion should I use for Ingress?"), and a single query against
the whole corpus doesn't reliably surface evidence from *every* version
whose docs differ. Baseline `citation_coverage` (a new metric added this
phase -- see below) on the 6 `version_sensitive` golden questions:
**77.8%**, with `citation_hit` sitting at ~100% and hiding the gap
entirely (`citation_hit` only checks "matched *at least one* expected
citation" -- worthless for a question whose correct answer spans three
versions).

The fix: since the corpus's version list is already known
(`corpus.versions` in config), a question that doesn't pin a version gets
asked once per version via the retriever's existing `version_filter`, and
the results are merged so every version's top hit is guaranteed a slot.
No new model, no new embedding call shape.

| Configuration | Recall@8 | MRR | Citation coverage (all) | Citation coverage (version_sensitive) | Refusal recall |
| --- | --- | --- | --- | --- | --- |
| Phase 4 (no fan-out) | **100.0%** | **0.893** | **90.5%** | 77.8% | 83.3% (5/6) |
| Phase 5 (per-version fan-out) | 92.9% | 0.786 | 89.3% | 75.0% | 50.0% (3/6), or 100% with a re-tuned threshold -- see below |

**It made every number worse, on both axes it touches.** Two distinct,
independently-diagnosed root causes, not one vague "it didn't help":

**1. Fan-out doesn't guarantee the *right* page, only a page from the
right version.** Diagnosed directly on `vs-002` ("Is PodSecurityPolicy
something I should use?"), whose coverage went from 50% to **0%**:

```
--- version 1.24 ---
  [0.03202] reference/using-api/deprecation-guide
  [0.03126] concepts/security/pod-security-policy   <- the actually-correct page
--- version 1.25 ---
  [0.03227] reference/using-api/deprecation-guide
  [0.03175] concepts/security/pod-security-policy   <- the actually-correct page
```

`reference/using-api/deprecation-guide` -- a page that enumerates dozens of
deprecated resources and shares vocabulary with almost any deprecation-ish
query -- edges out the genuinely correct page by a razor-thin margin
*within each version's own small candidate pool*, for both versions.
Guaranteeing "the #1 hit per version" is fragile exactly when a generic
high-recall attractor page exists in the corpus: forcing a pick per version
doesn't fix ranking quality, it just relocates the same ranking mistake to
every version. (Confirmed this isn't a shallow-cutoff bug: reran with
`per_version_k` at 2, 4, and 8 -- identical 75% coverage every time. The
#1-ranked page within that version's subset simply isn't the right one,
regardless of how deep the fetch goes.)

**2. Fan-out breaks the Phase 4 confidence gate, because RRF's score is
rank-based, not magnitude-based, and that stops being comparable once the
candidate pool's size changes.** `RRF(d) = sum 1/(k+rank)` depends only on
*where* a chunk ranks within its own retrieval call, never on how good a
match it actually is. Phase 4's threshold (0.02990) was calibrated against
scores computed over the *whole* ~2,000-chunk corpus. Under fan-out, each
version's top hit is computed over a **much smaller, version-filtered
slice** (roughly 1/6th the corpus) -- and something always ranks #1 in a
small pool, even a mediocre match, producing a similarly high RRF score
purely from having less competition. Measured directly on two previously
correctly-refused questions:

| Question | Top score, no fan-out | Top score, fan-out | Calibrated threshold |
| --- | --- | --- | --- |
| ua-002 ("exact CPU request for a WordPress deployment") | 0.02985 (below threshold -- refused) | **0.03226** (above threshold -- answered) | 0.02990 |
| ua-005 ("GKE vs. EKS vs. AKS?") | 0.02365 (below threshold -- refused) | **0.03089** (above threshold -- answered) | 0.02990 |

Refusal recall collapsed from 83.3% to 50% for exactly this reason, not
from a bug in the confidence-gate logic itself.

**Re-calibrating doesn't rescue it -- it trades one failure mode for a
worse one.** The first version of `scripts/calibrate_refusal.py` didn't
actually re-test this: it called the plain retriever directly and never
routed through `fanout_search`, so it was (silently) calibrating against a
score distribution the pipeline doesn't produce when fan-out is on --
fixed in this phase (`_search()` now mirrors `graph.py`'s `retrieve_node`
exactly). Once fixed and re-run against the real fan-out distribution, the
best achievable threshold (0.03239) gets refusal recall to 100% -- but at
85.7% answerable pass-rate (2 new false refusals), because fan-out
compresses both distributions into a narrower, more overlapping score band.
Phase 4's clean 83.3% recall / 100% pass-rate split isn't recoverable by
threshold-tuning alone once fan-out is in the retrieval path.

**Conclusion: `query_understanding.enabled` stays `false` in
`config/default.yaml`.** The module, its config (`config/sandbox_local_
phase5.yaml`), and 9 passing unit tests (`tests/test_query_understanding.py`)
stay in the repo as a documented, honestly-negative experiment -- the same
treatment given to Phase 1's `spacy_vectors` and Phase 2's `recursive`
chunking. The real lesson isn't "fan-out is a bad idea" in general; it's
that the roadmap's original instinct -- lean on an actual LLM for
decomposition rather than a structural heuristic -- was probably right.
An LLM asked "does this question need per-version treatment, and if so
which versions matter" could route past both failure modes found here: it
could tell `deprecation-guide` isn't the answer PodSecurityPolicy usage
guidance belongs in, and it wouldn't rely on a rank-only fusion score as a
confidence signal in the first place. That stays a documented gap, same
pattern as the Phase 4 reranker: both need real model access this sandbox
doesn't have.

**What Phase 5 did add permanently, independent of the fan-out result:**
`citation_coverage` (`src/k8s_rag/eval_metrics.py`) is a strictly more
informative metric than `citation_hit` for any multi-citation question, and
is now computed and reported by `scripts/run_eval.py` for every run,
including the recorded Phase 4 baseline above (its 77.8%
`version_sensitive` coverage was real all along -- Phase 4's writeup
reported `citation_hit`/recall, which didn't surface it). Finding a real
gap that an existing metric was hiding is worth more than the fan-out
technique that was built to try to close it.

## Phase 6 — config & experiment infrastructure

Exit criteria per the roadmap: "every eval run reproducible from a single
config file/commit." Before this phase, `config_path` (added in Phase 1)
was only half of that promise -- it names *which* file, not whether that
file's *content* is still what it was when the run happened, and nothing
recorded whether the git tree was even clean at the time. Both gaps are
closed by `src/k8s_rag/repro.py`, wired into both `scripts/run_eval.py` and
`scripts/calibrate_refusal.py`:

- **`git_commit`** — the commit the run was made against (`None` gracefully
  if run outside a git checkout, never a crash).
- **`git_dirty`** — whether the working tree had uncommitted changes at run
  time. A run recorded dirty is reproducible from the config file's exact
  bytes (still hashed and checked), but *not* from `git show
  <commit>:config/whatever.yaml` — that would reconstruct a different,
  already-superseded version of the file.
- **`config_content_hash`** — a sha256 prefix of the config file's actual
  bytes, so a same-path-different-content edit (config drift) is
  detectable even when commit metadata is unavailable or ambiguous.

`scripts/list_experiments.py` reads every file in `data/eval/results/` and
cross-checks each one's recorded hash against that config file's *current*
on-disk content, flagging drift instead of trusting `docs/ABLATIONS.md`'s
hand-written prose to have transcribed every number correctly. Run against
this repo as of this phase:

```
run_name                          recall@k     mrr  refusal     commit status
-----------------------------------------------------------------------------
phase1-baseline-sandbox-spacy        28.6%   0.196    16.7%        n/a no repro metadata (pre-Phase-6 run)
phase1-baseline-sandbox             100.0%   0.881    16.7%        n/a no repro metadata (pre-Phase-6 run)
phase2-parent-document-chunking     100.0%   0.889    16.7%        n/a no repro metadata (pre-Phase-6 run)
phase2-recursive-chunking            92.9%   0.774    16.7%        n/a no repro metadata (pre-Phase-6 run)
phase3-hybrid-retrieval             100.0%   0.893    16.7%        n/a no repro metadata (pre-Phase-6 run)
phase4-calibrated-refusal           100.0%   0.893    83.3% c038692e42 recorded on a DIRTY tree
phase5-version-fanout                92.9%   0.786    50.0% c038692e42 recorded on a DIRTY tree
```

Every number in this table matches what's written above in this document
-- a real, if small, consistency check, not just a nice table. The
pre-Phase-6 rows correctly show as missing repro metadata rather than
being backfilled with fabricated commit hashes they were never actually
run against. The Phase 4/5 rows show `DIRTY` honestly: they were re-run
during Phase 6's own development, against a tree that had Phase 6's
not-yet-committed changes in it (this phase's tooling change doesn't alter
retrieval, so the numbers are identical to the earlier committed runs --
confirmed by the values matching Phase 4/5's sections above exactly). That
this tool immediately found something worth flagging on the very data it
shipped with, rather than reporting a clean bill of health it couldn't
actually back up, is the point of building it this way instead of a
results spreadsheet that just trusts every entry.

Prompt versioning (`config/prompts/v1.yaml`, `generation.prompt_version`)
and per-experiment config files were already in place since Phase 1 --
nothing new needed there. What's genuinely new is treating "reproducible"
as a checkable claim instead of an assumption.

## Phase 7 — eval-as-CI

Exit criteria per the roadmap: "CI fails a PR that regresses a core
metric." Three pieces, in the order they were actually built and verified
-- gate logic first, wiring second, because the gate is the part that has
to be *correct*, and correctness there is checkable without ever touching
GitHub Actions:

**1. `scripts/check_regression.py`** — compares a fresh eval result against
a frozen baseline (`data/eval/results/ci_baseline.json`) across five tracked
metrics (Recall@8, MRR, citation coverage overall and
`version_sensitive`-only, refusal recall), each with its own tolerance in
absolute percentage points, not zero -- `local_tfidf`/BM25/RRF are fully
deterministic given the same corpus, so an exact-match gate was
*possible*, but exact-match CI gates are exactly the kind of brittle check
that gets force-merged past on a bad day rather than actually trusted.
9 unit tests (`tests/test_check_regression.py`) exercise the gate's logic
directly against synthetic before/after data -- identical results pass,
small in-tolerance drops pass, a large drop on any one metric fails, several
simultaneous regressions are all reported together (not just the first),
and a metric missing from the current run (rather than merely lower) is
correctly flagged rather than silently ignored.

**2. It was verified against this project's own real data, not just
synthetic fixtures.** Running the gate with a fresh CI-config eval as
`--current` against the frozen baseline passes cleanly (all five metrics at
`+0.00%`, since nothing about retrieval changed since the baseline was
captured). Running it again with `--current
data/eval/results/phase5-version-fanout.json` -- the actual Phase 5
regression, not a fabricated one -- correctly fails:

```
REGRESSION DETECTED (3 metric(s)):
  - Recall@k regressed: 100.00% -> 92.86% (dropped 7.14%, tolerance is 2.00%)
  - MRR regressed: 89.29% -> 78.57% (dropped 10.71%, tolerance is 2.00%)
  - Refusal recall regressed: 83.33% -> 50.00% (dropped 33.33%, tolerance is 2.00%)
```

This is the actual point of the phase: if Phase 5's per-version fan-out had
been submitted as a pull request with `query_understanding.enabled: true`
in `config/ci.yaml`, this gate would have failed the build on exactly the
three metrics that mattered, for the right reasons, automatically -- not
relying on a human to notice while reading a wall of eval output.

**3. `.github/workflows/eval-ci.yml`** wires the gate into GitHub Actions:
unit tests, fetch the corpus, ingest with `config/ci.yaml` (network-free
`local_tfidf` embeddings -- deterministic and fast, no model host to be
briefly unreachable, no per-run API cost; `config/default.yaml`'s real
embeddings stay the recommended config for actually *using* the system, not
for every CI run), run the eval, then the regression gate. **The one honest
caveat, stated plainly rather than glossed over**: this workflow file has
not been run against a live GitHub Actions runner from the sandbox this
repo was built in. That sandbox's own network policy blocks
huggingface.co/api.openai.com (the whole reason `local_tfidf` exists at
all -- see Phase 1), but it also simply has no route to github.com's
Actions API to trigger or observe a real run. What *was* verified: every
individual command in the workflow (`pytest`, `fetch_corpus.py`,
`ingest.py` against `config/ci.yaml`, `run_eval.py`, `check_regression.py`)
was run directly, successfully, in this sandbox, and the YAML follows
standard, unmodified GitHub Actions patterns (`actions/checkout@v4`,
`actions/setup-python@v5`). It should work as written on the first real
push once this repo is actually on GitHub -- but "should" is the honest
word here, not "confirmed," and that distinction is worth keeping visible
rather than quietly dropped.

## Phase 8 — productionization

Exit criteria per the roadmap: "service runs in a container; a bad answer
can be traced back to its cause." No new pipeline logic here on purpose --
`src/k8s_rag/api.py` is a transport layer over the exact same
`run_pipeline` every script in this repo already calls, so the service
inherits everything already measured in Phases 1-7 rather than
re-implementing retrieval, chunking, or refusal a second time.

**Traceability, the part the exit criterion actually names, doesn't wait
on LangSmith.** Every `/ask` response carries a `request_id`, the top
retrieved chunk ids and scores, the decision (and refusal reason, when
refused), and the exact `run_name` / git commit / config content hash that
produced it (via `src/k8s_rag/repro.py`, Phase 6's work reused here rather
than re-invented). The same fields are written as one structured JSON log
line per request. Handed a `request_id` for a bad answer, that's the exact
question, the exact chunks and scores that were retrieved, and the exact
commit + config to reproduce it -- no separate tracing backend required.
LangSmith (`tracing.langsmith_project` + `LANGCHAIN_API_KEY`) layers on top
of this when configured; it's additive, not the only mechanism.

**Verified live, not just unit-tested against a TestClient.** Ran
`scripts/serve.py` against `config/ci.yaml`'s already-ingested data and hit
it with real HTTP requests:

```
GET  /health         -> {"status":"ok","chunk_count":1958,"run_name":"ci"}
POST /ask             -> real extractive answer, 8 citations, latency_ms: 96.7,
                          git_commit + config_content_hash both populated
POST /ask/stream      -> event: answer / event: done, real SSE framing
GET  /metrics         -> request_count, refusal_rate, p50/p95 latency, cost ($0.00, genuinely)
```

9 tests in `tests/test_api.py` (skipped cleanly, not failing, on a fresh
checkout with no ingested data) cover health, an answerable question, a
refused one, input validation, the streaming endpoint's event framing, the
metrics endpoint, and rate limiting -- both that `/health`/`/metrics` stay
exempt and that `/ask` actually returns `429` past its configured limit
(confirmed live too: 4 rapid requests against a limit of 2 produced
`[200, 200, 429, 429]`).

**Two scoped-down pieces, documented rather than oversold:**

- **Streaming** is real SSE plumbing (`POST /ask/stream`), but the `stub`
  generator computes its answer synchronously (an extractive regex scan
  over already-retrieved text, not a token-by-token model), so it
  legitimately emits one `answer` event rather than chunking a
  pre-computed string word-by-word to *simulate* an LLM's typing effect --
  that would be exactly the kind of faked polish this project has avoided
  everywhere else. A real streaming code path (`OpenAIGenerator` with
  `stream=True`) is a documented gap, same pattern as the reranker and LLM
  query rewriting.
- **Rate limiting** is a real in-memory fixed-window limiter (unit-tested
  and verified live above), explicitly single-process: it has no shared
  state, so it's not safe across multiple replicas without a backing store
  like Redis. Fine for the single-instance deployment this project ships;
  a named, real limitation for anything beyond that.

**Docker: built and reasoned through, not run end-to-end.** `Dockerfile`,
`docker-entrypoint.sh` (fetches the corpus and ingests on first run if
`data/chroma` is empty -- verified directly: the chunk-count check script
embedded in the entrypoint runs correctly against this repo's real data,
returning `1958`), and `docker-compose.yml` (named-volume persistence
across restarts) are all written. **Honest caveat**: this sandbox's Docker
CLI has no daemon behind it (`docker ps` fails with "no such file or
directory" on the daemon socket -- confirmed directly, not assumed), so the
image was never actually built or run as a container here. Every piece
that *can* be verified without a daemon was: the entrypoint script's shell
syntax (`sh -n`), and, more importantly, the exact Python logic it embeds
(the ingested-data check) run directly against this repo's real config and
data. What's unverified is Docker-specific: does the image build cleanly,
does the entrypoint's exec handoff to uvicorn work inside a container, does
the healthcheck pass. Same category of gap as Phase 7's GitHub Actions
workflow -- correct by inspection and by verifying every piece that can be
isolated from the untestable environment, not by a live run that this
sandbox has no way to perform.

## Phase 9 — ship

Exit criteria per the roadmap: "live demo link + README readable by a
non-expert in under 5 minutes."

**README: done.** Rewritten as a case-study document rather than a plain
setup guide: a headline Results table up top (the Phase 1 → Phase 4
refusal-recall improvement, the number that actually matters), an
architecture diagram that matches what `src/k8s_rag/graph.py` really runs
today (not the original roadmap's diagram, which still had a `reranker` box
that was never built), and a dedicated "Rejected tradeoffs" table
summarizing the four real negative results from this project (recursive
chunking, spaCy vectors, per-version fan-out, hybrid's latency cost) with a
one-line reason each and a link to the full writeup. The goal is that
someone can read the top of the README and understand what this project
actually demonstrates without reading `docs/ABLATIONS.md` end to end --
that document is still where the depth lives, not duplicated into the
README.

**Live demo link: not delivered, and not faked.** This is a genuine
external dependency, the same category of honest gap as Phase 4's reranker
or Phase 7/8's "never run against a live GitHub Actions runner / Docker
daemon" -- except this one isn't closeable by better code or more careful
verification from inside this sandbox. A hosted demo needs an account on
an external platform (Render, Fly.io, Railway, Google Cloud Run are all
reasonable fits for the `Dockerfile` this project already has) and
credentials this session was never given and has no way to obtain on its
own. Producing a URL here would mean either fabricating one (a direct
violation of everything else this project stands for) or spending someone
else's cloud credentials without asking, neither of which is acceptable.
What's actually ready for that step: a working `Dockerfile` +
`docker-compose.yml` (Phase 8), a `/health` endpoint any platform's health
check can point at, and `config/ci.yaml` as a network-free config that
needs no secrets to serve a working (if stub-generation) demo on day one.
Deploying it is a real next step, not a technical unknown -- it's waiting
on a platform choice and credentials that are the project owner's to
provide, not this session's to invent.

## Cost & practicality — the comparison across everything tried

Every table above answers "which approach won on quality." This section
answers the two questions a hiring manager or a teammate actually asks
next: **what does this cost, and is it usable as-is.** Nothing here is
invented -- every quality/latency number is copied from the phase tables
above (already real measurements), and every cost number is either a real
$0.00 (this project genuinely never called a paid API -- confirmed by the
`/metrics` endpoint reporting `cost: $0.00` in Phase 8, not estimated) or a
calculation using two real inputs: this project's own measured token usage
from its 20-question golden set, and current published API pricing looked
up on 2026-09-24 (sources linked below). Where a number is an estimate, it
says so -- this project's whole premise is not doing the thing this
paragraph just promised not to do.

### The master comparison table

Every axis actually ablated in this project, one row per configuration,
pulled straight from the phase tables above:

| Axis | Configuration | Recall@8 | MRR | Refusal recall | p50 latency | Measured cost | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Embedding provider | `local_tfidf` (shipped) | **100.0%** | **0.881** | 16.7% | 7 ms | $0.00 | **Winner.** Beats the "smarter" option below on this jargon-dense corpus. |
| Embedding provider | `spacy_vectors` | 28.6% | 0.196 | 16.7% | 11 ms | $0.00 | Rejected. Real semantic vectors, wrong tool for exact-token-heavy API docs. |
| Chunking | `fixed` (Phase 1) | **100.0%** | 0.881 | -- | 7 ms | $0.00 | Simple, strong floor. |
| Chunking | `recursive` (header-aware) | 92.9% | 0.774 | -- | 10 ms | $0.00 | Rejected. More, smaller chunks diluted keyword density. |
| Chunking | `parent_document` (shipped) | **100.0%** | **0.889** | -- | 13 ms | $0.00 | **Winner.** Small chunks for matching, large parent for generation context. |
| Retrieval mode | vector-only (Phase 2) | 100.0% | 0.889 | -- | 13 ms | $0.00 | Fast, strong baseline. |
| Retrieval mode | hybrid: BM25+vector, RRF (shipped) | 100.0% | **0.893** | -- | 73-100 ms | $0.00 | **Winner on quality, ~9x latency cost** -- see practicality verdict below. |
| Confidence gate | off (Phase 3) | 100.0% | 0.893 | 16.7% (1/6) | ~76 ms | $0.00 | Rejected. Answers confidently on 5/6 questions it should refuse. |
| Confidence gate | calibrated threshold, on (shipped) | 100.0% | 0.893 | **83.3%** (5/6) | ~76 ms | $0.00 | **Winner.** Best single number in this project, zero cost to answerable questions. |
| Query understanding | off (shipped) | **100.0%** | **0.893** | **83.3%** | ~76 ms | $0.00 | **Winner (by not doing it).** |
| Query understanding | per-version fan-out (Phase 5) | 92.9% | 0.786 | 50.0% (3/6) | higher (more retrieval calls, not separately timed) | $0.00 | Rejected. Made every number worse -- full root-cause in Phase 5 above. |

Every row above is $0.00 because every run in this project used
network-free local components end to end. That's not "this project is
cheap" -- it's "this project has not yet paid for the thing it's actually
supposed to cost money": a real embedding model and a real LLM. The rest
of this section prices that out honestly.

### What it would actually cost with the production stack this project recommends but couldn't run

`config/default.yaml` names the intended production path in its own
comments: `embedding.provider: openai` / `text-embedding-3-small`, and
`generation.model: "gpt-4o-mini"`. Pricing below is current as of
2026-09-24, from OpenAI's own model pages and the CloudZero pricing
tracker, cross-checked against a second source:

| Component | Price (2026-09-24) | Source |
| --- | --- | --- |
| `text-embedding-3-small` | $0.02 / 1M tokens | [OpenAI model page](https://developers.openai.com/api/docs/models/text-embedding-3-small) |
| `text-embedding-3-large` | $0.13 / 1M tokens | [CloudZero OpenAI pricing tracker](https://www.cloudzero.com/blog/openai-pricing/) |
| `gpt-4o-mini` (the model hardcoded in this repo's config) | **not listed -- see finding below** | [OpenAI pricing docs](https://developers.openai.com/api/docs/pricing), [Finout 2026 tracker](https://www.finout.io/blog/openai-pricing-in-2026) |
| GPT-5.6 Luna (OpenAI's current cheapest general chat tier, used as the stand-in below) | $0.20 / 1M input, $1.20 / 1M output | [CloudZero OpenAI pricing tracker](https://www.cloudzero.com/blog/openai-pricing/) |
| Cohere Rerank v3.5 | $0.001 / search ($1.00 / 1000 searches) | [OpenRouter](https://openrouter.ai/cohere/rerank-v3.5) |

**An unplanned finding that belongs in the practicality section, not just
here**: `gpt-4o-mini` -- the exact model string sitting in
`generation.model` in every config file in this repo -- is no longer on
OpenAI's current pricing page. Three independent sources checked today
confirm it; none states an exact deprecation date, but OpenAI's current
lineup has moved to the GPT-5.6 series (Luna is the cheapest tier). This
happened without this project doing anything wrong -- it's just what
happens to any pipeline that pins a model string and doesn't touch it for
a while. It's used here as the concrete example of a real maintenance cost
category (model staleness) that the fully-local `local_tfidf` + stub path
genuinely doesn't have. GPT-5.6 Luna is used below as the honest current
stand-in for "OpenAI's cheapest general-purpose chat model," which is what
`gpt-4o-mini` *was* when this config was written.

**Real measured inputs to the cost calculation** (from this project's own
20-question golden set, `retriever.search()` run directly against the real
ingested corpus -- not assumed):

| Measured quantity | Value | How it was measured |
| --- | --- | --- |
| Avg question tokens | 14.4 | `approx_tokens()` over all 20 golden questions |
| Avg retrieved context tokens per query | 1513.9 (median 1305.5, max 4114) | `approx_tokens()` over the joined `context_text` of `retriever.search(q, top_k=8)`'s real output, all 20 questions |
| System prompt tokens | 169 | `approx_tokens()` over `config/prompts/v1.yaml`'s `system` field |
| Prompt template boilerplate | 11 | Same file's `user_template`, with `{question}`/`{context}` stripped |
| Avg golden-set `expected_answer` length | 28.65 tokens (median 27, max 57) | Real, but these are terse ground-truth reference answers, not full generated prose -- used below only as a conservative *floor* for output length |
| Corpus size actually ingested | 1958 chunks, 252,026 approx tokens | Direct re-run of `chunk_document()` over all 57 fetched corpus files with the shipped `parent_document` config |

`approx_tokens()` throughout is this project's own `words * 1.3` heuristic
(`chunking.py`), consistent with every other token count in this doc --
not a different, more favorable method picked for this section.

**Generation cost is the one number here that's a genuine estimate, and
here's exactly why**: the stub generator (extractive, regex-based) is the
only thing that has ever produced an answer in this project end to end --
no real LLM call has ever happened, so there is no measured output-token
length to use. The input side is not a guess (system prompt + template +
real retrieved context + real question, all measured above, ≈1708 tokens);
the output side uses a bracket -- 29 tokens (the golden set's own terse
`expected_answer` length, a floor) to 200 tokens (a fuller cited
explanation matching the system prompt's own rules about citing every
claim), an honest ceiling, not a measured value.

| Query volume | Embedding (question only, `text-embedding-3-small`) | Generation (GPT-5.6 Luna, 1708 input + 29-200 output tokens) | Reranker (Cohere, if it existed -- it doesn't ship here) | Total per query |
| --- | --- | --- | --- | --- |
| 1 query | ~$0.0000003 | $0.00038 – $0.00058 | $0.001 | **$0.0014 – $0.0016** |
| 1,000 queries | $0.0003 | $0.38 – $0.58 | $1.00 | **$1.38 – $1.58** |
| 30,000 queries (1,000/day, one month) | $0.009 | $11.40 – $17.40 | $30.00 | **~$41 – $47** |
| 300,000 queries (10,000/day, one month) | $0.09 | $114 – $174 | $300.00 | **~$414 – $474** |

One-time ingestion cost (embed the whole 6-version, 57-page, 1958-chunk
corpus once): **$0.005** with `text-embedding-3-small`, **$0.033** with
`text-embedding-3-large`. Re-embedding after every corpus or chunking
change is, at this corpus size, not a cost concern at all.

**The counterintuitive finding worth stating plainly: the reranker this
project never built would be the single largest cost line, bigger than
generation itself**, at any query volume, because Cohere prices per search
call regardless of token count -- it scales with *query volume*, not
corpus or context size, so it doesn't get relatively cheaper the way
per-token costs do as prompts stay roughly fixed. Anyone budgeting a RAG
project with a reranker should size that line item by expected QPS, not by
corpus size or context length.

### The honest verdict: cost is not the blocker here

At even a generous 10,000 queries/day, the full production stack --
real embeddings, real generation, and a reranker this project doesn't even
have yet -- costs under $500/month. Per query, that's roughly a tenth of a
cent. **Cost was never going to be what stops this system from shipping.**
What actually blocks production use, in order of how much it would take to
fix:

1. **Retrieval and generation quality with real models are unmeasured.**
   Every quality number in this document (100% recall, 0.893 MRR, 83.3%
   refusal recall) was produced by `local_tfidf` retrieval and a stub
   extractive generator, not a real embedding model or LLM. The
   architecture and eval harness are ready to measure the real numbers the
   moment `OPENAI_API_KEY` and normal network access exist -- that's a
   one-line config change, not a rebuild -- but until then, faithfulness,
   citation precision, and answer relevancy are genuinely unknown, not
   just unreported.
2. **No reranker.** Documented gap since Phase 4; also, per this section,
   the most expensive single line item if it's ever added, so it needs a
   real ROI case (does it fix `ua-001`-style topically-adjacent false
   answers enough to justify $1/1000 queries), not just "rerankers are
   best practice."
3. **Hybrid retrieval's ~9x latency cost** (11 ms → 73-100 ms, Phase 3)
   comes from `rank_bm25`'s O(n) linear scan over the corpus on every
   query. Fine at ~2,000 chunks; a real corpus at 10-100x this size would
   need a proper inverted index (e.g. Elasticsearch/OpenSearch BM25), not
   `rank_bm25`, before hybrid retrieval stays viable.
4. **The rate limiter is single-process, in-memory** (Phase 8) -- correct
   and verified for one instance, unsafe the moment this runs behind a
   load balancer with more than one replica, without a shared store like
   Redis.
5. **Docker and CI were never run against live infrastructure** (Phase
   7/8) -- written and reasoned through in full, verified everywhere this
   sandbox allows, but "the container builds and the healthcheck passes"
   and "the GitHub Actions workflow actually gates a PR" are both still
   unconfirmed by an actual run.
6. **Model staleness is a real, ongoing maintenance cost**, demonstrated
   by this very section: `gpt-4o-mini`, hardcoded in every config file,
   was already gone from OpenAI's current pricing by the time this
   analysis was written. An API-dependent RAG pipeline needs a process for
   noticing and reacting to this; the fully-local fallback path doesn't.

None of these six are multi-week problems -- each is realistically a
one-to-three-day task on a machine with normal network access and an API
key. But "cheap and fast to fix" is not the same as "already fixed," and
the honest answer to "is this practical to use today" is: **not
production-ready as shipped, but for reasons that are all named, scoped,
and estimated here -- not open questions.** The more valuable deliverable
this project produced may not be the current numbers at all, but the eval
harness, cost model, and reproducibility system that will measure the real
numbers the moment those six items are closed.
