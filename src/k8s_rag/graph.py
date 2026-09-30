"""The pipeline as an explicit LangGraph state graph, per the architecture
diagram in the project roadmap doc:

    retrieve -> decide (enough evidence?) --yes--> generate (with citations)
                                           --no---> refuse (explain why)

This is a genuine fit for LangGraph, not framework-chasing: the branch
between "generate" and "refuse" is real state-dependent control flow, and as
Phase 3-5 land (hybrid fusion, reranking, query rewriting) each becomes its
own node ahead of `decide` without changing this file's shape. The retrieval
and refusal LOGIC itself lives in vectorstore.py / llm.py / this module's
node functions in plain, readable Python -- LangGraph only owns the wiring
between them, on purpose, so the pipeline stays explainable without the
framework in the room.

LangSmith tracing is opt-in and additive: if LANGCHAIN_API_KEY is set (and
tracing.langsmith_project in config), every run is traced automatically by
LangGraph/LangChain's callback system -- no code here changes. Without a
key, the graph runs exactly the same, just untraced.
"""
from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from k8s_rag.bm25_index import BM25Index, bm25_persist_path
from k8s_rag.config import RunConfig
from k8s_rag.llm import get_generator
from k8s_rag.query_understanding import fanout_search, needs_version_fanout
from k8s_rag.retrieval import HybridRetriever
from k8s_rag.vectorstore import RetrievedChunk, VectorStore


def build_retriever(cfg: RunConfig, store: VectorStore):
    """Vector-only (`VectorStore`) or hybrid (`HybridRetriever`) -- both
    expose the same `.search(query, top_k, version_filter)` signature, so
    every caller downstream (this graph, scripts/ask.py, scripts/run_eval.py)
    is written against that shared interface and never has an if/else on
    retrieval mode of its own."""
    mode = cfg.get("retrieval", "mode", default="vector")
    if mode == "hybrid":
        bm25 = BM25Index()
        bm25.load(bm25_persist_path(cfg))
        return HybridRetriever(store, bm25, cfg)
    if mode == "vector":
        return store
    raise ValueError(f"Unknown retrieval mode: {mode}")


class PipelineState(TypedDict, total=False):
    question: str
    target_version: str | None
    retrieved_chunks: list[RetrievedChunk]
    decision: str  # "generate" | "refuse"
    refusal_reason: str
    answer: str


def _maybe_enable_langsmith(cfg: RunConfig) -> None:
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGCHAIN_PROJECT", cfg.get("tracing", "langsmith_project", default="k8s-rag-portfolio")
        )


def build_graph(cfg: RunConfig, store: VectorStore):
    _maybe_enable_langsmith(cfg)
    retriever = build_retriever(cfg, store)
    generator = get_generator(cfg)
    prompt = cfg.load_prompt()
    top_k = cfg.get("retrieval", "top_k", default=8)
    refuse_below = cfg.get("generation", "refuse_below_context_chunks", default=1)
    # `confidence_gate` is deliberately a separate config section from
    # `reranker` -- see docs/ABLATIONS.md Phase 4. There is no ML reranker
    # in this repo yet (huggingface.co-hosted, blocked here, confirmed via
    # scripts/calibrate_refusal.py's investigation). This gate uses the
    # retrieval score the pipeline already produces (RRF fused rank score
    # in hybrid mode), calibrated empirically against the golden set by
    # that same script -- a real, measured threshold, not a guess.
    gate_enabled = cfg.get("confidence_gate", "enabled", default=False)
    gate_threshold = cfg.get("confidence_gate", "min_score", default=0.0)

    def retrieve_node(state: PipelineState) -> PipelineState:
        question = state["question"]
        target_version = state.get("target_version")
        # Phase 5: fan out across every corpus version when the question
        # doesn't name one -- see query_understanding.py's module docstring
        # for why this is the honest, network-free version of "query
        # decomposition" rather than LLM-based rewriting/HyDE (still
        # unbuilt, same as the Phase 4 reranker gap).
        if needs_version_fanout(question, target_version, cfg):
            chunks = fanout_search(retriever, question, cfg, top_k)
        else:
            chunks = retriever.search(question, top_k=top_k, version_filter=target_version)
        return {"retrieved_chunks": chunks}

    def decide_node(state: PipelineState) -> PipelineState:
        chunks = state.get("retrieved_chunks", [])
        if len(chunks) < refuse_below:
            return {"decision": "refuse", "refusal_reason": "no relevant chunks were retrieved"}
        if gate_enabled and chunks and chunks[0].score < gate_threshold:
            return {
                "decision": "refuse",
                "refusal_reason": (
                    f"top retrieval score {chunks[0].score:.4f} is below the calibrated "
                    f"confidence threshold {gate_threshold:.4f}"
                ),
            }
        return {"decision": "generate"}

    def generate_node(state: PipelineState) -> PipelineState:
        answer = generator.generate(state["question"], state["retrieved_chunks"], prompt)
        return {"answer": answer}

    def refuse_node(state: PipelineState) -> PipelineState:
        return {"answer": prompt["refusal_template"].format(reason=state.get("refusal_reason", "insufficient evidence"))}

    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide", lambda s: s["decision"], {"generate": "generate", "refuse": "refuse"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def run_pipeline(cfg: RunConfig, store: VectorStore, question: str, target_version: str | None = None) -> PipelineState:
    app = build_graph(cfg, store)
    return app.invoke({"question": question, "target_version": target_version})
