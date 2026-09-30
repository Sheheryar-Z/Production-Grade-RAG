"""Generation provider. `stub` needs no API key and no network access, so the
retrieval/citation/refusal pipeline can be smoke-tested end-to-end anywhere,
including this sandbox. It is intentionally NOT a language model -- it's an
extractive stand-in (it surfaces the most relevant retrieved sentences with
their citations rather than synthesizing prose) so nobody mistakes it for a
real answer quality baseline. Real answer-quality and faithfulness numbers
require `generation.provider: openai` (or another real LLM) and are reported
as such in docs/ABLATIONS.md -- never backfilled from the stub.
"""
from __future__ import annotations

import re

from k8s_rag.config import RunConfig
from k8s_rag.vectorstore import RetrievedChunk


class Generator:
    def generate(self, question: str, context_chunks: list[RetrievedChunk], prompt: dict) -> str:
        raise NotImplementedError


class StubGenerator(Generator):
    """Extractive, not generative: pulls the sentence(s) most overlapping the
    question's keywords from each retrieved chunk and stitches them together
    with citations. Deterministic, free, and honest about what it is."""

    def generate(self, question: str, context_chunks: list[RetrievedChunk], prompt: dict) -> str:
        if not context_chunks:
            return prompt["refusal_template"].format(reason="no relevant chunks were retrieved")

        q_words = {w.lower() for w in re.findall(r"[a-zA-Z0-9/._-]+", question) if len(w) > 2}
        lines = []
        for chunk in context_chunks:
            # search the full context_text (== text, unless parent_document
            # chunking widened it to the parent section) so the "best
            # matching sentence" isn't artificially limited to a tiny child chunk
            sentences = re.split(r"(?<=[.!?])\s+", chunk.context_text)
            scored = sorted(
                sentences,
                key=lambda s: len(q_words & {w.lower() for w in re.findall(r"[a-zA-Z0-9/._-]+", s)}),
                reverse=True,
            )
            best = scored[0].strip() if scored else chunk.context_text[:200]
            if best:
                cite = f"[v{chunk.version}: {chunk.source_page}]"
                lines.append(f"{best} {cite}")
        return (
            "[stub extractive answer -- not a real generation model; "
            "for answer-quality/faithfulness numbers, run with generation.provider: openai]\n\n"
            + "\n".join(lines)
        )


class OpenAIGenerator(Generator):
    def __init__(self, model: str, temperature: float):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "generation.provider is 'openai' but OPENAI_API_KEY is not set. "
                "Set it, or switch config to provider: stub to smoke-test the pipeline without a key."
            )
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model
        self._temperature = temperature

    def generate(self, question: str, context_chunks: list[RetrievedChunk], prompt: dict) -> str:
        if not context_chunks:
            return prompt["refusal_template"].format(reason="no relevant chunks were retrieved")
        context = "\n\n".join(
            f"[v{c.version}: {c.source_page}]\n{c.context_text}" for c in context_chunks
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_template"].format(question=question, context=context)},
        ]
        resp = self._client.chat.completions.create(
            model=self._model, temperature=self._temperature, messages=messages,
        )
        return resp.choices[0].message.content


def get_generator(cfg: RunConfig) -> Generator:
    provider = cfg.get("generation", "provider", default="stub")
    if provider == "stub":
        return StubGenerator()
    if provider == "openai":
        return OpenAIGenerator(
            cfg.get("generation", "model", default="gpt-4o-mini"),
            cfg.get("generation", "temperature", default=0.0),
        )
    raise ValueError(f"Unknown generation provider: {provider}")
