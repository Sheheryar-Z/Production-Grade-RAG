"""Embedding provider. Four implementations, same interface:

- `sentence_transformers` (recommended default): a local, free, open-weight
  transformer model with contextual embeddings. This is what a real
  deployment (or this repo, cloned onto a machine with normal internet
  access) should use.
- `openai`: an API-backed embedding model, for production retrieval quality.
- `spacy_vectors`: a local, static word-vector model (GloVe-style, 300-dim,
  averaged per chunk). Real semantic similarity -- it correctly matches
  paraphrases and synonyms ("PSP" <-> "PodSecurityPolicy") that pure keyword
  overlap misses -- but weaker than a contextual transformer: no attention,
  no sense of word order, one fixed vector per word regardless of context.
  Its model file downloads from a GitHub release asset, not huggingface.co.
- `local_tfidf`: pure scikit-learn bag-of-words, zero network dependency at
  all, not even GitHub. Purely lexical: matches shared words, not meaning.

Why three "local" tiers instead of just using the best one: this project was
built in a sandboxed dev environment whose egress policy blocks
huggingface.co (confirmed still blocked even after the user changed their
own network -- this sandbox's policy is controlled at the organization
level, in Admin settings -> Capabilities, not by anything on the user's end;
see docs/ABLATIONS.md for the full story). GitHub release assets, however,
*are* reachable from here, which is what makes `spacy_vectors` possible as a
real, meaningfully-better-than-TF-IDF fallback in this specific sandbox.
`local_tfidf` stays in the codebase as the last-resort, zero-dependency
option for a fully offline environment. Switching to `sentence_transformers`
or `openai` anywhere with real network/API access is a one-line config
change (`embedding.provider`); nothing else in the pipeline changes.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from k8s_rag.config import RunConfig


class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


class SpacyVectorEmbeddings(EmbeddingProvider):
    """Averaged static word vectors from a spaCy medium/large English model.
    No fitting step needed (unlike TF-IDF) -- the vectors are pretrained and
    fixed, so a query at retrieval time embeds independently of what was
    seen at ingestion time, same as a real embedding model would.
    """
    def __init__(self, model_name: str):
        import spacy
        try:
            self._nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' isn't installed. Install it from a GitHub release "
                f"asset (works even where huggingface.co is blocked), e.g.:\n"
                f"  pip install https://github.com/explosion/spacy-models/releases/download/"
                f"{model_name}-3.7.1/{model_name}-3.7.1-py3-none-any.whl"
            ) from exc
        if not self._nlp.meta.get("vectors", {}).get("width"):
            raise RuntimeError(
                f"'{model_name}' has no word vectors (e.g. en_core_web_sm ships without them). "
                "Use en_core_web_md or en_core_web_lg instead."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        docs = self._nlp.pipe(texts)
        return [doc.vector.tolist() for doc in docs]


class TfidfEmbeddings(EmbeddingProvider):
    """A fit-once, network-free stand-in. `fit_and_persist` is called once
    during ingestion over the whole corpus; queries at retrieval time reuse
    that fitted vocabulary via `load`. Vectors are L2-normalized so cosine
    similarity (what Chroma uses here) behaves the same way it would for a
    real embedding model.
    """
    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit_and_persist(self, texts: list[str], path: Path) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._vectorizer, f)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(
                f"No fitted TF-IDF vectorizer at {path}. Run scripts/ingest.py first -- "
                "it fits the vectorizer on the corpus and persists it here."
            )
        with open(path, "rb") as f:
            self._vectorizer = pickle.load(f)
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings used before fit_and_persist()/load() was called.")
        return self._vectorizer.transform(texts).toarray().tolist()


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "embedding.provider is 'openai' but OPENAI_API_KEY is not set. "
                "Set it, or switch config to provider: sentence_transformers to run without a key."
            )
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]


def tfidf_persist_path(cfg: RunConfig) -> Path:
    from k8s_rag.config import REPO_ROOT
    persist_dir = REPO_ROOT / cfg.get("vector_store", "persist_dir", default="data/chroma")
    return persist_dir / "tfidf_vectorizer.pkl"


@lru_cache(maxsize=4)
def _cached_provider(provider: str, model: str, tfidf_path: str | None) -> EmbeddingProvider:
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddings(model)
    if provider == "openai":
        return OpenAIEmbeddings(model)
    if provider == "spacy_vectors":
        return SpacyVectorEmbeddings(model)
    if provider == "local_tfidf":
        emb = TfidfEmbeddings()
        path = Path(tfidf_path)
        if path.exists():
            emb.load(path)
        return emb  # unfitted is fine here; ingest.py fits it explicitly
    raise ValueError(f"Unknown embedding provider: {provider}")


def get_embedding_provider(cfg: RunConfig) -> EmbeddingProvider:
    provider = cfg.get("embedding", "provider", default="sentence_transformers")
    model = cfg.get("embedding", "model", default="all-MiniLM-L6-v2")
    tfidf_path = str(tfidf_persist_path(cfg)) if provider == "local_tfidf" else None
    return _cached_provider(provider, model, tfidf_path)
