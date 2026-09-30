# Phase 8: containerized service. Single-stage on purpose -- this project's
# dependency set (chromadb, sentence-transformers, langgraph, etc.) doesn't
# benefit much from a multi-stage build's main win (excluding build tools
# from the final image), and a second stage here would mostly add
# complexity without a measured size/security win worth documenting as
# fact rather than assumption.
FROM python:3.11-slim

WORKDIR /app

# System deps: git (fetch_corpus.py's version pinning uses raw.githubusercontent.com
# directly, not git, but curl/ca-certificates matter for any HTTPS fetch --
# ca-certificates especially, or every requests.get() call fails TLS verification).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs as a non-root user -- there's no reason this service needs root, and
# not dropping privileges when there's no reason not to is exactly the kind
# of gap that's cheap to close and easy to forget.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV K8S_RAG_HOST=0.0.0.0 \
    K8S_RAG_PORT=8000 \
    K8S_RAG_CONFIG=config/ci.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${K8S_RAG_PORT}/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python3", "scripts/serve.py"]
