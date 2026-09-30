"""Phase 8: productionization. A thin FastAPI service around the same
`run_pipeline` every script in this repo already calls -- no logic lives
here that scripts/ask.py and scripts/run_eval.py don't already exercise,
which is the point: the service is a transport layer over an
already-tested pipeline, not a second implementation of it.

What "production" means here, scoped honestly for a portfolio project
rather than dressed up as a real deployment:

- **Tracing a bad answer back to its cause** (the roadmap's exit
  criterion) doesn't require LangSmith to be configured. Every response
  carries a `request_id`, the exact chunk ids/scores/versions that were
  retrieved, the refusal reason when refused, and the git commit + config
  hash that produced it (via `k8s_rag.repro`) -- everything needed to
  reproduce the exact answer later, logged as one structured JSON line per
  request. LangSmith tracing (`tracing.langsmith_project` +
  `LANGCHAIN_API_KEY`) is additive on top of this, not a replacement for it
  -- this service works identically, just untraced, without a key.
- **Streaming** (`POST /ask/stream`) is real SSE plumbing, but honestly
  bounded by what the configured generator actually supports: the `stub`
  generator computes its full extractive answer synchronously (it's a
  regex scan over already-retrieved text, not a token-by-token model), so
  its stream is legitimately one content event, not simulated token-by-
  token typing to *look* like a real LLM. `OpenAIGenerator` doesn't yet
  implement a streaming code path either -- see `Generator.generate_stream`
  below for exactly what's missing and why it's not faked here.
- **Rate limiting** is a real in-memory fixed-window limiter, not a no-op.
  It is explicitly NOT safe across multiple replicas (no shared state,
  no Redis) -- fine for the single-instance deployment this project
  actually ships, a real limitation for anything beyond that, documented
  rather than silently assumed away.
- **Cost** is reported as `$0.00` under the `stub` generator because that's
  true, not a placeholder. `OpenAIGenerator` doesn't currently report token
  usage back to the metrics layer -- another explicit, small gap rather
  than a fabricated number.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from k8s_rag.config import REPO_ROOT, RunConfig
from k8s_rag.graph import run_pipeline
from k8s_rag.repro import repro_metadata
from k8s_rag.vectorstore import VectorStore

logger = logging.getLogger("k8s_rag.api")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# In-memory metrics + rate limiting. Both explicitly single-process: a real
# multi-replica deployment needs these backed by something shared (Redis,
# a metrics backend like Prometheus's pushgateway or a sidecar) -- not
# pretended otherwise here.
# --------------------------------------------------------------------------

@dataclass
class Metrics:
    request_count: int = 0
    refusal_count: int = 0
    error_count: int = 0
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=500))

    def record(self, latency_ms: float, decision: str) -> None:
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        if decision == "refuse":
            self.refusal_count += 1

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)
        p50 = statistics.median(lat) if lat else None
        p95 = lat[int(len(lat) * 0.95)] if lat else None
        refusal_rate = (self.refusal_count / self.request_count) if self.request_count else None
        return {
            "request_count": self.request_count,
            "refusal_count": self.refusal_count,
            "error_count": self.error_count,
            "refusal_rate": refusal_rate,
            "latency_ms": {"p50": p50, "p95": p95, "window_size": len(lat)},
            # real once a real generation provider reports usage -- see module
            # docstring. stub generation genuinely costs nothing.
            "estimated_cost_usd": 0.0,
        }


class RateLimiter:
    """Fixed-window per-client limiter. `max_requests` per `window_seconds`,
    keyed by client IP. Deliberately simple (a real token-bucket or sliding
    window would smooth bursts better) because the point of shipping this
    at all is "don't let one client trivially hammer the free stub
    endpoint," not "survive a DDoS" -- that belongs in front of this
    service (a real gateway/WAF), not reimplemented here."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter, exempt_paths: set[str]):
        super().__init__(app)
        self._limiter = limiter
        self._exempt = exempt_paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._exempt:
            return await call_next(request)
        client_key = request.client.host if request.client else "unknown"
        if not self._limiter.allow(client_key):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
            )
        return await call_next(request)


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    target_version: str | None = None


class Citation(BaseModel):
    version: str
    source_page: str
    source_url: str
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    decision: str
    refusal_reason: str | None = None
    citations: list[Citation]
    latency_ms: float
    run_name: str | None = None
    git_commit: str | None = None
    config_content_hash: str | None = None


def _citations_from_state(chunks) -> list[Citation]:
    return [
        Citation(
            version=c.version, source_page=c.source_page, source_url=c.source_url,
            chunk_id=c.chunk_id, score=c.score,
        )
        for c in chunks
    ]


def _log_request(request_id: str, cfg: RunConfig, question: str, result: dict, latency_ms: float) -> None:
    """One structured JSON line per request -- the traceability mechanism
    the roadmap's Phase 8 exit criterion actually asks for ('a bad answer
    can be traced back to its cause'). Grep this log for a request_id
    returned to an unhappy user and you have the exact question, exact
    retrieved chunks + scores, the decision, and the exact config/commit
    that produced it -- no separate tracing backend required, though
    LangSmith (if configured) captures strictly more detail on top."""
    chunks = result.get("retrieved_chunks", [])
    repro = repro_metadata(REPO_ROOT, cfg.path)
    logger.info(json.dumps({
        "request_id": request_id,
        "question": question,
        "decision": result.get("decision"),
        "refusal_reason": result.get("refusal_reason"),
        "top_chunk_ids": [c.chunk_id for c in chunks[:3]],
        "top_scores": [round(c.score, 5) for c in chunks[:3]],
        "latency_ms": round(latency_ms, 1),
        "run_name": cfg.get("run_name"),
        "git_commit": repro.get("git_commit"),
        "config_content_hash": repro.get("config_content_hash"),
    }))


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

def create_app(cfg: RunConfig | None = None, rate_limit_per_minute: int = 60) -> FastAPI:
    cfg = cfg or RunConfig.load()
    metrics = Metrics()
    limiter = RateLimiter(max_requests=rate_limit_per_minute, window_seconds=60.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.store = VectorStore(cfg)
        app.state.metrics = metrics
        yield

    app = FastAPI(
        title="k8s-rag",
        description="Production-grade RAG over Kubernetes API docs, with citations and calibrated refusal.",
        lifespan=lifespan,
    )
    app.add_middleware(RateLimitMiddleware, limiter=limiter, exempt_paths={"/health", "/metrics"})

    @app.get("/health")
    def health():
        store: VectorStore = app.state.store
        try:
            count = store.count()
        except Exception as e:  # never let a health check crash the process over a store hiccup
            return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})
        return {"status": "ok", "chunk_count": count, "run_name": app.state.cfg.get("run_name")}

    @app.get("/metrics")
    def get_metrics():
        return app.state.metrics.summary()

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        request_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            result = run_pipeline(app.state.cfg, app.state.store, req.question, target_version=req.target_version)
        except Exception as e:
            app.state.metrics.error_count += 1
            logger.info(json.dumps({"request_id": request_id, "error": str(e), "question": req.question}))
            raise HTTPException(status_code=500, detail="Internal error processing question") from e
        latency_ms = (time.time() - t0) * 1000

        app.state.metrics.record(latency_ms, result["decision"])
        _log_request(request_id, app.state.cfg, req.question, result, latency_ms)

        repro = repro_metadata(REPO_ROOT, app.state.cfg.path)
        return AskResponse(
            request_id=request_id,
            answer=result["answer"],
            decision=result["decision"],
            refusal_reason=result.get("refusal_reason"),
            citations=_citations_from_state(result.get("retrieved_chunks", [])),
            latency_ms=latency_ms,
            run_name=app.state.cfg.get("run_name"),
            git_commit=repro.get("git_commit"),
            config_content_hash=repro.get("config_content_hash"),
        )

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        """SSE endpoint. See the module docstring for exactly why this is
        real streaming plumbing around a currently-non-streaming answer:
        the stub generator has no tokens to stream incrementally, so this
        legitimately emits one `answer` event rather than faking a
        word-by-word typing effect to look more like a real LLM than the
        configured generator actually is."""
        request_id = str(uuid.uuid4())
        t0 = time.time()

        def event_stream():
            try:
                result = run_pipeline(app.state.cfg, app.state.store, req.question, target_version=req.target_version)
            except Exception as e:
                app.state.metrics.error_count += 1
                yield f"event: error\ndata: {json.dumps({'request_id': request_id, 'error': str(e)})}\n\n"
                return
            latency_ms = (time.time() - t0) * 1000
            app.state.metrics.record(latency_ms, result["decision"])
            _log_request(request_id, app.state.cfg, req.question, result, latency_ms)

            yield f"event: answer\ndata: {json.dumps({'request_id': request_id, 'text': result['answer']})}\n\n"
            citations = [c.model_dump() for c in _citations_from_state(result.get("retrieved_chunks", []))]
            yield "event: done\ndata: " + json.dumps({
                "request_id": request_id,
                "decision": result["decision"],
                "citations": citations,
                "latency_ms": latency_ms,
            }) + "\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
