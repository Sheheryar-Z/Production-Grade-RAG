"""Phase 8: the FastAPI service. Runs against config/ci.yaml's already-
ingested local_tfidf data (the same persisted Chroma collection every other
script in this repo uses) -- no network, no real LLM, exactly like the rest
of the test suite. Skips cleanly if that data hasn't been ingested yet
(fresh checkout, scripts/ingest.py never run) rather than failing with a
confusing Chroma error.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from k8s_rag.api import RateLimiter, create_app  # noqa: E402
from k8s_rag.config import RunConfig  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ci.yaml"


def _has_ingested_data() -> bool:
    try:
        cfg = RunConfig.load(str(CONFIG_PATH))
        from k8s_rag.vectorstore import VectorStore
        return VectorStore(cfg).count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_ingested_data(),
    reason="config/ci.yaml has no ingested data in this environment -- run scripts/ingest.py first",
)


@pytest.fixture()
def client():
    cfg = RunConfig.load(str(CONFIG_PATH))
    app = create_app(cfg, rate_limit_per_minute=1000)
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunk_count"] > 0


def test_ask_answerable_question(client):
    resp = client.post("/ask", json={
        "question": "What apiVersion should I use to create an Ingress resource?",
        "target_version": "1.31",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("generate", "refuse")
    assert "request_id" in body and len(body["request_id"]) > 0
    assert isinstance(body["citations"], list)
    assert body["latency_ms"] > 0
    # traceability fields the Phase 8 exit criterion actually asks for
    assert body["run_name"] == "ci"
    assert body["config_content_hash"] is not None


def test_ask_unanswerable_question_is_refused(client):
    resp = client.post("/ask", json={
        "question": "Which company should I choose to host my Kubernetes cluster -- GKE, EKS, or AKS?",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "refuse"
    assert body["refusal_reason"] is not None


def test_ask_rejects_empty_question(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_ask_stream_returns_event_stream(client):
    resp = client.post("/ask/stream", json={"question": "What apiVersion for Ingress?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: answer" in resp.text
    assert "event: done" in resp.text


def test_metrics_reflects_requests(client):
    client.post("/ask", json={"question": "What apiVersion for Ingress in 1.31?"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_count"] >= 1
    assert body["estimated_cost_usd"] == 0.0  # stub generator -- genuinely free, not a placeholder


def test_metrics_and_health_are_exempt_from_rate_limiting():
    cfg = RunConfig.load(str(CONFIG_PATH))
    app = create_app(cfg, rate_limit_per_minute=1)
    with TestClient(app) as c:
        for _ in range(5):
            assert c.get("/health").status_code == 200
            assert c.get("/metrics").status_code == 200


def test_ask_is_rate_limited():
    cfg = RunConfig.load(str(CONFIG_PATH))
    app = create_app(cfg, rate_limit_per_minute=2)
    with TestClient(app) as c:
        statuses = [
            c.post("/ask", json={"question": "What apiVersion for Ingress?"}).status_code
            for _ in range(4)
        ]
    assert statuses[:2] == [200, 200]
    assert 429 in statuses[2:]


def test_rate_limiter_unit_sliding_window():
    limiter = RateLimiter(max_requests=2, window_seconds=60.0)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False  # 3rd request in the window
    assert limiter.allow("client-b") is True  # different client, separate bucket
