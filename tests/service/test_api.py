"""FastAPI service tests.

All tests stub the pipeline via monkeypatch so nothing hits real LLMs /
ChromaDB / Cohere. That keeps them fast, deterministic, and CI-safe.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from boe_rag.models import PipelineResult, RetrievedDocument


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def fake_enhanced_result() -> PipelineResult:
    return PipelineResult(
        answer="The MPC voted 5-4 to maintain Bank Rate at 3.75%.",
        sources=(
            RetrievedDocument(
                chunk_id="mpc_feb_2026_p1",
                text="At its meeting ending on 4 February 2026 the Committee voted by a majority of 5-4...",
                score=0.91,
                metadata=None,
            ),
        ),
        pipeline_name="enhanced",
        chunks_retrieved=10,
        chunks_used=3,
        model="claude-sonnet-4-20250514",
        crag_rewrites=0,
        hallucination_retries=0,
        is_grounded=True,
        metadata_filters_used={"document_type": "MPC_minutes", "date": "2026-02"},
        pipeline_trace=("analyze_query", "retrieve", "grade_documents",
                        "rerank", "generate", "check_hallucination"),
        pre_rerank_ids=("a", "b", "c"),
        post_rerank_ids=("c", "a", "b"),
    )


@pytest.fixture
def fake_error_result() -> PipelineResult:
    return PipelineResult(
        answer="[Pipeline error: RuntimeError: something broke]",
        sources=(), pipeline_name="enhanced",
        chunks_retrieved=0, chunks_used=0,
        model="claude-sonnet-4-20250514",
        crag_rewrites=0, hallucination_retries=0,
        is_grounded=None, metadata_filters_used=None,
        pipeline_trace=("error",),
    )


@pytest.fixture
def client(fake_enhanced_result, monkeypatch) -> Iterator[TestClient]:
    """TestClient with a pipeline stub that returns fake_enhanced_result."""
    from boe_rag.pipelines import BaselinePipeline, EnhancedPipeline

    monkeypatch.setattr(EnhancedPipeline, "__init__", lambda self: None)
    monkeypatch.setattr(EnhancedPipeline, "run",
                        lambda self, q: fake_enhanced_result)
    monkeypatch.setattr(BaselinePipeline, "__init__", lambda self: None)
    monkeypatch.setattr(BaselinePipeline, "run",
                        lambda self, q: fake_enhanced_result)

    from service.main import app
    with TestClient(app) as c:
        yield c


# ── Happy path ────────────────────────────────────────────


class TestQueryEndpoint:
    def test_post_query_returns_200(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was the vote?"})
        assert resp.status_code == 200

    def test_response_shape_matches_schema(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was the vote?"})
        body = resp.json()
        for key in ("answer", "sources", "pipeline_name", "chunks_retrieved",
                    "chunks_used", "is_grounded", "is_abstain",
                    "crag_rewrites", "hallucination_retries",
                    "metadata_filters_used", "pipeline_trace"):
            assert key in body
        assert body["pipeline_name"] == "enhanced"
        assert body["is_abstain"] is False
        assert len(body["sources"]) == 1
        assert body["sources"][0]["chunk_id"] == "mpc_feb_2026_p1"
        assert "text" in body["sources"][0]
        assert "text_preview" in body["sources"][0]

    def test_baseline_selection(self, client: TestClient) -> None:
        resp = client.post("/query",
                           json={"question": "What was the vote?",
                                 "pipeline": "baseline"})
        assert resp.status_code == 200

    def test_missing_question_rejected(self, client: TestClient) -> None:
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_empty_question_rejected(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422


# ── Error path ─────────────────────────────────────────────


class TestErrorHandling:
    def test_pipeline_error_maps_to_500(
        self, fake_error_result, monkeypatch,
    ) -> None:
        from boe_rag.pipelines import EnhancedPipeline
        monkeypatch.setattr(EnhancedPipeline, "__init__", lambda self: None)
        monkeypatch.setattr(EnhancedPipeline, "run",
                            lambda self, q: fake_error_result)

        from service.main import app
        with TestClient(app) as c:
            resp = c.post("/query", json={"question": "boom"})
        assert resp.status_code == 500
        assert "RuntimeError" in resp.json()["detail"]


# ── Health / Ready ─────────────────────────────────────────


class TestHealthEndpoints:
    def test_health_always_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready_returns_200_when_collection_reachable(
        self, client: TestClient, monkeypatch,
    ) -> None:
        # Patch the readiness probe to succeed
        monkeypatch.setattr("service.main._probe_chroma",
                            lambda: {"status": "ready", "n_chunks": 42})
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["n_chunks"] == 42

    def test_ready_returns_503_on_failure(self, monkeypatch) -> None:
        from service import main as svc_main
        monkeypatch.setattr(
            svc_main, "_probe_chroma",
            lambda: (_ for _ in ()).throw(RuntimeError("chroma down")),
        )
        with TestClient(svc_main.app) as c:
            resp = c.get("/ready")
        assert resp.status_code == 503


# ── Auth ───────────────────────────────────────────────────


class TestAuth:
    def test_no_auth_when_env_unset(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        resp = client.post("/query", json={"question": "What was the vote?"})
        assert resp.status_code == 200

    def test_rejects_without_key_when_auth_enabled(
        self, fake_enhanced_result, monkeypatch,
    ) -> None:
        monkeypatch.setenv("SERVICE_API_KEY", "secret123")
        from boe_rag.pipelines import EnhancedPipeline
        monkeypatch.setattr(EnhancedPipeline, "__init__", lambda self: None)
        monkeypatch.setattr(EnhancedPipeline, "run",
                            lambda self, q: fake_enhanced_result)

        from service.main import app
        with TestClient(app) as c:
            resp = c.post("/query", json={"question": "q"})
        assert resp.status_code == 401

    def test_accepts_correct_key(
        self, fake_enhanced_result, monkeypatch,
    ) -> None:
        monkeypatch.setenv("SERVICE_API_KEY", "secret123")
        from boe_rag.pipelines import EnhancedPipeline
        monkeypatch.setattr(EnhancedPipeline, "__init__", lambda self: None)
        monkeypatch.setattr(EnhancedPipeline, "run",
                            lambda self, q: fake_enhanced_result)

        from service.main import app
        with TestClient(app) as c:
            resp = c.post(
                "/query",
                json={"question": "q"},
                headers={"X-API-Key": "secret123"},
            )
        assert resp.status_code == 200


# ── OpenAPI ────────────────────────────────────────────────


class TestOpenAPI:
    def test_openapi_schema_available(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        for p in ("/query", "/health", "/ready"):
            assert p in paths
