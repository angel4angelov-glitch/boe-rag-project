"""FastAPI app: ``POST /query``, ``GET /health``, ``GET /ready``.

Lifecycle:
  - ``lifespan`` startup: eagerly construct ``EnhancedPipeline`` so the
    first request doesn't pay the ~2-3s cold-start cost. BaselinePipeline
    is constructed lazily on first use.
  - Sync route handlers dispatch to FastAPI's threadpool automatically,
    so concurrent requests don't block on blocking ``pipeline.run()``
    calls. For true async we'd need ``ainvoke`` throughout — Tier C.

Error mapping:
  - Pipeline errors (``answer`` starts with ``"[Pipeline error"``)
    surface as HTTP 500 with the error string in the detail.
  - Abstains come back as HTTP 200 with ``is_abstain=true`` — a
    deliberate "I don't know" is a successful response, not an error.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from boe_rag.evaluation.adapters import is_abstain
from boe_rag.models import PipelineResult

from service.auth import require_api_key
from service.schemas import QueryRequest, QueryResponse, ReadyResponse, SourceItem

logger = logging.getLogger(__name__)


_state: dict[str, Any] = {"enhanced": None, "baseline": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Eagerly warm the enhanced pipeline on startup."""
    load_dotenv()
    logger.info("Service starting — initialising EnhancedPipeline (cold start)")
    from boe_rag.pipelines import EnhancedPipeline
    _state["enhanced"] = EnhancedPipeline()
    logger.info("EnhancedPipeline ready")
    yield
    logger.info("Service shutting down")


app = FastAPI(
    title="BoE RAG Service",
    description="Corrective RAG over Bank of England policy documents.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive by default for demo. Tighten origins for real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline(name: str):
    """Lazy-init baseline on first use; reuse warmed enhanced."""
    if name == "enhanced":
        pipe = _state.get("enhanced")
        if pipe is None:
            from boe_rag.pipelines import EnhancedPipeline
            pipe = EnhancedPipeline()
            _state["enhanced"] = pipe
        return pipe
    pipe = _state.get("baseline")
    if pipe is None:
        from boe_rag.pipelines import BaselinePipeline
        pipe = BaselinePipeline()
        _state["baseline"] = pipe
    return pipe


def _to_response(result: PipelineResult) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceItem(
                chunk_id=d.chunk_id,
                score=float(d.score),
                text=d.text,
                text_preview=d.text[:200],
                metadata=None,  # ChunkMetadata isn't JSON-friendly; omit
            )
            for d in result.sources
        ],
        pipeline_name=result.pipeline_name,  # type: ignore[arg-type]
        chunks_retrieved=result.chunks_retrieved,
        chunks_used=result.chunks_used,
        is_grounded=result.is_grounded,
        is_abstain=is_abstain(result.answer),
        crag_rewrites=result.crag_rewrites,
        hallucination_retries=result.hallucination_retries,
        metadata_filters_used=result.metadata_filters_used,
        pipeline_trace=list(result.pipeline_trace),
    )


@app.post("/query", response_model=QueryResponse,
          dependencies=[Depends(require_api_key)])
def query(req: QueryRequest) -> QueryResponse:
    """Run a query through the selected pipeline."""
    pipe = _get_pipeline(req.pipeline)
    result = pipe.run(req.question)
    if result.answer.startswith("[Pipeline error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.answer,
        )
    return _to_response(result)


@app.get("/health")
def health() -> dict:
    """Liveness probe — always 200 if the process is up."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness probe — verifies ChromaDB is reachable."""
    try:
        probe = _probe_chroma()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{type(e).__name__}: {e}",
        )
    return ReadyResponse(**probe)


def _probe_chroma() -> dict:
    """Count chunks in the enhanced collection — fast, fails loud if down."""
    from boe_rag.config import ENHANCED_COLLECTION
    from boe_rag.indexing.chroma_store import get_collection
    col = get_collection(ENHANCED_COLLECTION)
    n = col.count()
    return {"status": "ready", "n_chunks": n}
