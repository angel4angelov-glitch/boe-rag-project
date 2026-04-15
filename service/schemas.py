"""Pydantic request/response schemas for the FastAPI service.

The response mirrors ``boe_rag.models.PipelineResult`` with two
additions:
  - ``is_abstain``: computed boolean (so callers don't need to
    string-match the abstain message)
  - ``text_preview`` on each source (first 200 chars for UI truncation)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    pipeline: Literal["baseline", "enhanced"] = "enhanced"

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be blank")
        return v.strip()


class SourceItem(BaseModel):
    chunk_id: str
    score: float
    text: str
    text_preview: str
    metadata: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    pipeline_name: Literal["baseline", "enhanced"]
    chunks_retrieved: int
    chunks_used: int
    is_grounded: bool | None
    is_abstain: bool
    crag_rewrites: int
    hallucination_retries: int
    metadata_filters_used: dict | None
    pipeline_trace: list[str]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    n_chunks: int | None = None
    detail: str | None = None
