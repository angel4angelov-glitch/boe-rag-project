"""Tests for BaselinePipeline using stubbed ChromaDB collection + LLM.

End-to-end behaviour with real APIs is exercised by a separate notebook /
script — these tests pin the pipeline contract without burning credits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from boe_rag.config import BASELINE_TOP_K, GENERATION_MODEL
from boe_rag.models import PipelineResult, RetrievedDocument
from boe_rag.pipelines.baseline import BaselinePipeline, _parse_chroma_results


# ── Pure helper: distance → similarity ──────────────────────


def test_parse_chroma_results_converts_distance_to_similarity() -> None:
    """ChromaDB returns distances (lower = better). We expose similarity (higher = better)."""
    chroma_payload = {
        "ids": [["a", "b", "c"]],
        "documents": [["text a", "text b", "text c"]],
        "distances": [[0.1, 0.4, 0.9]],
    }
    docs = _parse_chroma_results(chroma_payload)
    assert [d.score for d in docs] == [0.9, 0.6, 0.1]
    assert [d.chunk_id for d in docs] == ["a", "b", "c"]
    # Baseline carries no metadata.
    assert all(d.metadata is None for d in docs)


def test_parse_chroma_results_handles_empty_query() -> None:
    empty = {"ids": [[]], "documents": [[]], "distances": [[]]}
    assert _parse_chroma_results(empty) == []


# ── Stubs for end-to-end pipeline behaviour ─────────────────


@dataclass
class _StubAIMessage:
    """Minimal stand-in for langchain_anthropic's AIMessage."""

    content: str


@dataclass
class _StubLLM:
    """Records the last prompt it was invoked with and returns canned text."""

    canned: str = "stub answer"
    last_prompt: str | None = None

    def invoke(self, prompt: Any) -> _StubAIMessage:
        self.last_prompt = prompt
        return _StubAIMessage(content=self.canned)


class _StubCollection:
    """Collection.query stub returning a fixed response."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.query_calls: list[dict] = []

    def query(self, *, query_texts, n_results, **kw):  # noqa: ANN001
        self.query_calls.append({"query_texts": query_texts, "n_results": n_results, **kw})
        return self._response


@pytest.fixture
def make_pipeline(monkeypatch):
    """Factory that builds a BaselinePipeline with stubbed collection + LLM."""

    def _build(chroma_response: dict, llm: _StubLLM | None = None):
        collection = _StubCollection(chroma_response)
        the_llm = llm or _StubLLM()
        # Skip __init__'s real env load, ChromaDB connect, and Anthropic client.
        pipeline = BaselinePipeline.__new__(BaselinePipeline)
        pipeline._collection = collection  # type: ignore[attr-defined]
        pipeline._llm = the_llm  # type: ignore[attr-defined]
        return pipeline, collection, the_llm

    return _build


def test_run_returns_pipelineresult_with_naive_metadata(make_pipeline) -> None:
    pipeline, _, _ = make_pipeline(
        chroma_response={
            "ids": [["c1", "c2"]],
            "documents": [["alpha", "beta"]],
            "distances": [[0.2, 0.3]],
        }
    )
    result = pipeline.run("What was the vote?")
    assert isinstance(result, PipelineResult)
    assert result.pipeline_name == "baseline"
    assert result.answer == "stub answer"
    assert result.chunks_retrieved == 2
    assert result.chunks_used == 2
    # Baseline-specific markers (the design contract):
    assert result.crag_rewrites == 0
    assert result.hallucination_retries == 0
    assert result.is_grounded is None
    assert result.metadata_filters_used is None
    assert tuple(result.pipeline_trace) == ("retrieve", "generate")
    assert result.model == GENERATION_MODEL


def test_run_uses_baseline_top_k_from_config(make_pipeline) -> None:
    pipeline, collection, _ = make_pipeline(
        chroma_response={"ids": [["c1"]], "documents": [["x"]], "distances": [[0.1]]}
    )
    pipeline.run("any question")
    assert collection.query_calls[0]["n_results"] == BASELINE_TOP_K


def test_run_concatenates_retrieved_text_into_prompt(make_pipeline) -> None:
    llm = _StubLLM(canned="generated")
    pipeline, _, _ = make_pipeline(
        chroma_response={
            "ids": [["c1", "c2"]],
            "documents": [["FIRST CHUNK", "SECOND CHUNK"]],
            "distances": [[0.2, 0.3]],
        },
        llm=llm,
    )
    pipeline.run("Q?")
    assert "FIRST CHUNK" in llm.last_prompt
    assert "SECOND CHUNK" in llm.last_prompt
    assert "Q?" in llm.last_prompt


def test_run_handles_empty_retrieval_without_calling_llm(make_pipeline) -> None:
    llm = _StubLLM()
    pipeline, _, the_llm = make_pipeline(
        chroma_response={"ids": [[]], "documents": [[]], "distances": [[]]},
        llm=llm,
    )
    result = pipeline.run("anything")
    assert result.chunks_retrieved == 0
    assert result.sources == ()
    assert result.answer == "No relevant documents found."
    assert tuple(result.pipeline_trace) == ("retrieve",)
    # LLM never invoked when retrieval is empty — no wasted API call.
    assert the_llm.last_prompt is None
