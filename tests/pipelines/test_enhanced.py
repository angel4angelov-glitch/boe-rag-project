"""End-to-end tests for the compiled EnhancedPipeline using stubbed clients.

These tests DO NOT hit real APIs. They inject deterministic stubs for
Claude (structured + plain), ChromaDB, and Cohere to exercise the full
LangGraph orchestration: analyze -> retrieve -> grade -> (rewrite ->
retrieve -> grade) -> rerank -> generate -> hallucination -> END, plus
the abstain branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from boe_rag.config import BASELINE_COLLECTION, GENERATION_MODEL  # noqa: F401 (sanity import)
from boe_rag.models import PipelineResult
from boe_rag.pipelines.enhanced import EnhancedPipeline
from boe_rag.pipelines.nodes import QueryFilters


# ── Reusable stubs (mirror test_nodes.py shape) ─────────────


@dataclass
class _StubMsg:
    content: str


class _QueueLLM:
    """Plain Claude stub — queue of canned text responses."""

    def __init__(self, responses: list[str]) -> None:
        self._q = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: Any) -> _StubMsg:
        self.calls.append(str(prompt))
        return _StubMsg(content=self._q.pop(0))

    def batch(self, prompts: list[Any]) -> list[_StubMsg]:
        return [self.invoke(p) for p in prompts]


class _StructuredLLM:
    """Claude .with_structured_output(QueryFilters) stub."""

    def __init__(self, out: QueryFilters) -> None:
        self.out = out

    def invoke(self, prompt: Any) -> QueryFilters:
        return self.out


class _StubCollection:
    def __init__(self, responses: list[dict]) -> None:
        self._q = list(responses)
        self.query_calls: list[dict] = []

    def query(self, *, query_texts, n_results, where=None, **kw):  # noqa: ANN001
        self.query_calls.append(
            {"query_texts": query_texts, "n_results": n_results, "where": where}
        )
        return self._q.pop(0)


@dataclass
class _RerankHit:
    index: int
    relevance_score: float


@dataclass
class _RerankResp:
    results: list[_RerankHit]


class _StubCohere:
    def __init__(self, response: _RerankResp) -> None:
        self.response = response
        self.calls = 0

    def rerank(self, *, model, query, documents, top_n):
        self.calls += 1
        return self.response


def _chroma(ids, texts, metas, dists):
    return {
        "ids": [ids],
        "documents": [texts],
        "distances": [dists],
        "metadatas": [metas],
    }


def _make_pipeline(
    structured_llm=None,
    grade_llm=None,
    rewrite_llm=None,
    generate_llm=None,
    hallucination_llm=None,
    collection=None,
    cohere=None,
) -> EnhancedPipeline:
    """Build an EnhancedPipeline with whatever stubs are provided."""
    return EnhancedPipeline(
        structured_llm=structured_llm,
        grading_llm=grade_llm,
        rewrite_llm=rewrite_llm,
        generation_llm=generate_llm,
        hallucination_llm=hallucination_llm,
        collection=collection,
        cohere_client=cohere,
    )


# ── Happy path: filter -> retrieve -> grade -> rerank -> generate -> OK ──


def test_happy_path_end_to_end() -> None:
    pipeline = _make_pipeline(
        structured_llm=_StructuredLLM(
            QueryFilters(document_type="MPC_minutes", date="2025-11")
        ),
        collection=_StubCollection(
            [
                _chroma(
                    ids=["c1", "c2"],
                    texts=["vote body", "context body"],
                    metas=[{"date": "2025-11"}, {"date": "2025-11"}],
                    dists=[0.2, 0.3],
                )
            ]
        ),
        grade_llm=_QueueLLM(["yes", "yes"]),
        rewrite_llm=_QueueLLM([]),  # never called on happy path
        cohere=_StubCohere(
            _RerankResp(
                results=[
                    _RerankHit(index=0, relevance_score=0.95),
                    _RerankHit(index=1, relevance_score=0.42),
                ]
            )
        ),
        generate_llm=_QueueLLM(["The MPC voted 5-4 to maintain Bank Rate."]),
        hallucination_llm=_QueueLLM(["yes"]),
    )

    result = pipeline.run("What was the MPC vote in November 2025?")

    assert isinstance(result, PipelineResult)
    assert result.pipeline_name == "enhanced"
    assert "5-4" in result.answer
    assert result.chunks_retrieved == 2
    assert result.chunks_used == 2  # both passed grading + rerank
    assert result.crag_rewrites == 0
    assert result.hallucination_retries == 0
    assert result.is_grounded is True
    # Filter was built with 2 keys -> $and wrapper.
    assert result.metadata_filters_used == {
        "$and": [{"document_type": "MPC_minutes"}, {"date": "2025-11"}]
    }
    # Trace reflects actual path (no rewrite, no retry).
    assert tuple(result.pipeline_trace) == (
        "analyze_query",
        "retrieve",
        "grade_documents",
        "rerank",
        "generate",
        "check_hallucination",
    )


# ── CRAG rewrite loop ───────────────────────────────────────


def test_crag_rewrite_triggers_when_first_grading_finds_nothing() -> None:
    pipeline = _make_pipeline(
        structured_llm=_StructuredLLM(QueryFilters(date="2025-11")),
        collection=_StubCollection(
            [
                # First retrieval: one doc, grading rejects.
                _chroma(
                    ids=["c_bad"],
                    texts=["irrelevant"],
                    metas=[{"date": "2025-11"}],
                    dists=[0.3],
                ),
                # Second retrieval (after rewrite, filters cleared): one doc, grading accepts.
                _chroma(
                    ids=["c_good"],
                    texts=["relevant content"],
                    metas=[{}],
                    dists=[0.2],
                ),
            ]
        ),
        grade_llm=_QueueLLM(["no", "yes"]),
        rewrite_llm=_QueueLLM(["rewritten form"]),
        cohere=_StubCohere(_RerankResp(results=[_RerankHit(index=0, relevance_score=0.9)])),
        generate_llm=_QueueLLM(["generated"]),
        hallucination_llm=_QueueLLM(["yes"]),
    )

    result = pipeline.run("ambiguous initial question")

    assert result.crag_rewrites == 1
    # Verify the SECOND retrieval was called with no filters (rewrite clears them).
    collection = pipeline._collection  # type: ignore[attr-defined]
    assert collection.query_calls[1]["where"] is None
    # Trace shows the rewrite detour.
    assert "rewrite_query" in result.pipeline_trace


# ── Abstain branch ──────────────────────────────────────────


def test_abstain_branch_triggers_on_two_empty_grading_passes() -> None:
    pipeline = _make_pipeline(
        structured_llm=_StructuredLLM(QueryFilters()),  # no filters
        collection=_StubCollection(
            [
                _chroma(ids=["c1"], texts=["a"], metas=[{}], dists=[0.3]),
                _chroma(ids=["c2"], texts=["b"], metas=[{}], dists=[0.3]),
            ]
        ),
        grade_llm=_QueueLLM(["no", "no"]),  # both passes reject
        rewrite_llm=_QueueLLM(["rewrite attempt"]),
        cohere=_StubCohere(_RerankResp(results=[])),  # never called
        generate_llm=_QueueLLM([]),  # never called
        hallucination_llm=_QueueLLM([]),  # never called
    )

    result = pipeline.run("What is the Federal Reserve's view?")

    assert "Bank of England" in result.answer
    assert "does not appear" in result.answer
    assert result.is_grounded is None  # abstain path doesn't run hallucination check
    assert result.chunks_used == 0
    assert result.sources == ()
    assert "abstain" in result.pipeline_trace
    assert "generate" not in result.pipeline_trace  # never called


# ── Hallucination retry loop ───────────────────────────────


def test_hallucination_retry_triggers_once_on_ungrounded_first_answer() -> None:
    pipeline = _make_pipeline(
        structured_llm=_StructuredLLM(QueryFilters()),
        collection=_StubCollection(
            [_chroma(ids=["c1"], texts=["source"], metas=[{}], dists=[0.2])]
        ),
        grade_llm=_QueueLLM(["yes"]),
        rewrite_llm=_QueueLLM([]),
        cohere=_StubCohere(
            _RerankResp(results=[_RerankHit(index=0, relevance_score=0.8)])
        ),
        generate_llm=_QueueLLM(
            ["ungrounded first answer", "grounded second answer"]
        ),
        hallucination_llm=_QueueLLM(["no", "yes"]),
    )

    result = pipeline.run("some question")

    assert result.hallucination_retries == 1
    assert result.is_grounded is True
    assert result.answer == "grounded second answer"
    assert result.pipeline_trace.count("generate") == 2
    assert result.pipeline_trace.count("check_hallucination") == 2


# ── Rerank capture for spec-09 demo log ─────────────────────


def test_reranking_capture_exposes_pre_post_ordering_via_rerank_score() -> None:
    """Confirms rerank swapped top-1 from c1 -> c3; sources are in rerank order."""
    pipeline = _make_pipeline(
        structured_llm=_StructuredLLM(QueryFilters()),
        collection=_StubCollection(
            [
                _chroma(
                    ids=["c1", "c2", "c3"],
                    texts=["a", "b", "c"],
                    metas=[{}, {}, {}],
                    dists=[0.2, 0.25, 0.3],  # c1 top by cosine
                )
            ]
        ),
        grade_llm=_QueueLLM(["yes", "yes", "yes"]),
        rewrite_llm=_QueueLLM([]),
        cohere=_StubCohere(
            _RerankResp(
                results=[
                    _RerankHit(index=2, relevance_score=0.95),  # c3 wins rerank
                    _RerankHit(index=0, relevance_score=0.50),
                ]
            )
        ),
        generate_llm=_QueueLLM(["answer"]),
        hallucination_llm=_QueueLLM(["yes"]),
    )

    result = pipeline.run("q")

    # sources are in rerank order -> c3 first.
    assert result.sources[0].chunk_id == "c3"
    assert result.sources[1].chunk_id == "c1"
