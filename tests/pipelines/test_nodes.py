"""Tests for the 8 graph nodes + 2 routing functions.

Every node is exercised through the public factory function
``make_*_node(llm, ...)`` so tests can inject deterministic stubs and
assert on the (partial) state update returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from boe_rag.pipelines.nodes import (
    QueryFilters,
    make_abstain_node,
    make_analyze_query_node,
    make_check_hallucination_node,
    make_generate_node,
    make_grade_documents_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_query_node,
    route_after_grading,
    route_after_hallucination,
)


# ── Shared stubs ────────────────────────────────────────────


@dataclass
class _StubAIMessage:
    content: str


class _StubLLM:
    """ChatAnthropic replacement — queue up canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: Any) -> _StubAIMessage:
        self.calls.append(prompt if isinstance(prompt, str) else str(prompt))
        return _StubAIMessage(content=self._responses.pop(0))

    def batch(self, prompts: list[Any]) -> list[_StubAIMessage]:
        return [self.invoke(p) for p in prompts]


class _StubStructuredLLM:
    """ChatAnthropic.with_structured_output(QueryFilters) replacement."""

    def __init__(self, filters: QueryFilters | Exception) -> None:
        self._out = filters
        self.calls: list[str] = []

    def invoke(self, prompt: Any) -> QueryFilters:
        self.calls.append(prompt if isinstance(prompt, str) else str(prompt))
        if isinstance(self._out, Exception):
            raise self._out
        return self._out


class _StubCollection:
    def __init__(self, response: dict) -> None:
        self._response = response
        self.query_calls: list[dict] = []

    def query(self, *, query_texts, n_results, where=None, **kw):  # noqa: ANN001
        self.query_calls.append(
            {"query_texts": query_texts, "n_results": n_results, "where": where, **kw}
        )
        return self._response


@dataclass
class _StubRerankResult:
    index: int
    relevance_score: float


@dataclass
class _StubRerankResponse:
    results: list[_StubRerankResult]


class _StubCohere:
    def __init__(self, results: list[_StubRerankResult] | Exception) -> None:
        self._out = results
        self.calls: list[dict] = []

    def rerank(self, *, model, query, documents, top_n):
        self.calls.append(
            {"model": model, "query": query, "documents": documents, "top_n": top_n}
        )
        if isinstance(self._out, Exception):
            raise self._out
        return _StubRerankResponse(results=list(self._out))


def _chroma_response(ids: list[str], texts: list[str], metas: list[dict], dists: list[float]) -> dict:
    return {
        "ids": [ids],
        "documents": [texts],
        "distances": [dists],
        "metadatas": [metas],
    }


def _starter_state(**overrides) -> dict:
    base = {
        "question": "What was the MPC vote in November 2025?",
        "pipeline_trace": [],
        "crag_rewrite_count": 0,
        "hallucination_retry_count": 0,
    }
    base.update(overrides)
    return base


# ── analyze_query ───────────────────────────────────────────


def test_analyze_query_extracts_three_field_filter() -> None:
    llm = _StubStructuredLLM(
        QueryFilters(document_type="MPC_minutes", date="2025-11", section_category="voting")
    )
    node = make_analyze_query_node(llm)
    out = node(_starter_state())
    assert out["metadata_filters"] == {
        "$and": [
            {"document_type": "MPC_minutes"},
            {"date": "2025-11"},
            {"section_category": "voting"},
        ]
    }
    assert out["pipeline_trace"][-1] == "analyze_query"


def test_analyze_query_single_field_returns_bare_dict() -> None:
    llm = _StubStructuredLLM(QueryFilters(section_category="box_analysis"))
    node = make_analyze_query_node(llm)
    out = node(_starter_state())
    assert out["metadata_filters"] == {"section_category": "box_analysis"}


def test_analyze_query_no_confident_filters_returns_none() -> None:
    llm = _StubStructuredLLM(QueryFilters())  # all fields None
    node = make_analyze_query_node(llm)
    out = node(_starter_state())
    assert out["metadata_filters"] is None


def test_analyze_query_normalises_speaker_before_filtering() -> None:
    """Claude may emit 'Catherine L Mann' or 'Professor Alan Taylor' — must normalise."""
    llm = _StubStructuredLLM(QueryFilters(speaker="Professor Catherine L Mann"))
    node = make_analyze_query_node(llm)
    out = node(_starter_state())
    assert out["metadata_filters"] == {"speaker": "Catherine Mann"}


def test_analyze_query_handles_llm_error_as_no_filters() -> None:
    llm = _StubStructuredLLM(RuntimeError("LLM timed out"))
    node = make_analyze_query_node(llm)
    out = node(_starter_state())
    assert out["metadata_filters"] is None
    assert out["pipeline_trace"][-1] == "analyze_query"


# ── retrieve ────────────────────────────────────────────────


def test_retrieve_uses_rewritten_question_when_present() -> None:
    collection = _StubCollection(
        _chroma_response(ids=["c1"], texts=["body"], metas=[{"k": "v"}], dists=[0.2])
    )
    node = make_retrieve_node(collection, top_k=5)
    state = _starter_state(
        rewritten_question="rewritten form of the question",
        metadata_filters={"date": "2025-11"},
    )
    out = node(state)
    assert collection.query_calls[0]["query_texts"] == ["rewritten form of the question"]
    assert collection.query_calls[0]["where"] == {"date": "2025-11"}
    assert len(out["documents"]) == 1


def test_retrieve_falls_back_to_original_question() -> None:
    collection = _StubCollection(
        _chroma_response(ids=["c1"], texts=["body"], metas=[{}], dists=[0.2])
    )
    node = make_retrieve_node(collection, top_k=10)
    out = node(_starter_state())
    assert collection.query_calls[0]["query_texts"] == [
        "What was the MPC vote in November 2025?"
    ]
    assert collection.query_calls[0]["n_results"] == 10


def test_retrieve_converts_distance_to_similarity_in_documents() -> None:
    collection = _StubCollection(
        _chroma_response(
            ids=["c1", "c2"],
            texts=["a", "b"],
            metas=[{"date": "2025-11"}, {"date": "2025-12"}],
            dists=[0.25, 0.50],
        )
    )
    node = make_retrieve_node(collection, top_k=5)
    out = node(_starter_state())
    docs = out["documents"]
    assert docs[0]["score"] == 0.75
    assert docs[1]["score"] == 0.50
    # Metadata preserved.
    assert docs[0]["metadata"] == {"date": "2025-11"}


# ── grade_documents ─────────────────────────────────────────


def test_grade_documents_keeps_yes_drops_no() -> None:
    llm = _StubLLM(responses=["yes", "no", "yes"])
    node = make_grade_documents_node(llm)
    state = _starter_state(
        documents=[
            {"chunk_id": "c1", "text": "relevant"},
            {"chunk_id": "c2", "text": "tangential"},
            {"chunk_id": "c3", "text": "relevant too"},
        ]
    )
    out = node(state)
    assert [d["chunk_id"] for d in out["graded_documents"]] == ["c1", "c3"]
    assert len(llm.calls) == 3


def test_grade_documents_empty_input_returns_empty_output() -> None:
    llm = _StubLLM(responses=[])
    node = make_grade_documents_node(llm)
    out = node(_starter_state(documents=[]))
    assert out["graded_documents"] == []
    assert llm.calls == []


def test_grade_documents_treats_ambiguous_as_no() -> None:
    """Conservative: anything other than 'yes' means drop the chunk."""
    llm = _StubLLM(responses=["YES", "perhaps", "maybe", "yes."])
    node = make_grade_documents_node(llm)
    state = _starter_state(
        documents=[
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c2", "text": "b"},
            {"chunk_id": "c3", "text": "c"},
            {"chunk_id": "c4", "text": "d"},
        ]
    )
    out = node(state)
    # YES (uppercase) and "yes." both accepted; "perhaps"/"maybe" rejected.
    ids = [d["chunk_id"] for d in out["graded_documents"]]
    assert "c1" in ids
    assert "c4" in ids
    assert "c2" not in ids
    assert "c3" not in ids


# ── rewrite_query ───────────────────────────────────────────


def test_rewrite_query_stores_rewrite_and_clears_filters() -> None:
    llm = _StubLLM(responses=["November 2025 Bank Rate vote MPC"])
    node = make_rewrite_query_node(llm)
    state = _starter_state(
        metadata_filters={"date": "2025-11"},
        crag_rewrite_count=0,
    )
    out = node(state)
    assert out["rewritten_question"] == "November 2025 Bank Rate vote MPC"
    assert out["metadata_filters"] is None  # critical: filters dropped for retry
    assert out["crag_rewrite_count"] == 1


# ── rerank ──────────────────────────────────────────────────


def test_rerank_reorders_and_captures_pre_post_ids() -> None:
    client = _StubCohere(
        results=[
            _StubRerankResult(index=2, relevance_score=0.9),
            _StubRerankResult(index=0, relevance_score=0.7),
        ]
    )
    node = make_rerank_node(client, model="rerank-v3.5", top_n=5)
    state = _starter_state(
        graded_documents=[
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c2", "text": "b"},
            {"chunk_id": "c3", "text": "c"},
        ]
    )
    out = node(state)
    assert out["pre_rerank_ids"] == ["c1", "c2", "c3"]
    assert out["post_rerank_ids"] == ["c3", "c1"]
    # Rerank score attached to reordered docs.
    assert out["reranked_documents"][0]["chunk_id"] == "c3"
    assert out["reranked_documents"][0]["rerank_score"] == 0.9


def test_rerank_skips_when_one_or_zero_graded_docs() -> None:
    """No value reordering a single doc — skip API call, pass through."""
    client = _StubCohere(results=RuntimeError("should not be called"))
    node = make_rerank_node(client, model="rerank-v3.5", top_n=5)
    state = _starter_state(graded_documents=[{"chunk_id": "c1", "text": "solo"}])
    out = node(state)
    assert out["reranked_documents"] == [{"chunk_id": "c1", "text": "solo"}]
    assert out["pre_rerank_ids"] == ["c1"]
    assert out["post_rerank_ids"] == ["c1"]
    assert client.calls == []  # API NOT called


# ── generate ────────────────────────────────────────────────


def test_generate_uses_reranked_documents_in_context() -> None:
    llm = _StubLLM(responses=["The MPC voted 5-4 to maintain Bank Rate at 4%."])
    node = make_generate_node(llm, prompt_template="ctx={context}\nq={question}\nanswer:")
    state = _starter_state(
        reranked_documents=[
            {"chunk_id": "c1", "text": "FIRST CHUNK"},
            {"chunk_id": "c2", "text": "SECOND CHUNK"},
        ]
    )
    out = node(state)
    assert out["answer"] == "The MPC voted 5-4 to maintain Bank Rate at 4%."
    # Both chunks concatenated into the prompt context.
    assert "FIRST CHUNK" in llm.calls[0]
    assert "SECOND CHUNK" in llm.calls[0]


# ── check_hallucination ─────────────────────────────────────


def test_check_hallucination_grounded_sets_flag_true() -> None:
    llm = _StubLLM(responses=["yes"])
    node = make_check_hallucination_node(llm)
    state = _starter_state(
        answer="grounded answer",
        reranked_documents=[{"chunk_id": "c1", "text": "source"}],
    )
    out = node(state)
    assert out["is_grounded"] is True


def test_check_hallucination_ungrounded_sets_flag_false_and_increments_retry() -> None:
    llm = _StubLLM(responses=["no"])
    node = make_check_hallucination_node(llm)
    state = _starter_state(
        answer="ungrounded",
        reranked_documents=[{"chunk_id": "c1", "text": "source"}],
    )
    out = node(state)
    assert out["is_grounded"] is False
    assert out["hallucination_retry_count"] == 1


# ── abstain ─────────────────────────────────────────────────


def test_abstain_returns_fixed_message_and_empty_sources() -> None:
    node = make_abstain_node()
    out = node(_starter_state())
    assert "Bank of England" in out["answer"]
    assert out["reranked_documents"] == []
    assert out["is_grounded"] is None
    assert out["pipeline_trace"][-1] == "abstain"


# ── Routing ─────────────────────────────────────────────────


def test_route_after_grading_rerank_when_relevant_docs_exist() -> None:
    state = {"graded_documents": [{"chunk_id": "c1"}], "crag_rewrite_count": 0}
    assert route_after_grading(state) == "rerank"


def test_route_after_grading_rewrite_on_first_empty() -> None:
    state = {"graded_documents": [], "crag_rewrite_count": 0}
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_abstain_on_second_empty() -> None:
    state = {"graded_documents": [], "crag_rewrite_count": 1}
    assert route_after_grading(state) == "abstain"


def test_route_after_hallucination_end_when_grounded() -> None:
    state = {"is_grounded": True, "hallucination_retry_count": 1}
    assert route_after_hallucination(state) == "end"


def test_route_after_hallucination_retry_once_on_first_failure() -> None:
    state = {"is_grounded": False, "hallucination_retry_count": 1}
    # After check_hallucination has set count=1, we retry once.
    assert route_after_hallucination(state) == "generate"


def test_route_after_hallucination_terminates_after_retry_budget() -> None:
    state = {"is_grounded": False, "hallucination_retry_count": 2}
    assert route_after_hallucination(state) == "end"
