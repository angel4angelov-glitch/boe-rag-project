"""Tests for the LangGraph state module.

Covers:
  - RAGState TypedDict (structural — just that the type loads).
  - _build_where: translates a flat filter dict into the exact shape
    ChromaDB 1.x accepts (None / single-filter / $and-wrapped).
"""

from __future__ import annotations

from boe_rag.pipelines.state import RAGState, _build_where


# ── RAGState ────────────────────────────────────────────────


def test_ragstate_can_be_constructed_partially() -> None:
    """total=False means every field is optional at entry."""
    state: RAGState = {"question": "What was the vote?"}
    assert state["question"] == "What was the vote?"


def test_ragstate_accepts_all_documented_fields() -> None:
    """Smoke test that the TypedDict declares every field the nodes will write."""
    state: RAGState = {
        "question": "q",
        "rewritten_question": None,
        "metadata_filters": None,
        "documents": [],
        "graded_documents": [],
        "pre_rerank_ids": [],
        "reranked_documents": [],
        "post_rerank_ids": [],
        "answer": "",
        "is_grounded": None,
        "crag_rewrite_count": 0,
        "hallucination_retry_count": 0,
        "pipeline_trace": [],
    }
    assert set(state.keys()) >= {
        "question",
        "metadata_filters",
        "documents",
        "graded_documents",
        "reranked_documents",
        "answer",
        "pipeline_trace",
    }


# ── _build_where ────────────────────────────────────────────


def test_build_where_empty_dict_returns_none() -> None:
    assert _build_where({}) is None


def test_build_where_dict_with_only_none_values_returns_none() -> None:
    """Claude's Pydantic output may include None fields; those must be dropped."""
    assert _build_where({"document_type": None, "date": None}) is None


def test_build_where_single_filter_returns_bare_dict() -> None:
    """ChromaDB 1.x rejects $and with a single operand."""
    assert _build_where({"section_category": "voting"}) == {"section_category": "voting"}


def test_build_where_single_filter_skips_none_siblings() -> None:
    """Mixed input with one real value + one None => bare dict for the real value."""
    assert _build_where({"section_category": "voting", "date": None}) == {
        "section_category": "voting"
    }


def test_build_where_two_filters_uses_and() -> None:
    where = _build_where({"document_type": "MPC_minutes", "date": "2025-11"})
    assert where == {
        "$and": [{"document_type": "MPC_minutes"}, {"date": "2025-11"}]
    }


def test_build_where_three_filters_uses_and_with_three_clauses() -> None:
    where = _build_where(
        {
            "document_type": "MPC_minutes",
            "date": "2025-11",
            "section_category": "voting",
        }
    )
    assert where is not None
    assert "$and" in where
    clauses = where["$and"]
    assert len(clauses) == 3
    # Operand order is deterministic (dict insertion order) for reproducibility.
    assert clauses[0] == {"document_type": "MPC_minutes"}
    assert clauses[1] == {"date": "2025-11"}
    assert clauses[2] == {"section_category": "voting"}


def test_build_where_drops_empty_strings() -> None:
    """Empty-string values (e.g. speaker='' for non-speech) must not create a filter."""
    assert _build_where({"speaker": "", "date": "2025-11"}) == {"date": "2025-11"}
