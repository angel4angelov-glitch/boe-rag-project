"""LangGraph state schema for the enhanced CRAG pipeline + filter helpers.

RAGState is the typed shape every node reads from and writes partial
updates to. Fields are documented inline so a future reader can trace
which node produces each one.

_build_where translates a flat filter dict (as emitted by analyze_query's
Pydantic output) into the exact ChromaDB ``where`` format:

    0 usable filters  -> None         (no where clause at all)
    1 usable filter   -> {k: v}       (bare dict — ChromaDB 1.x REJECTS $and
                                        with a single operand, this avoids it)
    2+ usable filters -> {"$and": [...]}   (AND-joined clauses)

"Usable" means non-None and non-empty-string — None-valued fields from
Pydantic's ``exclude_none=True`` are still possible via other sources,
and empty strings (e.g. speaker="" for non-speech chunks) must never
leak into the filter.
"""

from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict, total=False):
    """Typed state flowing through the enhanced pipeline graph.

    Every field is optional at graph entry (``total=False``); each node
    writes only the keys it is responsible for. LangGraph merges node
    updates into the evolving state.

    Field lifecycle:
      question                   -> written by caller
      metadata_filters           -> analyze_query
      rewritten_question         -> rewrite_query (if triggered)
      documents                  -> retrieve
      graded_documents           -> grade_documents
      pre_rerank_ids             -> rerank
      reranked_documents         -> rerank (or pass-through if skipped)
      post_rerank_ids            -> rerank
      answer                     -> generate (or abstain)
      is_grounded                -> check_hallucination (or abstain=None)
      crag_rewrite_count         -> rewrite_query
      hallucination_retry_count  -> check_hallucination
      pipeline_trace             -> appended to by every node
    """

    question: str
    rewritten_question: str | None
    metadata_filters: dict | None
    initial_metadata_filters: dict | None   # preserved across rewrite_query (which clears metadata_filters)
    out_of_corpus: bool                     # written by analyze_query when question is outside BoE scope
    documents: list[dict]
    graded_documents: list[dict]
    pre_rerank_ids: list[str]
    reranked_documents: list[dict]
    post_rerank_ids: list[str]
    answer: str
    is_grounded: bool | None
    crag_rewrite_count: int
    hallucination_retry_count: int
    pipeline_trace: list[str]


def _build_where(filters: dict) -> dict | None:
    """Translate a flat filter dict into a ChromaDB-compatible ``where`` clause.

    Args:
        filters: Keys are ChromaDB metadata field names; values are the
            desired matches. ``None`` and empty-string values are dropped
            (they represent "no preference" or "field not applicable").

    Returns:
        None when no usable filters exist, a bare ``{k: v}`` dict for a
        single filter, or a ``{"$and": [...]}`` wrapper for two or more.
    """
    usable = {k: v for k, v in filters.items() if v is not None and v != ""}
    if not usable:
        return None
    if len(usable) == 1:
        k, v = next(iter(usable.items()))
        return {k: v}
    return {"$and": [{k: v} for k, v in usable.items()]}
