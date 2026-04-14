"""LangGraph node implementations for the enhanced CRAG pipeline.

Every node is built via a ``make_*_node(deps...)`` factory that returns a
``callable(state) -> partial_state`` suitable for ``workflow.add_node``.
The factory pattern decouples the node body from its external dependencies
(LLM, ChromaDB collection, Cohere client, prompt templates), so tests can
inject deterministic stubs and assert on state updates.

Each node returns ONLY the state fields it writes, plus an appended
``pipeline_trace`` entry — LangGraph merges the partial into the
evolving RAGState automatically.

Node catalogue:
    analyze_query          Pydantic-typed Claude structured output -> filter dict
    retrieve               ChromaDB query with optional where clause
    grade_documents        Per-doc yes/no relevance, parallel via batch()
    rewrite_query          LLM rewrite + DROP filters so the retry widens search
    rerank                 Cohere rerank-v3.5; skipped when <=1 graded doc
    generate               Claude with the spec-05 enhanced generation prompt
    check_hallucination    yes/no groundedness of the generated answer
    abstain                Terminal fallback with fixed "out of corpus" message

Routing functions:
    route_after_grading         -> "rerank" | "rewrite_query" | "abstain"
    route_after_hallucination   -> "end" | "generate"
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from boe_rag.chunking.metadata import normalise_speaker
from boe_rag.pipelines.prompts import (
    ANALYZE_QUERY_PROMPT,
    ENHANCED_GENERATION_PROMPT,
    GRADING_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    HALLUCINATION_RETRY_PROMPT,
    REWRITE_QUERY_PROMPT,
)
from boe_rag.pipelines.state import RAGState, _build_where

logger = logging.getLogger(__name__)

NodeFn = Callable[[RAGState], dict]


# ── Structured-output schema for analyze_query ─────────────


_SectionCategoryLiteral = Literal[
    "global_economy",
    "inflation",
    "labour_market",
    "demand_output",
    "policy_discussion",
    "voting",
    "individual_statement",
    "box_analysis",
    "risk_assessment",
    "financial_stability",
    "forward_guidance",
    "speech_main",
]


class QueryFilters(BaseModel):
    """Claude-extracted metadata filters. Every field is optional.

    Only fields the LLM is confident about should be populated. ``None`` on
    every field means "use an unfiltered search" — grading and reranking
    still filter downstream.
    """

    document_type: Literal["MPR", "FSR", "MPC_minutes", "speech"] | None = None
    date: str | None = Field(
        default=None, description="YYYY-MM, e.g. '2025-11' for November 2025"
    )
    section_category: _SectionCategoryLiteral | None = None
    speaker: str | None = Field(
        default=None,
        description="First name + last name (no honorifics or middle initials)",
    )


def _append_trace(state: RAGState, name: str) -> list[str]:
    """Return the new pipeline_trace with ``name`` appended (non-destructive)."""
    return list(state.get("pipeline_trace", [])) + [name]


# ── analyze_query ───────────────────────────────────────────


def make_analyze_query_node(structured_llm: Any) -> NodeFn:
    """Build the query-analysis node.

    Args:
        structured_llm: An object whose ``.invoke(prompt) -> QueryFilters``
            call returns Pydantic-validated output. In production, build
            this with ``ChatAnthropic(...).with_structured_output(QueryFilters)``.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "analyze_query")
        try:
            filters: QueryFilters = structured_llm.invoke(
                ANALYZE_QUERY_PROMPT.format(question=state["question"])
            )
        except Exception as e:
            logger.warning("analyze_query failed: %s; continuing with no filters", e)
            return {"metadata_filters": None, "pipeline_trace": trace}

        raw = filters.model_dump(exclude_none=True)
        if "speaker" in raw and raw["speaker"]:
            raw["speaker"] = normalise_speaker(raw["speaker"])
        where = _build_where(raw)
        # Track BOTH the live filter (which rewrite_query will clear) and a
        # frozen initial copy so the final pipeline surface can report what
        # analyze_query originally inferred even after a rewrite has fired.
        return {
            "metadata_filters": where,
            "initial_metadata_filters": where,
            "pipeline_trace": trace,
        }

    return _node


# ── retrieve ────────────────────────────────────────────────


def make_retrieve_node(collection: Any, *, top_k: int) -> NodeFn:
    """Build the retrieval node.

    Uses ``state['rewritten_question']`` when present (CRAG second pass),
    otherwise the original ``state['question']``. Applies whatever
    ``metadata_filters`` are in state (None falls through to unfiltered).
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "retrieve")
        query = state.get("rewritten_question") or state["question"]
        where = state.get("metadata_filters")
        results = collection.query(
            query_texts=[query], n_results=top_k, where=where
        )
        documents = _parse_chroma_results(results)
        return {"documents": documents, "pipeline_trace": trace}

    return _node


def _parse_chroma_results(results: dict) -> list[dict]:
    """ChromaDB query dict -> list of flat chunk dicts with similarity scores.

    ChromaDB returns cosine DISTANCE (lower = more similar). We convert to
    similarity (``1 - distance``) so scores are comparable with Cohere
    rerank scores that will be stitched onto these docs later.
    """
    ids = results["ids"][0]
    texts = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[{}] * len(ids)])[0]
    return [
        {
            "chunk_id": cid,
            "text": text,
            "score": round(1.0 - dist, 4),
            "metadata": meta or {},
        }
        for cid, text, dist, meta in zip(ids, texts, distances, metadatas)
    ]


# ── grade_documents ─────────────────────────────────────────


def make_grade_documents_node(llm: Any) -> NodeFn:
    """Build the per-doc relevance grader.

    Runs the LLM in parallel via ``llm.batch()``. A response is kept only
    if, lower-cased and stripped of trailing punctuation, it starts with
    "yes". Everything else (including "YES.", "maybe", "perhaps") is
    accepted/rejected conservatively: only clear "yes" variants pass.

    Rate-limit protection is handled upstream via ``llm.with_retry()``
    (exponential backoff with jitter absorbs tier-1 TPM bursts). We do
    NOT truncate document text here — experiments showed that truncating
    past ~1500 chars causes the grader to miss relevant context in long
    box analyses and reject otherwise-good chunks.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "grade_documents")
        docs = state.get("documents", [])
        if not docs:
            return {"graded_documents": [], "pipeline_trace": trace}

        prompts = [
            GRADING_PROMPT.format(document=d["text"], question=state["question"])
            for d in docs
        ]
        responses = llm.batch(prompts)
        kept = [
            doc
            for doc, resp in zip(docs, responses)
            if _is_yes(resp.content)
        ]
        return {"graded_documents": kept, "pipeline_trace": trace}

    return _node


def _is_yes(text: str) -> bool:
    """True if ``text`` unambiguously starts with 'yes' (case-insensitive)."""
    cleaned = text.strip().lower().rstrip(".!?,")
    return cleaned == "yes" or cleaned.startswith("yes ")


# ── rewrite_query ───────────────────────────────────────────


def make_rewrite_query_node(llm: Any) -> NodeFn:
    """Build the CRAG query-rewriter node.

    Clears ``metadata_filters`` so the retry is unfiltered — rewriting the
    question alone cannot rescue a too-narrow filter that returned zero
    hits on the first pass.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "rewrite_query")
        resp = llm.invoke(REWRITE_QUERY_PROMPT.format(question=state["question"]))
        rewritten = resp.content.strip()
        return {
            "rewritten_question": rewritten,
            "metadata_filters": None,
            "crag_rewrite_count": state.get("crag_rewrite_count", 0) + 1,
            "pipeline_trace": trace,
        }

    return _node


# ── rerank ──────────────────────────────────────────────────


def make_rerank_node(cohere_client: Any, *, model: str, top_n: int) -> NodeFn:
    """Build the Cohere reranking node.

    Skips the rerank API entirely when the graded list has <= 1 document —
    reordering a single doc is a no-op and wastes an API call.

    Captures both pre- and post-rerank chunk-id orderings in state so the
    demo log (spec 09) can visualise reranker impact.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "rerank")
        graded = state.get("graded_documents", [])
        pre_ids = [d["chunk_id"] for d in graded]

        if len(graded) <= 1:
            return {
                "pre_rerank_ids": pre_ids,
                "reranked_documents": list(graded),
                "post_rerank_ids": list(pre_ids),
                "pipeline_trace": trace,
            }

        response = cohere_client.rerank(
            model=model,
            query=state.get("rewritten_question") or state["question"],
            documents=[d["text"] for d in graded],
            top_n=min(top_n, len(graded)),
        )
        reordered = [
            {**graded[r.index], "rerank_score": r.relevance_score}
            for r in response.results
        ]
        return {
            "pre_rerank_ids": pre_ids,
            "reranked_documents": reordered,
            "post_rerank_ids": [d["chunk_id"] for d in reordered],
            "pipeline_trace": trace,
        }

    return _node


# ── generate ────────────────────────────────────────────────


def make_generate_node(llm: Any, *, prompt_template: str = ENHANCED_GENERATION_PROMPT) -> NodeFn:
    """Build the answer-generation node.

    The default prompt is ENHANCED_GENERATION_PROMPT. A stricter variant
    (HALLUCINATION_RETRY_PROMPT) is injected when the hallucination check
    fails on the first pass — done by re-invoking this factory with the
    retry prompt at graph-wiring time or by the pipeline choosing a
    different node instance.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "generate")
        context = "\n\n---\n\n".join(
            d["text"] for d in state.get("reranked_documents", [])
        )
        prompt = prompt_template.format(
            context=context, question=state["question"]
        )
        response = llm.invoke(prompt)
        return {"answer": response.content, "pipeline_trace": trace}

    return _node


# ── check_hallucination ─────────────────────────────────────


def make_check_hallucination_node(llm: Any) -> NodeFn:
    """Build the groundedness-check node.

    Ambiguous LLM responses are treated as "no" (triggers retry) — the
    conservative move when the check is itself uncertain.
    """

    def _node(state: RAGState) -> dict:
        trace = _append_trace(state, "check_hallucination")
        context = "\n\n---\n\n".join(
            d["text"] for d in state.get("reranked_documents", [])
        )
        prompt = HALLUCINATION_CHECK_PROMPT.format(
            context=context, answer=state["answer"]
        )
        response = llm.invoke(prompt)
        grounded = _is_yes(response.content)
        return {
            "is_grounded": grounded,
            "hallucination_retry_count": state.get("hallucination_retry_count", 0)
            + (0 if grounded else 1),
            "pipeline_trace": trace,
        }

    return _node


# ── abstain ─────────────────────────────────────────────────


_ABSTAIN_MESSAGE = (
    "This question does not appear to be answerable from the Bank of England document corpus."
)


def make_abstain_node() -> NodeFn:
    """Build the terminal-abstain node.

    Reached when two consecutive grading passes find no relevant documents.
    Returns a fixed, neutral message so RAGAS evaluation of out-of-scope
    queries is reproducible.
    """

    def _node(state: RAGState) -> dict:
        return {
            "answer": _ABSTAIN_MESSAGE,
            "reranked_documents": [],
            "is_grounded": None,
            "pipeline_trace": _append_trace(state, "abstain"),
        }

    return _node


# ── Routing ─────────────────────────────────────────────────


def route_after_grading(
    state: RAGState,
) -> Literal["rerank", "rewrite_query", "abstain"]:
    """Decide the post-grading edge.

    - At least one relevant doc -> rerank
    - Zero relevant, no rewrite yet -> rewrite_query (budget: 1)
    - Zero relevant, already rewrote once -> abstain (terminal)
    """
    if state.get("graded_documents"):
        return "rerank"
    if state.get("crag_rewrite_count", 0) < 1:
        return "rewrite_query"
    return "abstain"


def route_after_hallucination(state: RAGState) -> Literal["end", "generate"]:
    """Decide the post-hallucination-check edge.

    - Grounded -> end
    - Not grounded AND retry budget (1) still available -> generate (retry)
    - Not grounded AND budget exhausted -> end (return ungrounded answer
      with the flag set to False; caller surfaces as a warning)
    """
    if state.get("is_grounded"):
        return "end"
    # After failed first pass, counter has been bumped to 1. Retry once,
    # then even another failure exits to prevent infinite loops.
    if state.get("hallucination_retry_count", 0) <= 1:
        return "generate"
    return "end"
