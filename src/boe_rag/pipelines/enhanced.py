"""Enhanced CRAG pipeline.

Implements BasePipeline over a compiled LangGraph state machine. All
external dependencies (Claude LLMs, ChromaDB collection, Cohere client)
are injectable via __init__ kwargs for testability; production callers
use the default __init__ which wires them up via config + lazy singletons.

The graph topology (see spec 06 v2 §6) is:

    START -> analyze_query -> retrieve -> grade_documents
        |--(ok)--> rerank -> generate -> check_hallucination
        |                                    |--(grounded)--> END
        |                                    |--(retry once)--> generate -> check -> END
        |--(empty, rewrite budget left)--> rewrite_query -> retrieve -> grade_documents
        |--(empty, budget exhausted)--> abstain -> END

Routing is declarative (``add_conditional_edges``); each node returns
only the state fields it wrote (see nodes.py).

The public interface is identical to BaselinePipeline: one method,
``run(query) -> PipelineResult``. Evaluation code in spec 07 can iterate
over ``[BaselinePipeline(), EnhancedPipeline()]`` without branching.
"""

from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from boe_rag.config import (
    ENHANCED_COLLECTION,
    ENHANCED_TOP_K,
    GENERATION_MODEL,
    GRADING_MODEL,
    LLM_TEMPERATURE,
    RERANK_MODEL,
    RERANK_TOP_N,
)
from boe_rag.models import PipelineResult, RetrievedDocument
from boe_rag.observability import traced_run
from boe_rag.pipelines.base import BasePipeline
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
from boe_rag.pipelines.prompts import HALLUCINATION_RETRY_PROMPT
from boe_rag.pipelines.state import RAGState

logger = logging.getLogger(__name__)


def _with_retries(llm):
    """Wrap an Anthropic LLM with retry-on-429 + exponential backoff with jitter.

    Tier-1 limits (30k TPM on Sonnet 4) are easy to burst past when
    grade_documents fires 10 parallel calls each quoting ~1-3k tokens of
    source text. Retrying with jittered exponential backoff absorbs the
    burst — the TPM window rolls every 60 seconds, so a few seconds of
    wait is usually enough to recover.
    """
    return llm.with_retry(
        stop_after_attempt=4,
        wait_exponential_jitter=True,
    )


class EnhancedPipeline(BasePipeline):
    """CRAG pipeline: metadata filters + grading + rewriting + reranking + grounding."""

    def __init__(
        self,
        *,
        structured_llm: Any = None,
        grading_llm: Any = None,
        rewrite_llm: Any = None,
        generation_llm: Any = None,
        hallucination_llm: Any = None,
        collection: Any = None,
        cohere_client: Any = None,
    ) -> None:
        """Build the pipeline.

        All arguments are optional. Leave them unset for production (the
        defaults wire up real Claude/ChromaDB/Cohere via config + lazy
        singletons). Tests pass stubs in explicitly.
        """
        load_dotenv()

        # Production defaults — imported lazily inside __init__ so tests that
        # inject stubs don't trigger real client construction or API-key
        # lookups. Cohere/ChromaDB imports are at module top level (cheap).
        # Every Anthropic LLM is wrapped with retry+jitter backoff so tier-1
        # rate-limit bursts (30k TPM on Sonnet) self-recover rather than
        # raising and losing a whole query's work.
        if structured_llm is None:
            # with_structured_output must be applied BEFORE with_retry:
            # retry-wrapping returns a RunnableRetry which lacks the
            # structured-output method.
            structured_llm = _with_retries(
                ChatAnthropic(
                    model=GRADING_MODEL, temperature=LLM_TEMPERATURE
                ).with_structured_output(QueryFilters)
            )
        if grading_llm is None:
            grading_llm = _with_retries(
                ChatAnthropic(model=GRADING_MODEL, temperature=LLM_TEMPERATURE)
            )
        if rewrite_llm is None:
            rewrite_llm = _with_retries(
                ChatAnthropic(model=GRADING_MODEL, temperature=LLM_TEMPERATURE)
            )
        if generation_llm is None:
            generation_llm = _with_retries(
                ChatAnthropic(model=GENERATION_MODEL, temperature=LLM_TEMPERATURE)
            )
        if hallucination_llm is None:
            hallucination_llm = _with_retries(
                ChatAnthropic(model=GRADING_MODEL, temperature=LLM_TEMPERATURE)
            )
        if collection is None:
            from boe_rag.indexing.chroma_store import get_collection

            collection = get_collection(ENHANCED_COLLECTION)
        if cohere_client is None:
            import cohere

            cohere_client = cohere.ClientV2()

        self._collection = collection
        self._cohere = cohere_client

        self._retry_generate_llm = generation_llm  # same LLM, different prompt
        self._graph = self._build_graph(
            structured_llm=structured_llm,
            grading_llm=grading_llm,
            rewrite_llm=rewrite_llm,
            generation_llm=generation_llm,
            hallucination_llm=hallucination_llm,
            collection=collection,
            cohere_client=cohere_client,
        )

    def _build_graph(
        self,
        *,
        structured_llm: Any,
        grading_llm: Any,
        rewrite_llm: Any,
        generation_llm: Any,
        hallucination_llm: Any,
        collection: Any,
        cohere_client: Any,
    ):
        """Wire the nodes + routing into a compiled LangGraph graph.

        Both generation passes reuse the SAME generation_llm object. The
        first pass uses ENHANCED_GENERATION_PROMPT (nodes.py default); the
        retry after a failed hallucination check uses HALLUCINATION_RETRY_PROMPT
        (stricter). LangGraph routes back into a ``generate_retry`` node so
        the two prompts are selected statically at wiring time rather than
        by conditional logic inside a single generate node.
        """
        analyze_node = make_analyze_query_node(structured_llm)
        retrieve_node = make_retrieve_node(collection, top_k=ENHANCED_TOP_K)
        grade_node = make_grade_documents_node(grading_llm)
        rewrite_node = make_rewrite_query_node(rewrite_llm)
        rerank_node = make_rerank_node(
            cohere_client, model=RERANK_MODEL, top_n=RERANK_TOP_N
        )
        generate_first_node = make_generate_node(generation_llm)
        generate_retry_node = make_generate_node(
            generation_llm, prompt_template=HALLUCINATION_RETRY_PROMPT
        )
        hallucination_node = make_check_hallucination_node(hallucination_llm)
        abstain_node = make_abstain_node()

        wf = StateGraph(RAGState)
        wf.add_node("analyze_query", analyze_node)
        wf.add_node("retrieve", retrieve_node)
        wf.add_node("grade_documents", grade_node)
        wf.add_node("rewrite_query", rewrite_node)
        wf.add_node("rerank", rerank_node)
        wf.add_node("generate", generate_first_node)
        wf.add_node("generate_retry", generate_retry_node)
        wf.add_node("check_hallucination", hallucination_node)
        wf.add_node("abstain", abstain_node)

        wf.add_edge(START, "analyze_query")
        wf.add_edge("analyze_query", "retrieve")
        wf.add_edge("retrieve", "grade_documents")
        wf.add_conditional_edges(
            "grade_documents",
            route_after_grading,
            {
                "rerank": "rerank",
                "rewrite_query": "rewrite_query",
                "abstain": "abstain",
            },
        )
        wf.add_edge("rewrite_query", "retrieve")
        wf.add_edge("rerank", "generate")
        wf.add_edge("generate", "check_hallucination")
        wf.add_conditional_edges(
            "check_hallucination",
            _route_after_hallucination_with_retry,
            {
                "end": END,
                "generate": "generate_retry",
            },
        )
        wf.add_edge("generate_retry", "check_hallucination")
        wf.add_edge("abstain", END)

        return wf.compile()

    @traced_run(pipeline_name="enhanced")
    def run(self, query: str) -> PipelineResult:
        initial: RAGState = {
            "question": query,
            "pipeline_trace": [],
            "crag_rewrite_count": 0,
            "hallucination_retry_count": 0,
        }
        try:
            final_state = self._graph.invoke(initial)
        except Exception as e:
            logger.warning("Enhanced pipeline raised: %s", e)
            return PipelineResult(
                answer=f"[Pipeline error: {type(e).__name__}: {e}]",
                sources=(),
                pipeline_name="enhanced",
                chunks_retrieved=0,
                chunks_used=0,
                model=GENERATION_MODEL,
                crag_rewrites=0,
                hallucination_retries=0,
                is_grounded=None,
                metadata_filters_used=None,
                pipeline_trace=("error",),
                pre_rerank_ids=(),
                post_rerank_ids=(),
            )

        return _state_to_pipeline_result(final_state)


def _route_after_hallucination_with_retry(state: RAGState) -> str:
    """Adapter that maps the routing function to the graph edge keys."""
    # route_after_hallucination returns "end" or "generate"; the graph
    # routes "generate" to a dedicated retry node with the stricter prompt.
    return route_after_hallucination(state)


def _state_to_pipeline_result(state: RAGState) -> PipelineResult:
    """Translate the final RAGState into the BasePipeline-compatible result."""
    reranked = state.get("reranked_documents", [])
    sources = tuple(
        RetrievedDocument(
            chunk_id=doc["chunk_id"],
            text=doc["text"],
            score=float(doc.get("rerank_score", doc.get("score", 0.0))),
            metadata=None,  # ChromaDB metadata dict isn't our ChunkMetadata type
        )
        for doc in reranked
    )
    return PipelineResult(
        answer=state.get("answer", ""),
        sources=sources,
        pipeline_name="enhanced",
        chunks_retrieved=len(state.get("documents", [])),
        chunks_used=len(reranked),
        model=GENERATION_MODEL,
        crag_rewrites=state.get("crag_rewrite_count", 0),
        hallucination_retries=state.get("hallucination_retry_count", 0),
        is_grounded=state.get("is_grounded"),
        metadata_filters_used=state.get("initial_metadata_filters") or state.get("metadata_filters"),
        pipeline_trace=tuple(state.get("pipeline_trace", [])),
        pre_rerank_ids=tuple(state.get("pre_rerank_ids", []) or []),
        post_rerank_ids=tuple(state.get("post_rerank_ids", []) or []),
    )
