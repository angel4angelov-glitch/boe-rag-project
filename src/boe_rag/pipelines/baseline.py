"""Baseline RAG pipeline — deliberately naive.

Flow:
    query -> ChromaDB top-K cosine -> concatenate chunks -> Claude generate
    -> PipelineResult

No grading, no rewriting, no reranking, no hallucination check, no
metadata filtering, no domain-specific prompt. Every one of those is the
enhanced pipeline's job, and the absence of them here is what makes the
baseline lose. Existing to lose is its purpose — without a credible
naive comparator, the enhanced pipeline's win is unmeasurable.

Initialisation is lazy in __init__ (not at module level): clients are
created the first time a pipeline is instantiated, not at import time.
This keeps notebook startup cheap and tests trivial to stub.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from boe_rag.config import (
    BASELINE_COLLECTION,
    BASELINE_TOP_K,
    GENERATION_MODEL,
    LLM_TEMPERATURE,
)
from boe_rag.indexing.chroma_store import get_collection
from boe_rag.models import PipelineResult, RetrievedDocument
from boe_rag.pipelines.base import BasePipeline
from boe_rag.pipelines.prompts import BASELINE_PROMPT

logger = logging.getLogger(__name__)


class BaselinePipeline(BasePipeline):
    """Top-k cosine retrieval + vanilla generation. No bells, no whistles."""

    def __init__(self) -> None:
        load_dotenv()
        self._collection = get_collection(BASELINE_COLLECTION)
        self._llm = ChatAnthropic(
            model=GENERATION_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    def run(self, query: str) -> PipelineResult:
        results = self._collection.query(
            query_texts=[query],
            n_results=BASELINE_TOP_K,
        )
        sources = _parse_chroma_results(results)

        if not sources:
            logger.warning("No chunks retrieved for query: %s", query[:100])
            return PipelineResult(
                answer="No relevant documents found.",
                sources=(),
                pipeline_name="baseline",
                chunks_retrieved=0,
                chunks_used=0,
                model=GENERATION_MODEL,
                crag_rewrites=0,
                hallucination_retries=0,
                is_grounded=None,
                metadata_filters_used=None,
                pipeline_trace=("retrieve",),
            )

        # Concatenate retrieved chunks with a clear separator so the LLM can
        # tell where one source ends and the next begins. The separator is
        # plain text (no metadata tags) — that's the baseline's whole point.
        context = "\n\n---\n\n".join(doc.text for doc in sources)
        prompt = BASELINE_PROMPT.format(context=context, question=query)
        response = self._llm.invoke(prompt)

        return PipelineResult(
            answer=response.content,
            sources=tuple(sources),
            pipeline_name="baseline",
            chunks_retrieved=len(sources),
            chunks_used=len(sources),
            model=GENERATION_MODEL,
            crag_rewrites=0,
            hallucination_retries=0,
            is_grounded=None,
            metadata_filters_used=None,
            pipeline_trace=("retrieve", "generate"),
        )


def _parse_chroma_results(results: dict) -> list[RetrievedDocument]:
    """Convert a ChromaDB query response into RetrievedDocument records.

    ChromaDB cosine returns DISTANCE (0 = identical, larger = more
    different). We expose SIMILARITY (``1 - distance``) so scores are
    directly comparable with the enhanced pipeline's Cohere rerank scores
    (also 0-1, higher = better).
    """
    ids = results["ids"][0]
    texts = results["documents"][0]
    distances = results["distances"][0]

    return [
        RetrievedDocument(
            chunk_id=chunk_id,
            text=text,
            score=round(1.0 - distance, 4),
            metadata=None,  # baseline carries no metadata
        )
        for chunk_id, text, distance in zip(ids, texts, distances)
    ]
