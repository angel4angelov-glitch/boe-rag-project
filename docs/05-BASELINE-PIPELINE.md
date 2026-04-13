# 05 — Baseline Pipeline

## Objective
A deliberately naive RAG pipeline: fixed chunks, cosine similarity, no grading, no reranking, vanilla prompt. This exists to lose to the enhanced pipeline — and to prove that the enhanced pipeline's improvements are real.

## Depends on
- 01-PROJECT-SETUP (`PipelineResult`, `RetrievedDocument` from `boe_rag.models`; config constants)
- 04-INDEXING (ChromaDB `boe_baseline` collection populated; `get_collection()`)

## Deliverables
- [ ] `src/boe_rag/pipelines/base.py` — abstract pipeline interface
- [ ] `src/boe_rag/pipelines/baseline.py` — naive RAG pipeline
- [ ] `src/boe_rag/pipelines/prompts.py` — prompt templates (baseline + enhanced share this file)
- [ ] Works end-to-end on a sample query, returns `PipelineResult`

---

## Pipeline Flow

```
Query → ChromaDB top-5 cosine similarity → Concatenate chunks → Claude generates answer → Return PipelineResult
```

No grading, no rewriting, no reranking, no hallucination check.

---

## Abstract Pipeline Interface (`base.py`)

Both pipelines implement the same interface so evaluation code is pipeline-agnostic:

```python
from abc import ABC, abstractmethod
from boe_rag.models import PipelineResult


class BasePipeline(ABC):
    """Abstract interface for RAG pipelines."""

    @abstractmethod
    def run(self, query: str) -> PipelineResult:
        """Execute the pipeline on a query and return a structured result."""
        ...
```

The evaluation module (07) calls `pipeline.run(query)` and gets `PipelineResult` regardless of which pipeline it's talking to.

---

## Implementation (`baseline.py`)

```python
import logging

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from boe_rag.config import (
    GENERATION_MODEL, LLM_TEMPERATURE, BASELINE_TOP_K, BASELINE_COLLECTION,
)
from boe_rag.indexing.chroma_store import get_collection
from boe_rag.models import PipelineResult, RetrievedDocument
from boe_rag.pipelines.base import BasePipeline
from boe_rag.pipelines.prompts import BASELINE_PROMPT

logger = logging.getLogger(__name__)


class BaselinePipeline(BasePipeline):
    """Deliberately naive RAG: retrieve top-k, concatenate, generate."""

    def __init__(self) -> None:
        load_dotenv()
        self._collection = get_collection(BASELINE_COLLECTION)
        self._llm = ChatAnthropic(
            model=GENERATION_MODEL,       # from config
            temperature=LLM_TEMPERATURE,  # from config
        )

    def run(self, query: str) -> PipelineResult:
        # 1. Retrieve
        results = self._collection.query(
            query_texts=[query],
            n_results=BASELINE_TOP_K,  # 5 from config
        )

        # 2. Parse ChromaDB response into RetrievedDocument objects
        sources = _parse_chroma_results(results)

        # 3. Handle empty retrieval
        if not sources:
            logger.warning("No chunks retrieved for query: %s", query[:100])
            return PipelineResult(
                answer="No relevant documents found.",
                sources=(),
                pipeline_name="baseline",
                chunks_retrieved=0, chunks_used=0,
                model=GENERATION_MODEL,
                crag_rewrites=0, hallucination_retries=0,
                is_grounded=None, metadata_filters_used=None,
                pipeline_trace=["retrieve"],
            )

        # 4. Build context string
        context = "\n\n---\n\n".join(doc.text for doc in sources)

        # 5. Generate
        prompt = BASELINE_PROMPT.format(context=context, question=query)
        response = self._llm.invoke(prompt)
        answer = response.content  # AIMessage → str

        # 6. Return structured result
        return PipelineResult(
            answer=answer,
            sources=sources,
            pipeline_name="baseline",
            chunks_retrieved=len(sources),
            chunks_used=len(sources),       # baseline uses all retrieved
            model=GENERATION_MODEL,
            crag_rewrites=0,                # baseline: no CRAG
            hallucination_retries=0,        # baseline: no hallucination check
            is_grounded=None,               # baseline: not checked
            metadata_filters_used=None,     # baseline: no filters
            pipeline_trace=["retrieve", "generate"],
        )


def _parse_chroma_results(results: dict) -> list[RetrievedDocument]:
    """Convert ChromaDB query response to RetrievedDocument objects.

    ChromaDB returns cosine DISTANCE (lower = more similar).
    We convert to similarity: score = 1 - distance.
    """
    documents = []
    # ChromaDB returns lists-of-lists (one list per query text)
    ids = results["ids"][0]
    texts = results["documents"][0]
    distances = results["distances"][0]

    for chunk_id, text, distance in zip(ids, texts, distances):
        documents.append(RetrievedDocument(
            chunk_id=chunk_id,
            text=text,
            score=round(1.0 - distance, 4),  # distance → similarity
            metadata=None,  # baseline has no metadata
        ))
    return documents
```

### Key implementation decisions

- **Lazy initialization in `__init__`**, not at module level. Importing the module doesn't trigger ChromaDB or Anthropic client creation.
- **`load_dotenv()` in `__init__`** — ensures API keys are available before creating clients.
- **ChromaDB distance → similarity conversion**: `score = 1 - distance`. ChromaDB cosine returns distance (0 = identical, 2 = opposite). Similarity = 1 - distance. This makes scores comparable with the enhanced pipeline's Cohere rerank scores (also 0-1, higher = better).
- **`response.content`** — `ChatAnthropic.invoke()` returns an `AIMessage` object. The actual text is in `.content`.
- **All config values imported**, not hardcoded.

---

## Prompt Template (`prompts.py`)

```python
# ── Baseline prompt ──────────────────────────────────────────

BASELINE_PROMPT = """You are a helpful assistant. Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""


# ── Enhanced prompt (used by 06-ENHANCED-PIPELINE) ───────────

ENHANCED_GENERATION_PROMPT = """You are a specialist analyst answering questions about Bank of England monetary policy using official BoE documents.

Rules:
1. ONLY use information from the provided source documents
2. Cite the source document for each claim (e.g., "According to the November 2025 MPC minutes, paragraph 19...")
3. If the documents don't contain enough information to fully answer the question, say so explicitly — do not speculate
4. Use precise policy language (e.g., "Bank Rate" not "interest rate", "CPI inflation" not just "inflation")
5. When quoting vote splits, give the exact numbers

Source documents:
{context}

Question: {question}

Answer:"""


# (Additional enhanced prompts — grading, rewriting, hallucination —
#  defined in 06-ENHANCED-PIPELINE.md)
```

**The baseline prompt is deliberately basic.** No citation instructions, no grounding rules, no domain-specific language. The enhanced prompt is the opposite. This prompt quality difference is an intentional part of the evaluation delta.

---

## What Makes This Deliberately Naive

| Aspect | Baseline | Enhanced (for contrast) |
|--------|----------|------------------------|
| Chunks | Fixed 500-token, no overlap, no metadata | Section-aware, metadata-tagged |
| Retrieval | Top-5 cosine, no filtering | Metadata-filtered + CRAG grading |
| Reranking | None | Cohere rerank |
| Query rewriting | None | LLM rewrites on retrieval failure |
| Hallucination check | None | LLM verifies grounding |
| Prompt | Vanilla "answer based on context" | Domain-specific with citation rules |
| Failure handling | None — always returns an answer | Detects and corrects bad retrieval |
| Return metadata | Minimal (`pipeline_trace` = 2 steps) | Full (`pipeline_trace` shows CRAG path) |

---

## Edge Cases

- **Empty retrieval**: ChromaDB can return fewer than `n_results` if the collection is small. If `results["ids"][0]` is empty, the pipeline returns `PipelineResult` with `answer="No relevant documents found."`, `sources=[]`, `chunks_retrieved=0`.
- **LLM error**: If `ChatAnthropic.invoke()` raises (rate limit, timeout), the exception propagates. The notebook catches it and logs the failure — the pipeline itself doesn't retry. (The enhanced pipeline has retry logic; the baseline doesn't. That's part of the delta.)

---

## Acceptance Criteria

1. `BaselinePipeline().run("What was the MPC vote in November 2025?")` returns a `PipelineResult` with non-empty `answer`
2. `sources` contains up to 5 `RetrievedDocument` objects with `score` in 0-1 range (similarity, not distance)
3. `pipeline_name == "baseline"`, `crag_rewrites == 0`, `is_grounded is None`
4. No metadata filtering, no grading, no reranking — confirm by reading the code
5. Uses `config.GENERATION_MODEL` and `config.BASELINE_TOP_K` — no hardcoded values
6. Implements `BasePipeline` abstract interface from `base.py`
7. `prompts.py` contains both `BASELINE_PROMPT` and `ENHANCED_GENERATION_PROMPT` (shared file)
8. Importing `baseline.py` does NOT trigger client creation (lazy init in `__init__`)
