# 06 — Enhanced CRAG Pipeline

## Objective
A Corrective RAG pipeline using LangGraph with: metadata-filtered retrieval, document grading, query rewriting, Cohere reranking, grounded generation, and hallucination checking.

## Depends on
04-INDEXING (ChromaDB `boe_enhanced` collection populated)

## Deliverables
- [ ] `src/pipelines/state.py` — LangGraph state schema
- [ ] `src/pipelines/nodes.py` — all graph node functions
- [ ] `src/pipelines/prompts.py` — all prompt templates
- [ ] `src/pipelines/enhanced.py` — compiled LangGraph graph + callable interface
- [ ] Returns same schema as baseline: `{"answer": str, "sources": list[dict], "metadata": dict}`
- [ ] Works end-to-end with CRAG loop triggering on adversarial queries

---

## Architecture: The LangGraph Graph

```
START
  │
  ▼
[analyze_query]──────────────────────────────┐
  │                                           │
  ▼                                           │
[retrieve] (with optional metadata filters)   │
  │                                           │
  ▼                                           │
[grade_documents]                             │
  │         │                                 │
  │ (all    │ (some/all                       │
  │ relevant)│ irrelevant)                    │
  │         ▼                                 │
  │    [rewrite_query]                        │
  │         │                                 │
  │         ▼                                 │
  │    [retrieve] (re-retrieve, max 1 retry)  │
  │         │                                 │
  │         ▼                                 │
  │    [grade_documents] (second pass)        │
  │         │                                 │
  ├─────────┘                                 │
  ▼                                           │
[rerank] (Cohere rerank)                      │
  │                                           │
  ▼                                           │
[generate] (Claude with grounding prompt)     │
  │                                           │
  ▼                                           │
[check_hallucination]                         │
  │         │                                 │
  │(grounded)│(not grounded)                  │
  │         ▼                                 │
  │    [generate] (retry with stricter prompt)│
  │         │                                 │
  ├─────────┘                                 │
  ▼                                           │
END                                           │
```

---

## LangGraph State Schema

```python
from typing import TypedDict
from langgraph.graph import MessagesState

class RAGState(TypedDict):
    question: str                    # original user question
    rewritten_question: str | None   # rewritten query (if CRAG triggered)
    documents: list[dict]            # retrieved chunks with scores
    graded_documents: list[dict]     # chunks that passed grading
    reranked_documents: list[dict]   # chunks after Cohere rerank
    answer: str                      # generated answer
    metadata_filters: dict | None    # ChromaDB where clause (if applicable)
    is_grounded: bool                # hallucination check result
    crag_rewrite_count: int          # number of query rewrites (0 or 1)
    hallucination_retry_count: int   # number of hallucination retries (0 or 1)
    pipeline_trace: list[str]        # ordered list of nodes visited
```

---

## Node Implementations

### 1. analyze_query
**Purpose**: Determine if the query targets a specific document type, date, or section. Produce optional ChromaDB metadata filters.

```
Input: question
Output: metadata_filters (dict | None)
```

Use Claude to classify the query:
- "What was the MPC vote in November 2025?" → `{"$and": [{"document_type": "MPC_minutes"}, {"date": "2025-11"}, {"section_category": "voting"}]}`
- "How did inflation projections change over 2025?" → No filter (cross-document)
- "What did Box D describe?" → `{"section_category": "box_analysis"}`

**Implementation**: Structured output with Pydantic model. If Claude can't confidently identify filters, return None (fall back to unfiltered search).

### 2. retrieve
**Purpose**: Query ChromaDB with optional metadata filters. Return top-10 chunks.

```
Input: question (or rewritten_question), metadata_filters
Output: documents (list of {chunk_id, text, score, metadata})
```

- If `metadata_filters` is set: use `collection.query(where=metadata_filters, n_results=10)`
- If None: use `collection.query(n_results=10)` (unfiltered)
- Return 10 chunks (more than baseline's 5 — grading will filter down)

### 3. grade_documents
**Purpose**: LLM grades each retrieved document for relevance. Binary: relevant or not.

```
Input: question, documents
Output: graded_documents (only relevant ones), routing decision
```

**Grading prompt** (adapted from LangGraph Adaptive RAG):
```
You are a grader assessing the relevance of a retrieved document to a user question
about Bank of England monetary policy.

Document: {document}
Question: {question}

If the document contains information that would help answer the question about
BoE policy, monetary decisions, financial stability, or economic conditions,
grade it as relevant.

Score: "yes" or "no"
```

**Routing logic**:
- If ≥1 document graded relevant → proceed to rerank
- If all documents irrelevant AND crag_rewrite_count == 0 → route to rewrite_query
- If all documents irrelevant AND crag_rewrite_count == 1 → proceed to generate with whatever we have (don't loop forever)

### 4. rewrite_query
**Purpose**: Rewrite the question to improve retrieval. Called only when grading found all chunks irrelevant.

```
Input: question
Output: rewritten_question, crag_rewrite_count += 1
```

**Prompt**:
```
The following question about Bank of England policy did not retrieve relevant
documents. Rewrite it to improve retrieval. Focus on the key policy concepts,
specific meeting dates, or document sections being asked about.

Original question: {question}
Rewritten question:
```

After rewriting, route back to `retrieve` with the new question.

### 5. rerank
**Purpose**: Use Cohere rerank API to reorder graded documents by relevance.

```
Input: question, graded_documents
Output: reranked_documents (top-5, reordered)
```

```python
import cohere
co = cohere.ClientV2()

results = co.rerank(
    query=question,
    documents=[doc["text"] for doc in graded_documents],
    model="rerank-v3.5",
    top_n=5,
)
```

**Fallback**: If Cohere API fails, use local cross-encoder:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([(question, doc["text"]) for doc in graded_documents])
```

### 6. generate
**Purpose**: Claude generates answer grounded in reranked context with citation.

```
Input: question, reranked_documents
Output: answer
```

**Prompt** (domain-specific, much better than baseline's vanilla prompt):
```
You are a specialist analyst answering questions about Bank of England monetary
policy using official BoE documents.

Rules:
1. ONLY use information from the provided source documents
2. Cite the source document for each claim (e.g., "According to the November 2025
   MPC minutes, paragraph 19...")
3. If the documents don't contain enough information to fully answer the question,
   say so explicitly — do not speculate
4. Use precise policy language (e.g., "Bank Rate" not "interest rate",
   "CPI inflation" not just "inflation")
5. When quoting vote splits, give the exact numbers

Source documents:
{context}

Question: {question}

Answer:
```

### 7. check_hallucination
**Purpose**: Verify the generated answer is supported by the retrieved documents.

```
Input: answer, reranked_documents
Output: is_grounded (bool), routing decision
```

**Prompt**:
```
You are a fact-checker verifying whether an answer about Bank of England policy
is fully supported by the source documents.

Source documents:
{context}

Generated answer:
{answer}

Is every factual claim in the answer supported by the source documents?
Score: "yes" (fully supported) or "no" (contains unsupported claims)
```

**Routing logic**:
- If grounded → END
- If not grounded AND hallucination_retry_count == 0 → regenerate with stricter prompt
- If not grounded AND hallucination_retry_count == 1 → return answer with warning flag

---

## Graph Assembly

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(RAGState)

workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)
workflow.add_node("check_hallucination", check_hallucination)

workflow.add_edge(START, "analyze_query")
workflow.add_edge("analyze_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,  # returns "rerank" or "rewrite_query" or "generate"
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "check_hallucination")

workflow.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination_check,  # returns END or "generate"
)

graph = workflow.compile()
```

---

## Callable Interface

```python
def enhanced_pipeline(query: str) -> dict:
    """Run the enhanced CRAG pipeline on a query."""
    result = graph.invoke({"question": query, "crag_rewrite_count": 0, "hallucination_retry_count": 0, "pipeline_trace": []})

    return {
        "answer": result["answer"],
        "sources": [
            {
                "chunk_id": doc["chunk_id"],
                "text": doc["text"][:200] + "...",
                "score": doc.get("rerank_score", doc.get("score")),
                "metadata": doc.get("metadata", {}),
            }
            for doc in result.get("reranked_documents", result.get("graded_documents", []))
        ],
        "metadata": {
            "pipeline": "enhanced",
            "chunks_retrieved": len(result.get("documents", [])),
            "chunks_after_grading": len(result.get("graded_documents", [])),
            "chunks_after_reranking": len(result.get("reranked_documents", [])),
            "crag_rewrites": result["crag_rewrite_count"],
            "hallucination_retries": result["hallucination_retry_count"],
            "is_grounded": result.get("is_grounded", None),
            "metadata_filters_used": result.get("metadata_filters"),
            "model": "claude-sonnet-4-20250514",
            "pipeline_trace": result["pipeline_trace"],
        },
    }
```

---

## Techniques Mapping to Rubric

| Rubric requirement | Our technique | Node |
|-------------------|---------------|------|
| "At least one advanced technique" | **Three**: | |
| Semantic/hierarchical chunking | Section-aware chunks + metadata | 03-CHUNKING + analyze_query |
| Self-refinement / query rewriting | CRAG grading + rewrite loop | grade_documents + rewrite_query |
| Re-ranking model | Cohere rerank-v3.5 | rerank |

---

## Acceptance Criteria

1. `enhanced_pipeline("What was the MPC vote in November 2025?")` returns correct answer with citations
2. CRAG rewrite triggers on adversarial query (e.g., "What structural changes did the BoE note?")
3. Hallucination check runs and returns `is_grounded: True` for a well-grounded answer
4. Metadata filters are applied when query targets a specific document/date
5. Cohere rerank changes the document order (verify pre/post rerank ordering)
6. Return schema matches baseline pipeline schema
7. `pipeline_trace` shows the actual nodes visited
8. Fallback to local cross-encoder works when Cohere API key is missing
