# 00 — Master Index

## Work Units

| # | File | What it delivers | Depends on | Est. effort |
|---|------|-----------------|------------|-------------|
| 01 | [01-PROJECT-SETUP.md](01-PROJECT-SETUP.md) | Repo structure, virtualenv, all dependencies pinned, `.env` template | Nothing | Low |
| 02 | [02-DATA-INGESTION.md](02-DATA-INGESTION.md) | Raw HTML/text files in `data/raw/` for every BoE document | 01 | Medium |
| 03 | [03-CHUNKING.md](03-CHUNKING.md) | Section-aware chunks with metadata as JSON in `data/chunks/` | 02 | **High** |
| 04 | [04-INDEXING.md](04-INDEXING.md) | ChromaDB collection populated with embedded chunks | 01, 03 | Low |
| 05 | [05-BASELINE-PIPELINE.md](05-BASELINE-PIPELINE.md) | Naive RAG pipeline callable via `baseline_pipeline(query) → answer` | 04 | Low |
| 06 | [06-ENHANCED-PIPELINE.md](06-ENHANCED-PIPELINE.md) | CRAG pipeline via LangGraph callable via `enhanced_pipeline(query) → answer` | 04 | **High** |
| 07 | [07-EVALUATION.md](07-EVALUATION.md) | RAGAS scores for both pipelines, comparison tables | 05, 06 | Medium |
| 08 | [08-REPORT.md](08-REPORT.md) | 1500-word report PDF/docx | 07 | Medium |
| 09 | [09-DEMO-LOG.md](09-DEMO-LOG.md) | 5-8 annotated query examples | 05, 06 | Low |
| 10 | [10-PACKAGING.md](10-PACKAGING.md) | `student_number.zip` ready for submission | All | Low |

## Dependency Graph

```
01-PROJECT-SETUP
       │
       ▼
02-DATA-INGESTION
       │
       ▼
03-CHUNKING
       │
       ▼
04-INDEXING
      ┌┴┐
      ▼  ▼
  05-BASELINE  06-ENHANCED
      └┬┘        └┬┘
       ▼          ▼
      07-EVALUATION
       │
      ┌┴┐
      ▼  ▼
 08-REPORT  09-DEMO-LOG
      └┬┘     └┬┘
       ▼       ▼
    10-PACKAGING
```

## Critical Path

The bottleneck is **03-CHUNKING**. Everything downstream depends on chunk quality. If chunks are wrong, evaluation deltas will be noise. Budget 40% of dev time here.

Second bottleneck is **06-ENHANCED-PIPELINE**. The LangGraph CRAG flow has the most moving parts (grading, rewriting, reranking, hallucination check). But the skeleton from the Adaptive RAG tutorial gives you ~60% of this.

## Execution Order

**Phase 1 — Foundation (do first, sequentially)**
1. 01-PROJECT-SETUP
2. 02-DATA-INGESTION
3. 03-CHUNKING
4. 04-INDEXING

**Phase 2 — Pipelines (can parallelise)**
5. 05-BASELINE-PIPELINE ← simple, do first as a sanity check
6. 06-ENHANCED-PIPELINE ← the real work

**Phase 3 — Outputs (sequentially, after both pipelines work)**
7. 07-EVALUATION
8. 09-DEMO-LOG (can run in parallel with report writing)
9. 08-REPORT
10. 10-PACKAGING

## Repo Decisions (Updated After Audit)

| Repo | Decision | Reason |
|------|----------|--------|
| LangGraph Adaptive RAG | **Fork as skeleton for 06** | Current APIs, ~60% reusable, swap LLM + vector store |
| vinodkrane/agentic-rag | **Dropped** | No RAGAS evaluation exists in this repo despite plan claiming otherwise |
| Grecil/Corrective-RAG | **Reference only** | Clean CRAG flow for understanding, nothing to copy |
| 0xshre/rag-evaluation | **Dropped** | Every dependency has breaking API changes. RAGAS eval is ~10 lines from docs |
