# BoE Policy Analysis RAG System

Corrective RAG system over Bank of England monetary policy documents (MPRs, FSRs, MPC minutes, speeches). MSc Financial Technology — IB9AU0 Individual Assignment 2.

## Structure

```
boe-rag-project/
├── PLAN.md                          # Full project plan and decisions
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .env.example                     # API keys template
│
├── data/
│   ├── raw/                         # Scraped BoE HTML/PDFs
│   └── processed/                   # Chunked + metadata-tagged documents
│
├── notebooks/
│   ├── 01_data_ingestion_indexing.ipynb    # Scrape, chunk, embed, store
│   ├── 02_baseline_and_enhanced.ipynb     # Both RAG pipelines
│   └── 03_evaluation.ipynb                # RAGAS comparison
│
├── evaluation/
│   ├── test_questions.csv           # 15-20 queries with ground truth
│   └── results/                     # RAGAS output scores
│
└── outputs/
    ├── report.pdf                   # 1500-word report
    └── demo_log.pdf                 # Sample inputs/outputs/commentary
```

## Quick Start

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add API keys
3. Run notebooks in order: 01 → 02 → 03

## Stack

- **Generation:** Claude (Anthropic API)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector store:** ChromaDB
- **Reranking:** Cohere rerank
- **Orchestration:** LangGraph
- **Evaluation:** RAGAS
- **Observability (optional):** LangSmith

## Observability — LangSmith tracing (optional)

Set the three LangSmith env vars in `.env` to capture every pipeline run
and LLM call as a nested trace on [smith.langchain.com](https://smith.langchain.com):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=boe-rag
```

What's captured: `EnhancedPipeline.run` / `BaselinePipeline.run` as a
parent span, with every `ChatAnthropic` call, retry attempt, and
LangGraph node execution nested beneath. Filter the dashboard by the
`pipeline:baseline` / `pipeline:enhanced` tag.

What's NOT captured (raw SDK calls, no LangChain object to wrap):
ChromaDB queries, Cohere rerank, OpenAI embeddings. These appear as
opaque node boxes in the trace with their state I/O visible.

Tracing is auto-disabled in tests (`tests/conftest.py`) and during
RAGAS runs (`scripts/run_ragas.py` top) — the free-tier 5k/month quota
is preserved for pipeline debugging.
