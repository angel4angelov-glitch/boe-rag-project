# BoE Policy Analysis RAG System

Corrective RAG system over Bank of England monetary policy documents (MPRs, FSRs, MPC minutes, speeches). MSc Financial Technology, IB9AU0 Individual Assignment 2, student u2212350.

## Submission deliverables

Per the IB9AU0 brief, this zip contains three main deliverables:

| # | Deliverable | Location in this zip |
|---|-------------|----------------------|
| 1 | **Report** describing design decisions, results, evaluation, and reflection (1500 words) | [`report.pdf`](report.pdf) |
| 2 | **Demo log** with sample inputs, system outputs, and commentary (6 representative queries) | [`demo_log.pdf`](demo_log.pdf) |
| 3 | **Notebooks + source code** for the full pipeline (baseline + enhanced CRAG) | [`notebooks/`](notebooks/) and [`src/boe_rag/`](src/boe_rag/) |

Supporting files: [`ai_disclosure.md`](ai_disclosure.md), [`requirements.txt`](requirements.txt), [`.env.example`](.env.example).

## Structure

```
u2212350.zip
├── report.pdf                       # Deliverable 1: 1500-word report
├── demo_log.pdf                     # Deliverable 2: 6 annotated queries
├── ai_disclosure.md                 # Required AI usage statement
├── README.md                        # This file
├── requirements.txt                 # Pinned Python dependencies
├── pyproject.toml                   # Editable-install metadata
├── .env.example                     # API keys template (no real keys)
│
├── notebooks/                       # Deliverable 3a: narrative notebooks
│   ├── 01_data_ingestion_indexing.ipynb   # Scrape, chunk, embed, store
│   ├── 02_baseline_and_enhanced.ipynb     # Both pipelines (source of demo_log.pdf)
│   └── 03_evaluation.ipynb                # RAGAS comparison + statistics
│
├── src/boe_rag/                     # Deliverable 3b: installable package
│   ├── scraper/                     # BoE HTML scrapers
│   ├── chunking/                    # Section-aware + baseline chunkers
│   ├── indexing/                    # ChromaDB embedding + storage
│   ├── pipelines/                   # BasePipeline, baseline, enhanced (LangGraph)
│   ├── evaluation/                  # RAGAS metrics + CRAG-specific metrics
│   ├── config.py                    # All model names, thresholds, paths
│   └── models.py                    # Frozen dataclasses + StrEnum types
│
├── tests/                           # 262+ tests (pytest)
├── service/                         # FastAPI wrapper (optional)
├── scripts/                         # Build/render helpers
├── figures/                         # Figure 1 (pipeline diagram), Figure 2 (per-category chart)
├── docs/                            # Implementation specs 01-10
│
└── data/
    ├── raw/                         # Scraped BoE source documents
    ├── chunks/                      # Processed chunks (baseline + enhanced)
    ├── evaluation_results/          # RAGAS scores, CRAG metrics, per-category CSV
    └── test_set.csv                 # 25-query evaluation set with ground truth
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # then add ANTHROPIC_API_KEY, OPENAI_API_KEY, COHERE_API_KEY
# run notebooks in order: 01 -> 02 -> 03
```

## Stack

- **Generation**: Claude Sonnet 4 via `langchain-anthropic`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector store**: ChromaDB (`boe_baseline`, `boe_enhanced` collections)
- **Reranking**: Cohere `rerank-v3.5`
- **Orchestration**: LangGraph (`StateGraph` with conditional edges)
- **Evaluation**: RAGAS v0.2+ (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- **Observability (optional)**: LangSmith

## HTTP service (optional)

Wrap the pipeline as a FastAPI service for curl / browser access:

```bash
pip install -e ".[service]"
uvicorn service.main:app --reload
```

Then:
- `POST http://localhost:8000/query` — body `{"question": "...", "pipeline": "enhanced"}`
- `GET http://localhost:8000/docs` — Swagger UI (auto-generated)
- `GET http://localhost:8000/health` / `/ready` — liveness + readiness probes

Optional API-key gate: set `SERVICE_API_KEY=...` in `.env` and pass `X-API-Key: ...` on every request. When the env var is unset, no auth.

Example:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question":"What was the February 2026 MPC vote split?"}'
```

## Observability: LangSmith tracing (optional)

Set the three LangSmith env vars in `.env` to capture every pipeline run and LLM call as a nested trace on [smith.langchain.com](https://smith.langchain.com):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=boe-rag
```

What's captured: `EnhancedPipeline.run` / `BaselinePipeline.run` as a parent span, with every `ChatAnthropic` call, retry attempt, and LangGraph node execution nested beneath. Filter the dashboard by the `pipeline:baseline` / `pipeline:enhanced` tag.

What's NOT captured (raw SDK calls, no LangChain object to wrap): ChromaDB queries, Cohere rerank, OpenAI embeddings. These appear as opaque node boxes in the trace with their state I/O visible.

Tracing is auto-disabled in tests (`tests/conftest.py`) and during RAGAS runs (`scripts/run_ragas.py` top), so the free-tier 5k/month quota is preserved for pipeline debugging.
