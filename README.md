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
