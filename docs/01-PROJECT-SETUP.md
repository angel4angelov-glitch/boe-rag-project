# 01 — Project Setup

## Objective
Production-grade Python project: proper package structure, pinned dependencies, type-safe configuration, structured logging, clean separation between library code and notebook demonstration layer.

## Depends on
Nothing.

## Deliverables
- [ ] Full directory tree created
- [ ] `pyproject.toml` with project metadata + pinned dependencies
- [ ] `requirements.txt` exported from pyproject.toml (for marker convenience)
- [ ] `.env.example` with all required API keys
- [ ] `.gitignore` comprehensive
- [ ] `src/boe_rag/` installable as editable package (`pip install -e .`)
- [ ] Assignment coversheet template ready
- [ ] Virtual environment boots and all imports succeed

---

## Design Principles

1. **Library code in `src/boe_rag/`** — well-structured Python package with proper modules, type hints, dataclasses, and clean interfaces. This is the engineering artefact.
2. **Notebooks as demonstration layer** — notebooks import from `boe_rag`, orchestrate calls, display results, and tell the analytical story. They contain minimal logic — just enough to show the pipeline working.
3. **Separation of concerns** — scraping, chunking, indexing, retrieval, generation, and evaluation are independent modules with clear interfaces. Each can be tested and used independently.
4. **Type safety** — enums for constrained string fields, frozen dataclasses for domain objects, type hints on all public functions. Catch bugs at definition time, not at evaluation time.
5. **Configuration as code** — all model names, collection names, chunk sizes, and thresholds in a single config module. No magic strings scattered through the codebase.
6. **Structured logging** — every module uses `logging.getLogger(__name__)`. No bare `print()` in library code. Notebooks configure the log level.
7. **Reproducibility** — exact dependency pins, seed-controlled randomness, saved notebook outputs.

---

## Directory Structure

```
boe-rag-project/
│
├── src/
│   └── boe_rag/
│       ├── __init__.py                 # Package version, top-level exports
│       ├── config.py                   # Configuration: models, thresholds, paths
│       ├── models.py                   # Domain types (enums) + dataclasses: Chunk, PipelineResult
│       │
│       ├── scraper/
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract base scraper with shared logic
│       │   ├── mpc.py                  # MPC minutes scraper
│       │   ├── mpr.py                  # Monetary Policy Report scraper
│       │   ├── fsr.py                  # Financial Stability Report scraper
│       │   └── speeches.py             # Speeches scraper
│       │
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base_chunker.py         # Fixed-size naive chunker (baseline)
│       │   ├── section_chunker.py      # Section-aware chunker (enhanced)
│       │   ├── metadata.py             # Metadata tagging logic
│       │   └── validators.py           # Chunk QA: distribution, content, integrity checks
│       │
│       ├── indexing/
│       │   ├── __init__.py
│       │   └── chroma_store.py         # ChromaDB: create, populate, query collections
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract pipeline interface (shared return schema)
│       │   ├── baseline.py             # Naive RAG: retrieve → generate
│       │   ├── enhanced.py             # CRAG graph: compile + callable interface
│       │   ├── nodes.py                # LangGraph node functions (7 nodes)
│       │   ├── state.py                # LangGraph state TypedDict
│       │   └── prompts.py              # All prompt templates (grading, generation, etc.)
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── ragas_eval.py           # RAGAS metric computation
│           └── metrics.py              # CRAG-specific metrics (rewrite rate, grounding rate)
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb         # Scrape → chunk → validate → embed → index
│   ├── 02_pipelines.ipynb              # Run baseline + enhanced, side-by-side comparison
│   └── 03_evaluation.ipynb             # RAGAS scores, comparison tables, analysis
│
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py                 # Chunk output validation
│   ├── test_metadata.py                # Metadata schema + enum correctness
│   ├── test_pipelines.py               # Pipeline return schema, edge cases
│   └── conftest.py                     # Shared fixtures
│
├── data/
│   ├── html_cache/                     # Raw HTML responses (gitignored, recreatable)
│   ├── raw/                            # Processed text with structural markers
│   │   ├── mpc_minutes/
│   │   ├── mpr/
│   │   ├── fsr/
│   │   └── speeches/
│   ├── chunks/                         # Processed chunks as JSON
│   │   ├── baseline/
│   │   └── enhanced/
│   ├── evaluation_results/             # RAGAS scores, comparison CSVs, per-query results
│   └── test_set.csv                    # Evaluation questions + ground truth
│
├── chroma_db/                          # ChromaDB persistent storage (gitignored)
│
├── coversheet.pdf                      # Warwick assignment coversheet (REQUIRED)
├── report.pdf                          # 1500-word report (includes AI disclosure section)
├── demo_log.pdf                        # 5-8 annotated examples
│
├── pyproject.toml                      # Project metadata + dependencies
├── requirements.txt                    # Exported pins (marker convenience)
├── .env.example                        # API key template
├── .gitignore
└── README.md                           # Setup instructions, architecture overview
```

---

## Domain Types + Data Models (`models.py`)

Enums and dataclasses together — domain types are foundational, everything else imports from here.

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


# ── Domain enums ──────────────────────────────────────────────
# StrEnum: serialises to string for ChromaDB metadata,
# but catches typos at definition time (not at query time).

class DocumentType(StrEnum):
    MPR         = "MPR"
    FSR         = "FSR"
    MPC_MINUTES = "MPC_minutes"
    SPEECH      = "speech"


class SectionCategory(StrEnum):
    GLOBAL_ECONOMY       = "global_economy"
    INFLATION            = "inflation"
    LABOUR_MARKET        = "labour_market"
    DEMAND_OUTPUT        = "demand_output"
    POLICY_DISCUSSION    = "policy_discussion"
    VOTING               = "voting"
    INDIVIDUAL_STATEMENT = "individual_statement"
    BOX_ANALYSIS         = "box_analysis"
    RISK_ASSESSMENT      = "risk_assessment"
    FINANCIAL_STABILITY  = "financial_stability"
    FORWARD_GUIDANCE     = "forward_guidance"
    SPEECH_MAIN          = "speech_main"


# ── Dataclasses ───────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkMetadata:
    document_type: DocumentType
    date: str                           # "2025-11" (YYYY-MM)
    section_category: SectionCategory
    speaker: str | None                 # for speeches / individual statements
    source_url: str
    paragraph_range: str                # "15-18"
    title: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int


@dataclass(frozen=True)
class RetrievedDocument:
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata | None      # None for baseline (no metadata stored)


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    sources: Sequence[RetrievedDocument]
    pipeline_name: str                  # "baseline" | "enhanced"
    chunks_retrieved: int
    chunks_used: int
    model: str
    crag_rewrites: int                  # 0 for baseline
    hallucination_retries: int          # 0 for baseline
    is_grounded: bool | None            # None for baseline
    metadata_filters_used: dict | None
    pipeline_trace: Sequence[str]
```

Both pipelines return `PipelineResult`. Evaluation code never needs to know which pipeline produced the result.

---

## Configuration Module (`config.py`)

Pure configuration — no domain types. Imports enums from `models.py` only if needed for defaults.

```python
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
# Requires editable install (pip install -e .).
# This project is not distributed as a wheel — it's a portfolio/assignment codebase.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_path(relative: str) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / relative


class Paths:
    HTML_CACHE     = get_path("data/html_cache")
    DATA_RAW       = get_path("data/raw")
    DATA_CHUNKS    = get_path("data/chunks")
    DATA_EVAL      = get_path("data/evaluation_results")
    CHROMA_DB      = get_path("chroma_db")
    TEST_SET       = get_path("data/test_set.csv")


# ── Model configuration ───────────────────────────────────────

GENERATION_MODEL    = "claude-sonnet-4-20250514"
GRADING_MODEL       = "claude-sonnet-4-20250514"
EMBEDDING_MODEL     = "text-embedding-3-small"
RERANK_MODEL        = "rerank-v3.5"
LLM_TEMPERATURE     = 0.0


# ── Chunking parameters ──────────────────────────────────────

BASELINE_CHUNK_SIZE    = 500
BASELINE_CHUNK_OVERLAP = 0

ENHANCED_MIN_CHUNK     = 100
ENHANCED_MAX_CHUNK     = 1200
ENHANCED_OVERLAP       = 50


# ── Retrieval parameters ─────────────────────────────────────

BASELINE_TOP_K             = 5
ENHANCED_TOP_K             = 10
RERANK_TOP_N               = 5
MAX_CRAG_REWRITES          = 1
MAX_HALLUCINATION_RETRIES  = 1


# ── ChromaDB ─────────────────────────────────────────────────

BASELINE_COLLECTION  = "boe_baseline"
ENHANCED_COLLECTION  = "boe_enhanced"
DISTANCE_METRIC      = "cosine"
EMBEDDING_BATCH_SIZE = 100


# ── Logging ──────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the boe_rag package.

    Call once from notebook first cell or script entry point.
    Every module uses: logger = logging.getLogger(__name__)
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("boe_rag")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False
```

No magic numbers anywhere in the codebase. Every threshold, model name, and path is here. Logging setup lives here too — it's 15 lines, not worth a dedicated module.

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "boe-rag"
version = "0.1.0"
description = "Corrective RAG system for Bank of England policy documents"
requires-python = ">=3.11"

dependencies = [
    # Core LLM + orchestration
    "anthropic==0.42.0",
    "langchain==0.3.14",
    "langchain-anthropic==0.3.6",
    "langchain-openai==0.3.4",
    "langchain-community==0.3.14",
    "langchain-text-splitters==0.3.4",
    "langgraph==0.3.10",

    # Retrieval & storage
    "chromadb==0.5.23",
    "openai==1.58.0",
    "cohere==5.13.0",

    # Evaluation
    "ragas==0.2.8",
    "datasets==3.2.0",

    # Data ingestion
    "beautifulsoup4==4.12.3",
    "requests==2.32.3",
    "lxml==5.3.0",

    # Tokenisation
    "tiktoken==0.8.0",

    # Utilities
    "python-dotenv==1.0.1",
    "pandas==2.2.3",
    "tqdm==4.67.1",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.4",
    "pytest-cov==6.0.0",
    "jupyter==1.1.1",
    "ipykernel==6.29.5",
    "ruff==0.8.6",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

**Version pin strategy**: The version numbers above are targets. On setup day: install everything, run `pip freeze`, replace with actual installed versions. Those become the real pins.

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Exporting requirements.txt for the marker
```bash
pip freeze > requirements.txt
```

---

## Environment Variables

```bash
# .env.example
ANTHROPIC_API_KEY=sk-ant-...       # Claude: generation, grading, hallucination check
OPENAI_API_KEY=sk-...              # OpenAI: text-embedding-3-small embeddings
COHERE_API_KEY=...                 # Cohere: rerank-v3.5
```

### Reproducibility Without API Keys

The marker will not have your API keys. The submission must be fully reviewable without live API access:

1. **Notebook outputs saved** — Kernel → Restart & Run All → Save with outputs visible
2. **Evaluation results persisted** — `data/evaluation_results/` contains all RAGAS scores, comparison tables, and per-query results as CSV/JSON
3. **ChromaDB excluded from zip** (too large) — but Notebook 01 can recreate it from the chunk JSONs in `data/chunks/` if the marker runs with their own keys

---

## Assignment Coversheet

The brief states: **"Inserting a completed assignment coversheet as the first page of your submission."**

- Download the Warwick MSFT assignment coversheet template
- Fill in: student number, module code (IB9AU0), word count, AI usage declaration
- Save as `coversheet.pdf`
- Include at root level in the zip

---

## .gitignore

```
# Large data (recreatable via notebooks)
data/html_cache/
chroma_db/

# Secrets
.env

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

**Shipped in zip** (NOT gitignored):
- `data/raw/` — processed text files, marker can inspect scraper output
- `data/chunks/` — small JSONs, marker can inspect chunk quality
- `data/evaluation_results/` — RAGAS scores and comparison tables
- `data/test_set.csv` — questions + ground truth

---

## Tests

Lightweight test suite to validate core contracts:

- `test_chunker.py` — chunk output conforms to `Chunk` dataclass, metadata uses valid enum values, token counts within configured bounds
- `test_metadata.py` — `DocumentType` and `SectionCategory` enums cover all expected values, chunk_ids are unique across a document set
- `test_pipelines.py` — both pipelines return `PipelineResult`, baseline has `crag_rewrites=0` and `is_grounded=None`, enhanced has `pipeline_trace` populated
- `conftest.py` — shared fixtures: sample chunks, mock ChromaDB collection

Run with: `pytest --cov=boe_rag tests/`

---

## Acceptance Criteria

1. `pip install -e ".[dev]"` succeeds in a clean venv
2. `python -c "from boe_rag.models import DocumentType, SectionCategory, Chunk, PipelineResult"` succeeds
3. `python -c "from boe_rag.config import GENERATION_MODEL, Paths, setup_logging"` succeeds
4. `python -c "import anthropic, langchain, langgraph, chromadb, ragas, cohere"` succeeds
5. All directories in the tree above exist
6. `pyproject.toml` has exact version pins (`==`) and correct `build-backend`
7. `requirements.txt` exported via `pip freeze`
8. `.env.example` lists all 3 API keys
9. `.gitignore` is comprehensive
10. `pytest` discovers test files (even if tests are stubs initially)
11. `ruff check src/` passes with no errors
12. Assignment coversheet template downloaded and ready
