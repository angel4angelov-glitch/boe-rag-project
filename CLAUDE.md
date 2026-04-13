# BoE RAG Project

## What This Is
Corrective RAG system over Bank of England policy documents (MPRs, FSRs, MPC minutes, speeches). MSc FinTech assignment (IB9AU0), due 16 April 2026. Two pipelines (baseline naive vs enhanced CRAG), evaluated with RAGAS.

## Tech Stack
- **Python 3.11+**, installed as editable package: `pip install -e ".[dev]"`
- **LLM**: Claude via `langchain-anthropic` (`ChatAnthropic`)
- **Embeddings**: OpenAI `text-embedding-3-small` via ChromaDB's `OpenAIEmbeddingFunction`
- **Vector store**: ChromaDB (`PersistentClient`, two collections: `boe_baseline`, `boe_enhanced`)
- **Reranking**: Cohere `rerank-v3.5`
- **Orchestration**: LangGraph (`StateGraph`, conditional edges, compiled graph)
- **Evaluation**: RAGAS v0.2+ (`Faithfulness`, `ResponseRelevancy`, `LLMContextPrecisionWithoutReference`, `LLMContextRecall`)
- **Scraping**: `requests` + `beautifulsoup4` + `lxml`
- **Token counting**: `tiktoken` `cl100k_base`

## Project Structure
```
src/boe_rag/           # Installable Python package
  config.py            # ALL constants: model names, thresholds, paths, setup_logging()
  models.py            # Domain types (StrEnum) + frozen dataclasses: Chunk, ChunkMetadata, PipelineResult, RetrievedDocument
  scraper/             # BoE HTML scrapers (base + MPC/MPR/FSR/speech)
  chunking/            # Section-aware chunker + baseline naive chunker
  indexing/            # ChromaDB embedding + storage
  pipelines/           # BasePipeline ABC, baseline, enhanced CRAG (LangGraph), prompts
  evaluation/          # RAGAS metrics + CRAG-specific metrics
notebooks/             # 3 Jupyter notebooks (ingestion, pipelines, evaluation)
docs/                  # Detailed specs (01-10) — read these for implementation guidance
data/                  # raw/, chunks/, evaluation_results/, html_cache/, test_set.csv
```

## Coding Conventions
- **Immutable data**: All domain objects use `@dataclass(frozen=True)`. Never mutate.
- **Type safety**: `DocumentType` and `SectionCategory` are `StrEnum` in `models.py`. Use enum members, never raw strings.
- **No module-level side effects**: Importing a module must NOT trigger API calls, client creation, or file I/O. Use lazy init in `__init__` or singleton functions.
- **Config from one place**: All model names, thresholds, paths, collection names live in `config.py`. Import constants, never hardcode.
- **Logging**: `logger = logging.getLogger(__name__)` in every module. No `print()` in library code. Notebooks call `setup_logging()` from config.
- **Error handling**: `load_dotenv()` before any API key access. Validate keys exist with clear error messages.

## Key Architecture Decisions
- **Two ChromaDB collections**: `boe_baseline` (no metadata) and `boe_enhanced` (full metadata). Same embedding model for both.
- **Embedding function**: Must pass `OpenAIEmbeddingFunction` to `get_or_create_collection` every time — ChromaDB doesn't persist it.
- **Both pipelines return `PipelineResult`**: Evaluation code is pipeline-agnostic.
- **Scraper output format**: Plain text with structural markers (`##`, `###`, `N:`, `**Name:**`, `[BOX START]`, etc.) — this is the API between scraper and chunker. See spec 02 interface contract table.
- **ChromaDB returns distances, not similarities**: Convert with `score = 1 - distance`.

## API Keys Required (.env)
```
ANTHROPIC_API_KEY=   # Claude generation + grading
OPENAI_API_KEY=      # text-embedding-3-small
COHERE_API_KEY=      # rerank-v3.5
```

## Specs
Detailed implementation specs are in `docs/01-*.md` through `docs/10-*.md`. Read the relevant spec before implementing each component. Specs 01-05 have been through multiple review passes and are implementation-ready.
