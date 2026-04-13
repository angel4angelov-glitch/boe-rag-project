"""Configuration: models, thresholds, paths, logging.

Single source of truth for all tuneable parameters.
No domain types here — those live in models.py.
"""

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
    HTML_CACHE = get_path("data/html_cache")
    DATA_RAW = get_path("data/raw")
    DATA_CHUNKS = get_path("data/chunks")
    DATA_EVAL = get_path("data/evaluation_results")
    CHROMA_DB = get_path("chroma_db")
    TEST_SET = get_path("data/test_set.csv")


# ── Model configuration ───────────────────────────────────────

GENERATION_MODEL = "claude-sonnet-4-20250514"
GRADING_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "rerank-v3.5"
LLM_TEMPERATURE = 0.0


# ── Chunking parameters ──────────────────────────────────────

BASELINE_CHUNK_SIZE = 500
BASELINE_CHUNK_OVERLAP = 0

ENHANCED_MIN_CHUNK = 100
ENHANCED_MAX_CHUNK = 1200
ENHANCED_OVERLAP = 50


# ── Retrieval parameters ─────────────────────────────────────

BASELINE_TOP_K = 5
ENHANCED_TOP_K = 10
RERANK_TOP_N = 5
MAX_CRAG_REWRITES = 1
MAX_HALLUCINATION_RETRIES = 1


# ── ChromaDB ─────────────────────────────────────────────────

BASELINE_COLLECTION = "boe_baseline"
ENHANCED_COLLECTION = "boe_enhanced"
DISTANCE_METRIC = "cosine"
EMBEDDING_BATCH_SIZE = 100


# ── Scraping ─────────────────────────────────────────────────

SCRAPE_DELAY_SECONDS = 2.0
SCRAPE_USER_AGENT = "BoE-RAG-Academic-Research/1.0 (University of Warwick MSc FinTech)"
SCRAPE_TIMEOUT = 30


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
    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.propagate = False
