"""Chunking package: section-aware (enhanced) + fixed-size (baseline) chunkers.

Public interface:
  - count_tokens / assign_category     (metadata.py)
  - chunk_document_baseline            (base_chunker.py)
  - chunk_document / parse_document    (section_chunker.py)
  - validate_chunks / validate_corpus  (validators.py)
  - chunk_all                          (runner.py)
"""

from boe_rag.chunking.base_chunker import chunk_document_baseline
from boe_rag.chunking.metadata import assign_category, count_tokens
from boe_rag.chunking.runner import ChunkingSummary, chunk_all
from boe_rag.chunking.section_chunker import chunk_document, parse_document
from boe_rag.chunking.validators import (
    CheckResult,
    CheckStatus,
    ValidationReport,
    validate_chunks,
    validate_corpus,
)

__all__ = [
    "CheckResult",
    "CheckStatus",
    "ChunkingSummary",
    "ValidationReport",
    "assign_category",
    "chunk_all",
    "chunk_document",
    "chunk_document_baseline",
    "count_tokens",
    "parse_document",
    "validate_chunks",
    "validate_corpus",
]
