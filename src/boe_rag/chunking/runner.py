"""Chunking runner: read manifest → chunk every doc → write JSON files.

Produces two output trees:
  data/chunks/enhanced/<doc>.json   - section-aware chunks with full metadata
  data/chunks/baseline/<doc>.json   - fixed-size chunks with no metadata

Skips manifest rows whose status != 'ok' (keeps partial runs from cascading).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from boe_rag.chunking.base_chunker import chunk_document_baseline
from boe_rag.chunking.section_chunker import chunk_document
from boe_rag.config import Paths
from boe_rag.models import Chunk, DocumentType

logger = logging.getLogger(__name__)


# Folder per document type — must match scraper output layout.
_DOC_TYPE_DIRS: dict[DocumentType, str] = {
    DocumentType.MPC_MINUTES: "mpc_minutes",
    DocumentType.MPR: "mpr",
    DocumentType.FSR: "fsr",
    DocumentType.SPEECH: "speeches",
}


@dataclass(frozen=True)
class ChunkingSummary:
    documents_processed: int
    enhanced_chunk_count: int
    baseline_chunk_count: int


def chunk_all() -> ChunkingSummary:
    """Chunk every document listed in manifest.csv and write JSON outputs."""
    manifest_path = Paths.DATA_RAW / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at {manifest_path}")

    enhanced_dir = Paths.DATA_CHUNKS / "enhanced"
    baseline_dir = Paths.DATA_CHUNKS / "baseline"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    total_enhanced = 0
    total_baseline = 0
    n_docs = 0

    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                logger.info("Skipping %s (status=%s)", row["filename"], row["status"])
                continue
            try:
                doc_type = DocumentType(row["document_type"])
            except ValueError:
                logger.warning(
                    "Unknown document_type '%s' for %s — skipping",
                    row["document_type"],
                    row["filename"],
                )
                continue

            sub_dir = _DOC_TYPE_DIRS[doc_type]
            text_path = Paths.DATA_RAW / sub_dir / row["filename"]
            if not text_path.exists():
                logger.warning("Missing text file %s — skipping", text_path)
                continue
            text = text_path.read_text(encoding="utf-8")

            doc_id = text_path.stem  # e.g. mpc_2025_11
            enhanced_chunks = chunk_document(
                text=text,
                document_type=doc_type,
                date=row["date"],
                source_url=row["source_url"],
                title=row["title"],
                doc_id=doc_id,
            )
            baseline_chunks = chunk_document_baseline(text, doc_id)

            _write_enhanced_json(
                enhanced_dir / f"{doc_id}.json", doc_id, row, enhanced_chunks
            )
            _write_baseline_json(baseline_dir / f"{doc_id}.json", doc_id, baseline_chunks)

            total_enhanced += len(enhanced_chunks)
            total_baseline += len(baseline_chunks)
            n_docs += 1
            logger.info(
                "%s: %d enhanced chunks, %d baseline chunks",
                doc_id,
                len(enhanced_chunks),
                len(baseline_chunks),
            )

    summary = ChunkingSummary(
        documents_processed=n_docs,
        enhanced_chunk_count=total_enhanced,
        baseline_chunk_count=total_baseline,
    )
    logger.info(
        "Done: %d docs, %d enhanced chunks, %d baseline chunks",
        summary.documents_processed,
        summary.enhanced_chunk_count,
        summary.baseline_chunk_count,
    )
    return summary


def _write_enhanced_json(
    path: Path, doc_id: str, manifest_row: dict, chunks: list[Chunk]
) -> None:
    total_tokens = sum(c.token_count for c in chunks)
    payload = {
        "document": doc_id,
        "document_type": manifest_row["document_type"],
        "date": manifest_row["date"],
        "source_url": manifest_row["source_url"],
        "title": manifest_row["title"],
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "chunks": [_chunk_to_dict(c) for c in chunks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_baseline_json(path: Path, doc_id: str, chunks: list[dict]) -> None:
    payload = {
        "document": doc_id,
        "total_chunks": len(chunks),
        "total_tokens": sum(c["token_count"] for c in chunks),
        "chunks": chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _chunk_to_dict(chunk: Chunk) -> dict:
    """Serialise a Chunk, converting StrEnum fields to their string values."""
    raw = asdict(chunk)
    md = raw["metadata"]
    md["document_type"] = chunk.metadata.document_type.value
    md["section_category"] = chunk.metadata.section_category.value
    return raw
