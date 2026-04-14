"""Embedding + ChromaDB storage for the two chunk collections.

Two collections, one embedding model:
  - ``boe_baseline``  - chunks with NO metadata (baseline pipeline)
  - ``boe_enhanced``  - chunks with full ChunkMetadata (enhanced pipeline)

Both collections use the same OpenAI ``text-embedding-3-small`` embedding
function (1536 dims, cl100k_base tokenizer — matches the token counting in
spec 03). ChromaDB does not persist the embedding function between
sessions; it must be passed to ``get_or_create_collection`` every time a
collection is opened, or ChromaDB silently falls back to its default
MiniLM model with different dimensions.

All heavy work is lazy: importing this module performs NO I/O, NO API key
lookup, and NO client construction. The first call to ``get_collection``
(or to the private singletons) triggers initialisation. This keeps tests
fast and avoids failures when importing before ``.env`` is loaded.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from boe_rag.chunking.metadata import count_tokens
from boe_rag.config import (
    BASELINE_COLLECTION,
    DISTANCE_METRIC,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    ENHANCED_COLLECTION,
    Paths,
)

# OpenAI tier-1 tokens-per-minute limit for text-embedding-3-small is 40 000.
# Cap each upsert call below this so a single batch cannot overflow the TPM
# budget, with head-room for the OpenAI server's own token counting which is
# slightly looser than tiktoken for some documents.
_EMBEDDING_TOKEN_BUDGET_PER_BATCH = 30_000

logger = logging.getLogger(__name__)


# ── Lazy singletons ───────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_embedding_fn: OpenAIEmbeddingFunction | None = None


def _get_client() -> chromadb.PersistentClient:
    """Return the persistent ChromaDB client, creating it on first use."""
    global _client
    if _client is None:
        Paths.CHROMA_DB.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(Paths.CHROMA_DB))
    return _client


def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    """Return the OpenAI embedding function, validating the API key on first use."""
    global _embedding_fn
    if _embedding_fn is None:
        load_dotenv()  # Safe no-op if already loaded.
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Create a .env file or export it before "
                "calling any indexing or retrieval function."
            )
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL,
        )
    return _embedding_fn


def get_collection(name: str) -> chromadb.Collection:
    """Return a ChromaDB collection handle bound to the OpenAI embedding function.

    Pipelines must call this ONCE at init and reuse the handle — each call
    performs a ``get_or_create_collection`` I/O round-trip.
    """
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


# ── Chunk loaders ─────────────────────────────────────────────


def load_enhanced_chunks(chunks_dir: Path) -> list[dict]:
    """Load all enhanced chunk JSONs into flat indexable records.

    ChromaDB's metadata API rejects ``None`` values — the ``speaker`` field
    is therefore coerced to an empty string when absent (non-speech chunks).
    All other metadata values are already strings by construction in spec 03.
    """
    records: list[dict] = []
    for json_path in sorted(chunks_dir.glob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        for chunk in doc["chunks"]:
            m = chunk["metadata"]
            records.append(
                {
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": {
                        "document_type": m["document_type"],
                        "date": m["date"],
                        "section_category": m["section_category"],
                        "speaker": m["speaker"] or "",
                        "source_url": m["source_url"],
                        "paragraph_range": m["paragraph_range"],
                        "title": m["title"],
                    },
                }
            )
    return records


def load_baseline_chunks(chunks_dir: Path) -> list[dict]:
    """Load all baseline chunk JSONs. Baseline records carry no metadata."""
    records: list[dict] = []
    for json_path in sorted(chunks_dir.glob("*.json")):
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        for chunk in doc["chunks"]:
            records.append(
                {
                    "id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )
    return records


# ── Indexing ──────────────────────────────────────────────────


def index_collection(
    collection,
    records: list[dict],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    force: bool = False,
) -> None:
    """Upsert records into a ChromaDB collection in batches.

    Idempotency strategy:
      - ``count matches expected`` and ``force=False`` → skip entirely.
      - ``count mismatch`` OR ``force=True`` → delete every existing ID
        first, then upsert fresh. ``upsert`` alone leaves orphaned IDs from
        an older chunk-set — a subtle source of duplicates in retrieval.

    Args:
        collection: A ChromaDB collection (or any object exposing the same
            ``count``/``upsert``/``delete``/``get`` surface — used in tests).
        records: Output of ``load_enhanced_chunks`` / ``load_baseline_chunks``.
            Each dict has ``id``, ``text``, and optionally ``metadata``.
        batch_size: Upsert batch size (keeps a single HTTP request under
            OpenAI's token limit and gives progress telemetry).
        force: Pass ``True`` to re-embed even when the count already matches.
    """
    if not records:
        logger.info("No records to index for '%s' — nothing to do", collection.name)
        return

    existing = collection.count()
    expected = len(records)

    if existing == expected and not force:
        logger.info(
            "Collection '%s' already has %d chunks — skipping (use force=True to re-index)",
            collection.name,
            existing,
        )
        return

    if existing > 0:
        logger.warning(
            "Collection '%s' has %d chunks, expected %d — clearing and re-indexing",
            collection.name,
            existing,
            expected,
        )
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    include_metadata = "metadata" in records[0]

    processed = 0
    for batch in _token_budgeted_batches(
        records,
        max_records=batch_size,
        max_tokens=_EMBEDDING_TOKEN_BUDGET_PER_BATCH,
    ):
        kwargs = {
            "ids": [r["id"] for r in batch],
            "documents": [r["text"] for r in batch],
        }
        if include_metadata:
            kwargs["metadatas"] = [r["metadata"] for r in batch]
        collection.upsert(**kwargs)
        processed += len(batch)
        logger.info(
            "Indexed %d/%d chunks into '%s'",
            processed,
            expected,
            collection.name,
        )


def _token_budgeted_batches(
    records: list[dict], *, max_records: int, max_tokens: int
):
    """Yield batches of records bounded by BOTH a record count and a token budget.

    OpenAI's per-request tokens-per-minute limit (40 000 on tier-1 for
    text-embedding-3-small) can be blown by a single batch containing even
    a handful of long box-analysis chunks. Switching from fixed-count
    batching to a token-budgeted strategy keeps each upsert call under the
    limit regardless of chunk-size distribution.

    Pre-computes per-record token counts once so the cl100k_base encoder
    is not hit inside the hot loop.
    """
    batch: list[dict] = []
    batch_tokens = 0
    for record in records:
        record_tokens = count_tokens(record["text"])
        # A single oversized record still gets its own batch (nothing we can
        # do about it here — chunk splitting upstream already caps at 1200).
        if batch and (
            len(batch) >= max_records or batch_tokens + record_tokens > max_tokens
        ):
            yield batch
            batch = [record]
            batch_tokens = record_tokens
        else:
            batch.append(record)
            batch_tokens += record_tokens
    if batch:
        yield batch


def build_index(force: bool = False) -> dict[str, int]:
    """Build (or refresh) both ChromaDB collections from chunk JSONs.

    Returns a summary of final collection counts:
        {"enhanced": int, "baseline": int}
    """
    enhanced_records = load_enhanced_chunks(Paths.DATA_CHUNKS / "enhanced")
    baseline_records = load_baseline_chunks(Paths.DATA_CHUNKS / "baseline")

    logger.info(
        "Loaded %d enhanced and %d baseline records from %s",
        len(enhanced_records),
        len(baseline_records),
        Paths.DATA_CHUNKS,
    )

    enhanced_col = get_collection(ENHANCED_COLLECTION)
    baseline_col = get_collection(BASELINE_COLLECTION)

    index_collection(enhanced_col, enhanced_records, force=force)
    index_collection(baseline_col, baseline_records, force=force)

    summary = {
        "enhanced": enhanced_col.count(),
        "baseline": baseline_col.count(),
    }
    logger.info("Index build complete: %s", summary)
    return summary
