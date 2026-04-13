# 04 — Embedding & Indexing

## Objective
Embed all chunks using OpenAI `text-embedding-3-small` and store in ChromaDB with metadata. Two separate collections: `boe_baseline` (no metadata), `boe_enhanced` (full metadata).

## Depends on
- 01-PROJECT-SETUP (chromadb, openai, python-dotenv installed; `config.py` constants)
- 03-CHUNKING (JSON files in `data/chunks/baseline/` and `data/chunks/enhanced/`)

## Deliverables
- [ ] `src/boe_rag/indexing/chroma_store.py` — embedding + ChromaDB storage module
- [ ] ChromaDB collection `boe_baseline` populated
- [ ] ChromaDB collection `boe_enhanced` populated with metadata
- [ ] Index validation: count check, query check, metadata filter check

---

## Embedding Model

**Choice**: OpenAI `text-embedding-3-small`
- 1536 dimensions
- Tokenizer: `cl100k_base` (same as our chunk token counting in 03)
- Cost: ~$0.02 per 1M tokens

**Estimated cost**: ~23 documents, ~600K total tokens across both chunk sets (baseline + enhanced) = ~$0.012. Trivially cheap.

**Why not a finance-specific model**: `text-embedding-3-small` is the safe, well-documented default. Finance-specific embeddings (Fin-E5, FinBERT) are a future improvement noted in the report's Section 5 (Future Improvements), not a requirement.

---

## Architecture: Lazy Initialization

**No module-level side effects.** The ChromaDB client, embedding function, and collection handles are created on first use, not at import time. This prevents:
- Crashes when importing the module before `.env` is loaded
- Unnecessary ChromaDB initialization in tests
- API key lookup before the environment is configured

```python
import json
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

from boe_rag.config import (
    Paths, EMBEDDING_MODEL, BASELINE_COLLECTION,
    ENHANCED_COLLECTION, DISTANCE_METRIC, EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_embedding_fn: OpenAIEmbeddingFunction | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        Paths.CHROMA_DB.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(Paths.CHROMA_DB))
    return _client


def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        load_dotenv()  # Load .env if not already loaded
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Check .env file.")
        _embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL,  # "text-embedding-3-small"
        )
    return _embedding_fn


def get_collection(name: str) -> chromadb.Collection:
    """Get a ChromaDB collection with the configured embedding function.

    Pipelines should call this ONCE at initialization and store the handle.
    Each call triggers get_or_create_collection (I/O operation).

    Args:
        name: Collection name (use config.BASELINE_COLLECTION or
              config.ENHANCED_COLLECTION).
    """
    return _get_client().get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": DISTANCE_METRIC},
    )
```

**Key point**: The `embedding_function` must be passed EVERY time you call `get_or_create_collection` — ChromaDB does not persist the embedding function between sessions. If you open the collection without it, ChromaDB falls back to the default `all-MiniLM-L6-v2` (sentence-transformers) — a completely different model with 384 dimensions instead of 1536.

---

## Loading Chunks from JSON

The indexing module reads chunk JSONs produced by 03. Two different formats:

### Enhanced chunks (from `data/chunks/enhanced/`)
```python
def load_enhanced_chunks(chunks_dir: Path) -> list[dict]:
    """Load all enhanced chunk JSONs and flatten into indexable records."""
    records = []
    for json_path in sorted(chunks_dir.glob("*.json")):
        doc = json.loads(json_path.read_text())
        for chunk in doc["chunks"]:
            records.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "document_type": chunk["metadata"]["document_type"],
                    "date": chunk["metadata"]["date"],
                    "section_category": chunk["metadata"]["section_category"],
                    "speaker": chunk["metadata"]["speaker"] or "",  # ChromaDB: no None
                    "source_url": chunk["metadata"]["source_url"],
                    "paragraph_range": chunk["metadata"]["paragraph_range"],
                    "title": chunk["metadata"]["title"],
                },
            })
    return records
```

### Baseline chunks (from `data/chunks/baseline/`)
```python
def load_baseline_chunks(chunks_dir: Path) -> list[dict]:
    """Load all baseline chunk JSONs."""
    records = []
    for json_path in sorted(chunks_dir.glob("*.json")):
        doc = json.loads(json_path.read_text())
        for chunk in doc["chunks"]:
            records.append({
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                # No metadata
            })
    return records
```

**ChromaDB metadata constraints:**
- Values must be `str`, `int`, `float`, or `bool`. No `None`, no lists, no nested dicts.
- `speaker` field: use `""` (empty string) for non-speech chunks, not `None`.

---

## Indexing Logic

### Upsert in batches with skip-if-populated

```python
def index_collection(
    collection: chromadb.Collection,
    records: list[dict],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    force: bool = False,
) -> None:
    """Upsert records into a ChromaDB collection in batches.

    Skips indexing if the collection already has the expected count
    (avoids re-embedding and wasting OpenAI API credits on re-run).
    Pass force=True to re-index regardless.
    """
    existing = collection.count()
    expected = len(records)

    if existing == expected and not force:
        logger.info(
            "Collection '%s' already has %d chunks — skipping (use force=True to re-index)",
            collection.name, existing,
        )
        return

    # Clear before re-indexing to prevent orphaned chunks.
    # upsert alone would leave old IDs that no longer exist in the new chunk set.
    if existing > 0:
        logger.warning(
            "Collection '%s' has %d chunks, expected %d — clearing and re-indexing",
            collection.name, existing, expected,
        )
        # Delete all existing entries
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        kwargs = {
            "ids": [r["id"] for r in batch],
            "documents": [r["text"] for r in batch],
        }
        if "metadata" in batch[0]:
            kwargs["metadatas"] = [r["metadata"] for r in batch]

        collection.upsert(**kwargs)
        logger.info("Indexed %d/%d chunks into '%s'",
                    min(i + batch_size, total), total, collection.name)
```

**Idempotency strategy**:
- **Count matches, no force** → skip entirely (no API calls, no cost)
- **Count mismatch OR force=True** → delete all existing entries, then upsert fresh. This prevents orphaned chunks from old chunk sets persisting alongside new ones. `upsert` alone doesn't remove IDs that no longer exist in the new data.
- Cost of a full re-index: ~$0.012 for ~600 chunks. The skip-if-populated check avoids this on casual notebook re-runs.

### Full indexing flow (called from Notebook 01)

```python
def build_index(force: bool = False) -> None:
    """Build both ChromaDB collections from chunk JSONs."""
    enhanced_records = load_enhanced_chunks(Paths.DATA_CHUNKS / "enhanced")
    baseline_records = load_baseline_chunks(Paths.DATA_CHUNKS / "baseline")

    logger.info("Loaded — Enhanced: %d chunks, Baseline: %d chunks",
                len(enhanced_records), len(baseline_records))

    enhanced_col = get_collection(ENHANCED_COLLECTION)
    baseline_col = get_collection(BASELINE_COLLECTION)

    index_collection(enhanced_col, enhanced_records, force=force)
    index_collection(baseline_col, baseline_records, force=force)

    logger.info("Index build complete. Enhanced: %d, Baseline: %d",
                enhanced_col.count(), baseline_col.count())
```

---

## What This Module Does NOT Do

- **Retrieval logic** — querying ChromaDB with metadata filters is the responsibility of the pipelines (05-BASELINE-PIPELINE, 06-ENHANCED-PIPELINE). This module only populates the index.
- **Embedding model selection** — that decision is made in `config.py`. This module consumes it.
- **Chunk creation** — that's 03-CHUNKING. This module reads the JSON output.

### Public interface for downstream specs

```python
# Pipelines import this:
from boe_rag.indexing.chroma_store import get_collection
from boe_rag.config import ENHANCED_COLLECTION, BASELINE_COLLECTION

# Call ONCE at pipeline init — store the handle, don't call per query
enhanced = get_collection(ENHANCED_COLLECTION)
baseline = get_collection(BASELINE_COLLECTION)

# Then use the handle for queries:
results = enhanced.query(query_texts=["..."], n_results=10, where={...})
```

---

## Validation Checks

1. **Count check**: `enhanced_collection.count()` == total enhanced chunks from 03 JSON files. `baseline_collection.count()` == total baseline chunks.
2. **Query check**: Run `collection.query(query_texts=["MPC vote November 2025"], n_results=5)` on both collections. Verify non-empty results with non-zero distances.
3. **Metadata filter check**: Query enhanced collection with `where={"section_category": "voting"}`. Verify ALL returned chunks have `section_category == "voting"` in their metadata.
4. **Date filter check**: Query enhanced with `where={"date": "2025-11"}`. Verify all returned chunks have `date == "2025-11"`.
5. **Combined filter check**: Query with `where={"$and": [{"document_type": "MPC_minutes"}, {"section_category": "voting"}]}`. Verify results match both filters.
6. **No duplicates**: `len(set(enhanced_collection.get()["ids"])) == enhanced_collection.count()`
7. **Embedding dimension check**: `len(_get_embedding_fn()(["test"])[0]) == 1536` — confirms `text-embedding-3-small` is active, not the default 384-dimension model.

---

## Acceptance Criteria

1. Both ChromaDB collections exist and are populated
2. `boe_baseline` has chunks with NO metadata
3. `boe_enhanced` has chunks with complete metadata (all fields non-null, `speaker` is `""` not `None`)
4. All 7 validation checks pass
5. ChromaDB persists to disk (`Paths.CHROMA_DB`) — survives kernel restart
6. Indexing is idempotent — re-running `build_index()` skips if already populated, re-indexes with `force=True`
7. Embedding function is explicitly `OpenAIEmbeddingFunction` with `text-embedding-3-small`, NOT ChromaDB's default
8. All config values imported from `boe_rag.config`, not hardcoded
9. No module-level side effects — importing `chroma_store` does NOT trigger client creation or API key lookup
10. `load_dotenv()` called before API key access
