"""Indexing package: OpenAI embeddings + ChromaDB persistent storage."""

from boe_rag.indexing.chroma_store import (
    build_index,
    get_collection,
    index_collection,
    load_baseline_chunks,
    load_enhanced_chunks,
)

__all__ = [
    "build_index",
    "get_collection",
    "index_collection",
    "load_baseline_chunks",
    "load_enhanced_chunks",
]
