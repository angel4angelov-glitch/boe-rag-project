"""Baseline chunker — deliberately naive.

Fixed 500-token windows with zero overlap, zero metadata, zero structural
awareness. The baseline exists to lose: if the enhanced pipeline does not
outperform this, the evaluation delta collapses and the chunking thesis
fails. Keep it simple on purpose.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from boe_rag.chunking.metadata import count_tokens
from boe_rag.config import BASELINE_CHUNK_OVERLAP, BASELINE_CHUNK_SIZE


def chunk_document_baseline(text: str, doc_id: str) -> list[dict]:
    """Split text into fixed-size token chunks with sequential IDs.

    Splits on token boundaries using tiktoken cl100k_base — the same encoding
    the embedding model uses — so the reported token_count matches what the
    model will see. Default separators (``\\n\\n``, ``\\n``, ``" "``, ``""``)
    are used as fallback hierarchy; structural markers (``##``, ``[BOX...]``)
    are treated as literal text.

    Args:
        text: Full document text.
        doc_id: Short document identifier (e.g. 'mpc_2025_11'). Used to build
            sequential ``chunk_id`` values like ``baseline_mpc_2025_11_001``.

    Returns:
        (list[dict]) Each dict has exactly three keys: ``chunk_id``, ``text``,
        ``token_count``. No metadata — the baseline carries none by design.
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(text)
    return [
        {
            "chunk_id": f"baseline_{doc_id}_{i + 1:03d}",
            "text": piece,
            "token_count": count_tokens(piece),
        }
        for i, piece in enumerate(pieces)
    ]
