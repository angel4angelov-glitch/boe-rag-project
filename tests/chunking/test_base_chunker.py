"""Tests for the baseline (naive) fixed-size chunker."""

from __future__ import annotations

from boe_rag.chunking.base_chunker import chunk_document_baseline


def test_baseline_returns_empty_for_empty_text() -> None:
    assert chunk_document_baseline("", "mpc_2025_11") == []


def test_baseline_produces_chunks_with_sequential_ids() -> None:
    """Small text yields 1 chunk with id suffix _001."""
    chunks = chunk_document_baseline("Short document body.", "mpc_2025_11")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "baseline_mpc_2025_11_001"
    assert chunks[0]["text"] == "Short document body."
    assert chunks[0]["token_count"] > 0


def test_baseline_splits_long_text_respecting_token_limit() -> None:
    """A long text (~3000 tokens) should split into roughly six 500-token chunks."""
    paragraph = "Inflation has moderated over the past year and policy-makers continue to monitor pass-through. "
    text = paragraph * 200  # ~3000 tokens
    chunks = chunk_document_baseline(text, "mpr_2025_11")

    assert len(chunks) >= 3, f"Expected multiple chunks, got {len(chunks)}"
    for chunk in chunks:
        # RecursiveCharacterTextSplitter is approximate; allow generous tolerance.
        assert chunk["token_count"] <= 550, chunk
        assert chunk["token_count"] > 0
    # IDs are sequential and zero-padded.
    assert chunks[0]["chunk_id"] == "baseline_mpr_2025_11_001"
    assert chunks[1]["chunk_id"] == "baseline_mpr_2025_11_002"


def test_baseline_chunks_have_no_metadata_keys() -> None:
    """Baseline is deliberately naive — no document_type, no section_category, etc."""
    chunks = chunk_document_baseline("Some text here.", "speech_bailey_2025_02")
    assert set(chunks[0].keys()) == {"chunk_id", "text", "token_count"}


def test_baseline_treats_structural_markers_as_regular_text() -> None:
    """Baseline must NOT interpret ## or [BOX START] as split boundaries — they stay inline."""
    text = "## Heading\n\nBody text.\n\n[BOX START: Box A]\n\nBox body.\n\n[BOX END]"
    chunks = chunk_document_baseline(text, "mpr_test")
    # Whole thing fits in one chunk; markers survived verbatim.
    joined = "".join(c["text"] for c in chunks)
    assert "## Heading" in joined
    assert "[BOX START: Box A]" in joined
    assert "[BOX END]" in joined
