"""Tests for chunking validators.

Covers both per-document checks and cross-document checks (duplicate ids,
corpus-level token balance) from spec 03's validation-checks list.
"""

from __future__ import annotations

from boe_rag.chunking.validators import (
    CheckStatus,
    validate_chunks,
    validate_corpus,
)
from boe_rag.models import Chunk, ChunkMetadata, DocumentType, SectionCategory


def _meta(cat: SectionCategory = SectionCategory.POLICY_DISCUSSION) -> ChunkMetadata:
    return ChunkMetadata(
        document_type=DocumentType.MPC_MINUTES,
        date="2025-11",
        section_category=cat,
        speaker=None,
        source_url="https://example.com",
        paragraph_range="1-3",
        title="November 2025 MPC Minutes",
    )


def _chunk(
    chunk_id: str,
    text: str,
    cat: SectionCategory = SectionCategory.POLICY_DISCUSSION,
    tokens: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=_meta(cat),
        token_count=tokens if tokens is not None else len(text.split()),
    )


# ── Per-doc: count + category ────────────────────────────────


def test_chunk_count_within_expected_range_passes() -> None:
    chunks = [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    count_check = next(c for c in report.checks if c.name == "chunk_count")
    assert count_check.status is CheckStatus.PASS, count_check


def test_chunk_count_below_minimum_fails() -> None:
    # MPC expects 15-30; 2 chunks is way too low.
    chunks = [_chunk("a", "x " * 20), _chunk("b", "y " * 20)]
    report = validate_chunks(chunks, original_text="x " * 50, doc_type=DocumentType.MPC_MINUTES)
    count_check = next(c for c in report.checks if c.name == "chunk_count")
    assert count_check.status is CheckStatus.FAIL


def test_mpc_missing_voting_category_fails_when_markers_in_source() -> None:
    """Chunker bug: source has markers but chunks don't have the category."""
    chunks = [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    source_with_markers = (
        "body " * 400 + "\n\n**Votes to maintain Bank Rate at 4%**\n\n"
        "**Andrew Bailey:** My rationale."
    )
    report = validate_chunks(chunks, original_text=source_with_markers, doc_type=DocumentType.MPC_MINUTES)
    missing = next(c for c in report.checks if c.name == "required_categories")
    assert missing.status is CheckStatus.FAIL, missing
    assert "voting" in missing.detail.lower()
    assert "individual_statement" in missing.detail.lower()


def test_mpc_missing_voting_category_is_warn_when_source_has_no_markers() -> None:
    """Prose-only MPC minutes: WARN (data characteristic, not a bug)."""
    chunks = [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    missing = next(c for c in report.checks if c.name == "required_categories")
    assert missing.status is CheckStatus.WARN, missing


def test_mpc_with_required_categories_passes() -> None:
    chunks = [_chunk(f"c_{i:03d}", "body " * 20) for i in range(15)]
    chunks.append(_chunk("vote", "Votes to maintain Bank Rate at 4%", SectionCategory.VOTING, tokens=80))
    chunks.append(_chunk("bailey", "Andrew Bailey reasons", SectionCategory.INDIVIDUAL_STATEMENT, tokens=80))
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    check = next(c for c in report.checks if c.name == "required_categories")
    assert check.status is CheckStatus.PASS


# ── Per-doc: content sanity ─────────────────────────────────


def test_voting_chunk_without_keywords_fails() -> None:
    bad = _chunk("bad_vote", "Some random prose with no relevant markers.", SectionCategory.VOTING)
    chunks = [bad] + [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    keyword_check = next(c for c in report.checks if c.name == "voting_keywords")
    assert keyword_check.status is CheckStatus.FAIL


def test_short_chunk_fails_length_check() -> None:
    """Every content chunk text must be >50 chars (check #10)."""
    chunks = [_chunk("short", "too short", tokens=2)] + [
        _chunk(f"c_{i:03d}", "body " * 20) for i in range(20)
    ]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    check = next(c for c in report.checks if c.name == "min_text_length")
    assert check.status is CheckStatus.FAIL
    assert "short" in check.detail


def test_short_structural_chunk_is_exempt_from_length_check() -> None:
    """VOTING / INDIVIDUAL_STATEMENT chunks carry value in metadata even when short."""
    chunks = [
        _chunk("vote", "**Votes to maintain Bank Rate at 4%**", SectionCategory.VOTING, tokens=10),
    ] + [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPC_MINUTES)
    check = next(c for c in report.checks if c.name == "min_text_length")
    assert check.status is CheckStatus.PASS


def test_oversized_non_box_chunk_fails_size_check() -> None:
    big = _chunk("big", "body " * 1000, SectionCategory.INFLATION, tokens=1500)
    chunks = [big] + [_chunk(f"c_{i:03d}", "body " * 20) for i in range(20)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPR)
    check = next(c for c in report.checks if c.name == "max_size")
    assert check.status is CheckStatus.FAIL


def test_oversized_box_under_hard_cap_is_warn_not_fail() -> None:
    box = _chunk("b", "[BOX START: Box A] body", SectionCategory.BOX_ANALYSIS, tokens=1800)
    chunks = [box] + [_chunk(f"c_{i:03d}", "body " * 20) for i in range(40)]
    report = validate_chunks(chunks, original_text="body " * 500, doc_type=DocumentType.MPR)
    check = next(c for c in report.checks if c.name == "max_size")
    # Box between max and hard cap is allowed.
    assert check.status is CheckStatus.PASS


# ── Corpus-level: uniqueness + balance ──────────────────────


def test_duplicate_chunk_ids_across_corpus_fails() -> None:
    a = [_chunk("dupe", "text one", tokens=50)]
    b = [_chunk("dupe", "text two", tokens=50)]
    report = validate_corpus(enhanced={"doc_a": a, "doc_b": b}, baseline={})
    check = next(c for c in report.checks if c.name == "unique_chunk_ids")
    assert check.status is CheckStatus.FAIL
    assert "dupe" in check.detail
