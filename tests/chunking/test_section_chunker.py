"""Tests for chunk_document — the full pass-2 pipeline.

Covers categorisation, paragraph_range formatting, box atomicity, oversized
section splitting, small-section merging, and overlap behaviour.
"""

from __future__ import annotations

from boe_rag.chunking.section_chunker import chunk_document
from boe_rag.models import DocumentType, SectionCategory


def _kwargs_mpc() -> dict:
    return {
        "document_type": DocumentType.MPC_MINUTES,
        "date": "2025-11",
        "source_url": "https://www.bankofengland.co.uk/mpc/nov-2025",
        "title": "November 2025 MPC Minutes",
    }


def _kwargs_mpr() -> dict:
    return {
        "document_type": DocumentType.MPR,
        "date": "2025-11",
        "source_url": "https://www.bankofengland.co.uk/mpr/nov-2025",
        "title": "November 2025 Monetary Policy Report",
    }


def _kwargs_speech() -> dict:
    return {
        "document_type": DocumentType.SPEECH,
        "date": "2025-01",
        "source_url": "https://www.bankofengland.co.uk/speech/taylor",
        "title": "The last half mile - speech by Alan Taylor",
    }


# ── MPC: voting + individual statement ─────────────────────


def test_mpc_vote_and_member_sections_get_distinct_categories() -> None:
    text = (
        "### The immediate policy decision\n\n"
        "Members reviewed the balance of risks carefully.\n\n"
        "**Votes to maintain Bank Rate at 4%**\n\n"
        "Five members voted to maintain Bank Rate.\n\n"
        "**Andrew Bailey:** Upside risks to inflation have become less pressing since August. "
        "Labour costs remain elevated. I find the mechanisms underlying upside risks less convincing.\n\n"
        "**Clare Lombardelli:** I worry there may be more underlying inflationary pressure in the economy "
        "than embodied in the central projection. Forward-looking indicators of inflation have been less benign."
    )
    chunks = chunk_document(text, **_kwargs_mpc())

    categories = {c.metadata.section_category for c in chunks}
    assert SectionCategory.VOTING in categories
    assert SectionCategory.INDIVIDUAL_STATEMENT in categories

    # Member chunks carry the speaker name.
    member_chunks = [c for c in chunks if c.metadata.section_category is SectionCategory.INDIVIDUAL_STATEMENT]
    speakers = {c.metadata.speaker for c in member_chunks}
    assert "Andrew Bailey" in speakers
    assert "Clare Lombardelli" in speakers


def test_mpc_paragraph_range_spans_numbered_paragraphs() -> None:
    text = (
        "### The Committee's discussions\n\n"
        "1: Paragraph one " + ("about inflation dynamics and outlook " * 30) + "\n\n"
        "2: Paragraph two " + ("about labour market softening and wages " * 30) + "\n\n"
        "3: Paragraph three " + ("about policy stance and restrictiveness " * 30)
    )
    chunks = chunk_document(text, **_kwargs_mpc())

    assert chunks, "expected at least one chunk"
    # paragraph_range spans all numbered paragraphs in the section.
    discussion = [c for c in chunks if c.metadata.section_category is SectionCategory.POLICY_DISCUSSION]
    assert discussion
    # Range should cover 1-3.
    ranges = [c.metadata.paragraph_range for c in discussion]
    assert any("1" in r and "3" in r for r in ranges), ranges


# ── MPR: box atomicity + chapter/subsection mapping ─────────


def test_mpr_box_never_split_even_when_large() -> None:
    """Box integrity > size limits: a 1500-ish token box stays as ONE chunk.

    Sized to sit between the 1200 max-chunk and 2000 hard-cap thresholds so
    the box-atomicity branch must fire (not the hard-cap fallback).
    """
    # ~110 tokens per paragraph x 12 paragraphs ≈ 1320 tokens — above 1200, below 2000.
    para = " ".join(["Firms report elevated cost pressures and margin rebuilding throughout the period."] * 8)
    box_body = "\n\n".join([para] * 12)
    text = (
        "## 1: The economic outlook\n\n"
        "Intro paragraph.\n\n"
        "[BOX START: Box A - Developments in firms' costs]\n\n"
        f"{box_body}\n\n"
        "[BOX END]\n\n"
        "## Next chapter\n\nBody."
    )
    chunks = chunk_document(text, **_kwargs_mpr())

    box_chunks = [c for c in chunks if c.metadata.section_category is SectionCategory.BOX_ANALYSIS]
    assert len(box_chunks) == 1, f"expected 1 box chunk, got {len(box_chunks)}"
    assert box_chunks[0].token_count > 1200, "box should still be the full size"
    # paragraph_range for a box is its short label.
    assert box_chunks[0].metadata.paragraph_range == "Box A"


def test_mpr_subsection_gets_granular_category() -> None:
    text = (
        "## 1: Current economic conditions\n\n"
        "Intro.\n\n"
        "### 1.1: Inflation\n\n"
        "Inflation has eased to 3.8% in September. Underlying services inflation continues to moderate.\n\n"
        "### 1.2: Activity\n\n"
        "GDP growth remained subdued through Q3. Household consumption has been weak.\n\n"
        "### 1.3: Global and financial conditions\n\n"
        "Global activity has slowed. Market-implied policy paths have shifted lower."
    )
    chunks = chunk_document(text, **_kwargs_mpr())
    cats_to_headings = {c.metadata.section_category: c.text for c in chunks}
    assert SectionCategory.INFLATION in cats_to_headings
    assert SectionCategory.DEMAND_OUTPUT in cats_to_headings
    assert SectionCategory.GLOBAL_ECONOMY in cats_to_headings


def test_mpr_oversized_section_splits_at_paragraph_boundaries_with_overlap() -> None:
    """Sections above 1200 tokens split on blank-line boundaries, with 50-token overlap."""
    paragraph = " ".join(["Inflation dynamics have evolved alongside shifts in global demand and supply."] * 20)
    # ~13 x 160-token paragraphs = ~2080 tokens — forces a split.
    body = "\n\n".join([paragraph] * 13)
    text = "## Outlook\n\n" + body  # Outlook → FORWARD_GUIDANCE

    chunks = chunk_document(text, **_kwargs_mpr())
    fg_chunks = [c for c in chunks if c.metadata.section_category is SectionCategory.FORWARD_GUIDANCE]

    assert len(fg_chunks) >= 2, "oversized section should split into multiple chunks"
    # Every chunk must be under the max.
    for c in fg_chunks:
        assert c.token_count <= 1200, c
    # Adjacent same-section chunks overlap: the tail of chunk N is a prefix of chunk N+1.
    first_tail_words = fg_chunks[0].text.split()[-5:]
    assert " ".join(first_tail_words) in fg_chunks[1].text, (
        "expected 50-token overlap between adjacent pieces"
    )


# ── Speech: sequential paragraph_range ──────────────────────


def test_speech_paragraph_range_is_sequential() -> None:
    text = (
        "## Speech\n\n"
        "Hero summary paragraph.\n\n"
        "### Where have we come from?\n\n"
        "Historical context.\n\n"
        "### Where are we now?\n\n"
        "Current conditions.\n\n"
        "### Where next? The scenarios for policy\n\n"
        "Forward-looking discussion about scenarios."
    )
    chunks = chunk_document(text, **_kwargs_speech())
    ranges = [c.metadata.paragraph_range for c in chunks]
    # Must be strictly increasing integers starting at 1.
    assert ranges == [str(i + 1) for i in range(len(chunks))], ranges

    # All chunks tagged with the right doc type.
    assert all(c.metadata.document_type is DocumentType.SPEECH for c in chunks)

    # Outlook-themed heading triggers FORWARD_GUIDANCE category.
    fg = [c for c in chunks if c.metadata.section_category is SectionCategory.FORWARD_GUIDANCE]
    assert fg, "expected the 'Where next? The scenarios' heading to map to FORWARD_GUIDANCE"


# ── Integrity: ids, metadata completeness ───────────────────


def test_all_chunks_have_unique_ids_and_full_metadata() -> None:
    text = (
        "## Monetary Policy Summary\n\n"
        "At its meeting the MPC voted 5-4 to maintain Bank Rate at 4%.\n\n"
        "### The Committee's discussions\n\n"
        "1: Progress on disinflation continues.\n\n"
        "2: The labour market has softened.\n\n"
        "**Votes to maintain Bank Rate at 4%**\n\n"
        "Five members voted to maintain.\n\n"
        "**Andrew Bailey:** Risks more balanced."
    )
    chunks = chunk_document(text, **_kwargs_mpc())

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique within a document"
    for c in chunks:
        assert c.chunk_id
        assert c.text.strip()
        assert c.token_count > 0
        assert c.metadata.document_type is DocumentType.MPC_MINUTES
        assert c.metadata.date == "2025-11"
        assert c.metadata.title == "November 2025 MPC Minutes"
        assert c.metadata.source_url
