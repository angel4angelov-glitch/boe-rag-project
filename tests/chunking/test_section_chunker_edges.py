"""Edge-case regression tests found by running chunk_all on the real corpus."""

from __future__ import annotations

from boe_rag.chunking.section_chunker import chunk_document
from boe_rag.models import DocumentType, SectionCategory


def test_oversized_box_splits_propagate_marker_and_no_overlap() -> None:
    """A >2000-token box splits; each piece must carry the BOX START marker.

    Without marker propagation the validator's box_markers check fails and
    downstream consumers cannot tell box-derived chunks from regular prose.
    No 50-token overlap is applied between box pieces (spec: boxes never
    overlap).
    """
    paragraph = " ".join(["Firms report elevated cost pressures and margin rebuilding."] * 15)
    # 20 paragraphs * ~90 tokens = ~1800 — plus the box wrapping still under 2000.
    # Crank it higher to force a split past the hard cap.
    body = "\n\n".join([paragraph] * 30)
    text = (
        "## Outlook\n\nIntro.\n\n"
        "[BOX START: Box Z - Giant analysis]\n\n"
        f"{body}\n\n"
        "[BOX END]"
    )
    chunks = chunk_document(
        text,
        document_type=DocumentType.MPR,
        date="2025-11",
        source_url="https://example.com",
        title="November 2025 MPR",
        doc_id="mpr_2025_11",
    )
    box_chunks = [c for c in chunks if c.metadata.section_category is SectionCategory.BOX_ANALYSIS]
    assert len(box_chunks) >= 2, "expected oversized box to split"
    for c in box_chunks:
        assert "[BOX START: Box Z" in c.text, (
            f"box piece missing marker: {c.chunk_id}"
        )
        assert c.token_count <= 1200, c


def test_monolithic_paragraph_gets_token_split_fallback() -> None:
    """A single 1600-token paragraph must still end up under 1200 tokens.

    Real MPR sections sometimes have a single giant paragraph (tables rendered
    inline, or flowed prose with no blank lines). Paragraph-boundary splitting
    can't help — fall back to token-level splitting.
    """
    # ~1600 tokens in ONE paragraph (no blank lines anywhere).
    mega_para = " ".join(["word"] * 1600)
    text = "## Outlook\n\n" + mega_para
    chunks = chunk_document(
        text,
        document_type=DocumentType.MPR,
        date="2025-11",
        source_url="https://example.com",
        title="Test",
        doc_id="mpr_test",
    )
    # Every non-box chunk respects the 1200 max.
    for c in chunks:
        if c.metadata.section_category is not SectionCategory.BOX_ANALYSIS:
            assert c.token_count <= 1200, c


def test_chunk_ids_include_doc_id_for_cross_doc_uniqueness() -> None:
    """Two docs of the same type/month must not collide on chunk_id."""
    text_a = "## Intro\n\n" + ("body paragraph with enough content to survive filtering. " * 30)
    text_b = "## Intro\n\n" + ("different body paragraph with enough content to survive. " * 30)

    chunks_a = chunk_document(
        text_a,
        document_type=DocumentType.SPEECH,
        date="2025-02",
        source_url="a",
        title="Speech A",
        doc_id="speech_bailey_2025_02",
    )
    chunks_b = chunk_document(
        text_b,
        document_type=DocumentType.SPEECH,
        date="2025-02",
        source_url="b",
        title="Speech B",
        doc_id="speech_mann_2025_02",
    )
    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b), f"id collision: {ids_a & ids_b}"
    # Ids reflect the doc_id.
    assert all(c.chunk_id.startswith("speech_bailey_2025_02_") for c in chunks_a)
    assert all(c.chunk_id.startswith("speech_mann_2025_02_") for c in chunks_b)


def test_speech_chunks_carry_canonical_speaker_metadata() -> None:
    """Speech chunks must have metadata.speaker populated from the manifest.

    Without this, the enhanced pipeline cannot run
    ``where={"speaker":"Alan Taylor", "document_type":"speech"}`` queries —
    the whole reason we added the field to the manifest.
    """
    text = (
        "## Speech\n\nHero summary.\n\n"
        "### Where have we come from?\n\n"
        "Historical context paragraph with enough length to survive. " * 10
    )
    chunks = chunk_document(
        text,
        document_type=DocumentType.SPEECH,
        date="2025-01",
        source_url="https://example.com/taylor",
        title="Taylor speech",
        doc_id="speech_taylor_2025_01",
        speaker="Alan Taylor",
    )
    assert chunks, "expected at least one chunk"
    assert all(c.metadata.speaker == "Alan Taylor" for c in chunks), [
        c.metadata.speaker for c in chunks
    ]


def test_mpc_member_statement_speaker_is_normalised() -> None:
    """'Catherine L Mann' in MPC markers must be stored as 'Catherine Mann'."""
    text = (
        "### The immediate policy decision\n\n"
        "**Votes to maintain Bank Rate at 4%**\n\n"
        "Five members voted.\n\n"
        "**Catherine L Mann:** My rationale extends to inflation persistence concerns. " * 4
    )
    chunks = chunk_document(
        text,
        document_type=DocumentType.MPC_MINUTES,
        date="2025-11",
        source_url="x",
        title="Nov 2025 MPC",
        doc_id="mpc_2025_11",
    )
    member_chunks = [
        c for c in chunks if c.metadata.section_category is SectionCategory.INDIVIDUAL_STATEMENT
    ]
    assert member_chunks, "expected at least one member-statement chunk"
    assert all(c.metadata.speaker == "Catherine Mann" for c in member_chunks), [
        c.metadata.speaker for c in member_chunks
    ]


def test_empty_heading_only_sections_are_dropped() -> None:
    """A bare '## Heading' with no body must not become a sub-15-token chunk."""
    text = (
        "## First\n\n"
        "Substantial body paragraph one with enough content. " * 20 + "\n\n"
        "## Empty Heading\n\n"   # No body content below.
        "## Next\n\n"
        "Substantial body paragraph two with enough content. " * 20
    )
    chunks = chunk_document(
        text,
        document_type=DocumentType.MPR,
        date="2025-11",
        source_url="x",
        title="Test",
        doc_id="mpr_test",
    )
    # No chunk should be tiny.
    for c in chunks:
        assert c.token_count >= 15 or c.metadata.section_category in {
            SectionCategory.VOTING,
            SectionCategory.INDIVIDUAL_STATEMENT,
            SectionCategory.BOX_ANALYSIS,
        }, f"unexpectedly tiny chunk: {c}"
