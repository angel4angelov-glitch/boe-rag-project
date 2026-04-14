"""Tests for parse_document — the two-pass chunker's first pass.

These tests pin the structural-marker contract defined in spec 02. They use
small hand-built fragments so failures point precisely at the parser.
"""

from __future__ import annotations

from boe_rag.chunking.section_chunker import parse_document


def test_parse_h2_headings_create_sections() -> None:
    text = "## Chapter One\n\nBody of chapter one.\n\n## Chapter Two\n\nBody of chapter two."
    sections = parse_document(text)
    types = [s.section_type for s in sections]
    headings = [s.heading for s in sections]
    assert types == ["h2", "h2"]
    assert headings == ["Chapter One", "Chapter Two"]


def test_parse_h3_nested_under_h2_carries_parent() -> None:
    text = (
        "## Chapter 1: The economy\n\nIntro.\n\n"
        "### 1.1: Inflation\n\nBody.\n\n"
        "### 1.2: Activity\n\nMore body."
    )
    sections = parse_document(text)
    types = [s.section_type for s in sections]
    assert "h2" in types
    assert types.count("h3") == 2
    h3s = [s for s in sections if s.section_type == "h3"]
    # Parent H2 is tracked on nested H3 sections.
    assert all(s.parent_h2 == "Chapter 1: The economy" for s in h3s)
    assert h3s[0].heading == "1.1: Inflation"
    assert h3s[1].heading == "1.2: Activity"


def test_parse_box_region_is_atomic() -> None:
    """BOX START..BOX END must be one section with section_type='box'."""
    text = (
        "## Chapter\n\nIntro.\n\n"
        "[BOX START: Box A - Firms' costs]\n\n"
        "[Summary: condensed summary]\n\n"
        "Box body paragraph 1.\n\n"
        "Box body paragraph 2.\n\n"
        "[BOX END]\n\n"
        "## Next chapter"
    )
    sections = parse_document(text)
    boxes = [s for s in sections if s.section_type == "box"]
    assert len(boxes) == 1
    box = boxes[0]
    assert box.heading == "Box A - Firms' costs"
    body_text = "\n".join(box.lines)
    assert "Box body paragraph 1." in body_text
    assert "Box body paragraph 2." in body_text
    # BOX_END marker line is captured inside the box region.
    assert "[BOX END]" in body_text


def test_parse_vote_header_creates_vote_section() -> None:
    text = "### The immediate policy decision\n\n**Votes to maintain Bank Rate at 4%**\n\nProposition body."
    sections = parse_document(text)
    votes = [s for s in sections if s.section_type == "vote"]
    assert len(votes) == 1
    assert "Votes to maintain" in votes[0].lines[0]


def test_parse_member_statement_creates_member_section_with_speaker() -> None:
    text = "**Andrew Bailey:** My rationale is that inflation has moderated further."
    sections = parse_document(text)
    members = [s for s in sections if s.section_type == "member"]
    assert len(members) == 1
    assert members[0].speaker == "Andrew Bailey"
    # Full text preserved including name + rationale.
    joined = "\n".join(members[0].lines)
    assert "Andrew Bailey" in joined
    assert "inflation has moderated" in joined


def test_parse_numbered_paragraphs_tracked_on_section() -> None:
    text = (
        "### The Committee's discussions\n\n"
        "1: First paragraph about inflation.\n\n"
        "2: Second paragraph about labour market.\n\n"
        "3: Third paragraph on policy stance."
    )
    sections = parse_document(text)
    # Find the section carrying paragraph numbers (either the h3 section or the trailing text).
    nums: list[int] = []
    for s in sections:
        nums.extend(s.paragraph_numbers)
    assert nums == [1, 2, 3]


def test_parse_period_colon_para_format_both_supported() -> None:
    """Regex must accept both '1: text' and '1. text' per spec risk-matrix."""
    text = "1: Colon form paragraph.\n\n2. Period form paragraph."
    sections = parse_document(text)
    nums: list[int] = []
    for s in sections:
        nums.extend(s.paragraph_numbers)
    assert nums == [1, 2]


def test_parse_empty_text_returns_empty() -> None:
    assert parse_document("") == []
    assert parse_document("\n\n\n") == []


def test_parse_vote_header_inside_voting_section() -> None:
    """H3 'immediate policy decision' then vote header then member statements."""
    text = (
        "### The immediate policy decision\n\n"
        "Members considered the options.\n\n"
        "**Votes to maintain Bank Rate at 4%**\n\n"
        "Five members.\n\n"
        "**Andrew Bailey:** Rationale one.\n\n"
        "**Clare Lombardelli:** Rationale two."
    )
    sections = parse_document(text)
    types = [s.section_type for s in sections]
    assert "h3" in types
    assert "vote" in types
    # Two member sections, both with distinct speakers.
    members = [s for s in sections if s.section_type == "member"]
    speakers = [s.speaker for s in members]
    assert speakers == ["Andrew Bailey", "Clare Lombardelli"]
