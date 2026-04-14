"""Tests for base scraper helpers."""

from __future__ import annotations

from boe_rag.scraper.base import normalise_text


def test_normalise_text_strips_inline_footnote_refs() -> None:
    """'footnote [N]' orphan references must be removed, including with multi-digit N."""
    raw = "As I argued footnote [1] and later footnote [12] , inflation is falling."
    cleaned = normalise_text(raw)
    assert "footnote [" not in cleaned, cleaned
    # Collapsed whitespace left behind should not break sentence flow.
    assert "As I argued" in cleaned
    assert "and later" in cleaned
    assert "inflation is falling." in cleaned


def test_normalise_text_preserves_real_bracketed_content() -> None:
    """Don't over-match: '[BOX START: ...]' and '[CHART: ...]' markers must survive."""
    raw = "[BOX START: Box A] body [BOX END] [CHART: Chart 1.1: CPI] [TABLE: Table 3.A]"
    cleaned = normalise_text(raw)
    assert "[BOX START: Box A]" in cleaned
    assert "[BOX END]" in cleaned
    assert "[CHART: Chart 1.1: CPI]" in cleaned
    assert "[TABLE: Table 3.A]" in cleaned


def test_normalise_text_unicode_replacements() -> None:
    """Non-breaking space, curly quotes, en-dash continue to be normalised."""
    raw = "It\u2019s a \u201ctest\u201d\u00a0with 5\u20136% range."
    cleaned = normalise_text(raw)
    assert "\u2019" not in cleaned
    assert "\u201c" not in cleaned
    assert "\u00a0" not in cleaned
    assert "\u2013" not in cleaned
    assert "It's" in cleaned
    assert '"test"' in cleaned
