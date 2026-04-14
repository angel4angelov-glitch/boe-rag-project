"""Tests for SpeechScraper — regression tests for caption leak and footnotes."""

from __future__ import annotations

from boe_rag.scraper.speeches import SpeechScraper


def _wrap_speech_page(body: str) -> str:
    """Minimum structure the SpeechScraper expects (sidebar + div#output)."""
    return f"""
    <html><body>
      <h1 itemprop="name">Test Speech Title</h1>
      <div class="col3">
        <a class="med-block-cta"><h3>Jane Doe</h3></a>
      </div>
      <div class="published-date">Published on  12 May 2025</div>
      <div id="output">
        {body}
      </div>
    </body></html>
    """


def test_chart_caption_list_items_are_stripped() -> None:
    """Chart captions ('- Source: ONS.', '- (a) ...') must not leak into text."""
    html = _wrap_speech_page(
        """
        <p>Chart 1 shows inflation.</p>
        <div class="img-block"><img src="data:..." /></div>
        <ul>
          <li>Source: ONS, Bank of England.</li>
          <li>(a) Outturn CPI data at quarterly frequency.</li>
        </ul>
        <p>Next paragraph.</p>
        """
    )
    text, _ = SpeechScraper().scrape(html)

    assert "Source: ONS" not in text, text
    assert "(a) Outturn CPI data" not in text, text
    # Body prose should still be there.
    assert "Chart 1 shows inflation." in text
    assert "Next paragraph." in text


def test_footnote_inline_refs_removed() -> None:
    """Orphaned 'footnote [N]' text (footnotes container already stripped) must be removed."""
    html = _wrap_speech_page(
        """
        <p>As I argued last year footnote [1] inflation is falling.</p>
        <p>See also the August Report footnote [12] .</p>
        <div class="footnotes-container">
          <p>1. Some footnote text.</p>
        </div>
        """
    )
    text, _ = SpeechScraper().scrape(html)

    assert "footnote [1]" not in text, text
    assert "footnote [12]" not in text, text
    # Surrounding prose preserved.
    assert "As I argued last year" in text
    assert "inflation is falling." in text


def test_speaker_extracted_from_sidebar() -> None:
    """Speaker metadata comes from the right-hand sidebar, not the content column."""
    html = _wrap_speech_page("<p>Speech body.</p>")
    _, meta = SpeechScraper().scrape(html)

    assert meta["speaker"] == "Jane Doe"
    assert meta["title"] == "Test Speech Title"
    # Published date whitespace collapsed to single spaces.
    assert meta["published_date"] == "Published on 12 May 2025"
