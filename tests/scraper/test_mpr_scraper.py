"""Tests for MPRScraper — regression tests for P0 bugs.

Bugs covered:
  - P0: duplicate content when <li> wraps <h3>/<p> (heading + paragraphs
        emitted once inside li's concatenated text, and again as their own
        blocks during descendant walk).
  - P0: table titles leaking into the chart marker list (both <h3.img-title>
        wrapped inside a figure and those wrapping a <table>).
"""

from __future__ import annotations

from boe_rag.scraper.mpr import MPRScraper


def _wrap_page(body: str) -> str:
    """Wrap an HTML body fragment in the minimum structure MPRScraper expects."""
    return f"""
    <html><body>
      <main>
        <div id="output">
          {body}
        </div>
      </main>
    </body></html>
    """


def test_li_wrapping_heading_and_paragraphs_does_not_duplicate() -> None:
    """A <li> containing <h3> + <p> must be emitted once, not twice."""
    html = _wrap_page(
        """
        <h2>Annex</h2>
        <ul>
          <li>
            <h3>Symbols and conventions</h3>
            <p>Except where otherwise stated, the source is ONS.</p>
            <p>n.a. = not available.</p>
          </li>
        </ul>
        """
    )
    text, _ = MPRScraper().scrape(html)

    # The symbols-and-conventions heading should appear exactly once.
    assert text.count("Symbols and conventions") == 1, text
    # Paragraph text should appear exactly once.
    assert text.count("Except where otherwise stated, the source is ONS.") == 1, text
    assert text.count("n.a. = not available.") == 1, text


def test_table_titles_not_emitted_as_chart_markers() -> None:
    """h3.img-title whose text starts with 'Table' must NOT appear as [CHART: ...]."""
    html = _wrap_page(
        """
        <h2>Projections</h2>
        <p>Forecast overview.</p>
        <div class="img-block">
          <h3 class="img-title">Chart 1.1: CPI inflation was 3.8%</h3>
        </div>
        <div class="img-block">
          <h3 class="img-title">Table 3.A: Forecast summary</h3>
          <table>
            <tr><th>Year</th><th>CPI</th></tr>
            <tr><td>2025</td><td>3.5</td></tr>
          </table>
        </div>
        """
    )
    text, _ = MPRScraper().scrape(html)

    # Chart marker for Chart 1.1 is expected.
    assert "[CHART: Chart 1.1: CPI inflation was 3.8%]" in text, text
    # Table 3.A must NOT appear as a CHART marker.
    assert "[CHART: Table 3.A" not in text, text
    # Table 3.A should appear exactly once as a TABLE marker.
    assert text.count("[TABLE: Table 3.A: Forecast summary]") == 1, text


def test_non_img_block_tables_are_still_captured() -> None:
    """Tables outside div.img-block (e.g. FSR stress-test annex tables) must also be extracted."""
    html = _wrap_page(
        """
        <h2>Annex 2</h2>
        <p>Stress test results.</p>
        <h3 class="img-title">Table A2.1: Projected CET1 capital ratios</h3>
        <table>
          <tr><th>Bank</th><th>CET1</th></tr>
          <tr><td>Barclays</td><td>12.5</td></tr>
        </table>
        <h3 class="img-title">Table A2.2: Leverage ratios</h3>
        <table>
          <tr><th>Bank</th><th>Leverage</th></tr>
          <tr><td>HSBC</td><td>5.2</td></tr>
        </table>
        """
    )
    text, _ = MPRScraper().scrape(html)

    # Both annex tables captured with correct titles.
    assert "[TABLE: Table A2.1: Projected CET1 capital ratios]" in text, text
    assert "[TABLE: Table A2.2: Leverage ratios]" in text, text
    # Table bodies present (at least the column data).
    assert "Barclays" in text
    assert "HSBC" in text
    # These must NOT leak as chart markers.
    assert "[CHART: Table A2.1" not in text
    assert "[CHART: Table A2.2" not in text


def test_box_rendered_once_with_markers() -> None:
    """Box content must be wrapped with [BOX START: ...] / [BOX END] and not duplicated."""
    html = _wrap_page(
        """
        <h2>Section</h2>
        <p>Body text.</p>
        <div class="box-highlight">
          <h2 class="box__h2">Box A: Test box</h2>
          <p class="text-box-highlight">Summary sentence.</p>
          <p>Detailed body of the box.</p>
        </div>
        """
    )
    text, _ = MPRScraper().scrape(html)

    assert text.count("[BOX START: Box A: Test box]") == 1, text
    assert text.count("[BOX END]") == 1, text
    # Body paragraph only appears inside the box, not duplicated outside.
    assert text.count("Detailed body of the box.") == 1, text
