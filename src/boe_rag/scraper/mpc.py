"""Scraper for Bank of England MPC minutes.

Handles the two div.page-content blocks inside div.col9, preserves numbered
paragraphs (N:/N. format), vote groupings (**Votes to ...**), and individual
member statements (**Name:** rationale).
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from boe_rag.scraper.base import BaseScraper, ScraperError


class MPCScraper(BaseScraper):
    """Scraper for MPC minutes pages."""

    def _find_content(self, soup: BeautifulSoup) -> Tag:
        """Wrap the two div.page-content blocks inside <main> into a synthetic root.

        The MPC page has two page-content blocks:
          - Block 1: contains only the summary <h2>
          - Block 2: contains the summary body + the full minutes
        Both are needed; concatenate in document order.
        """
        main = soup.select_one("main")
        if main is None:
            raise ScraperError("No <main> element found on MPC page")

        blocks = main.select("div.page-content")
        if not blocks:
            raise ScraperError("No div.page-content blocks inside <main> on MPC page")

        synthetic = soup.new_tag("div")
        for block in blocks:
            synthetic.append(block.extract())
        return synthetic

    def _walk_content_tree(
        self,
        content: Tag,
        charts: list[str],
        tables: list[str],
    ) -> str:
        """Emit MPC minutes text with ##, ###, N:, **Name:** and **Votes** markers."""
        lines: list[str] = []

        for el in content.descendants:
            if not isinstance(el, Tag):
                continue

            name = el.name

            if name == "h2":
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"## {text}")
            elif name == "h3":
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"### {text}")
            elif name == "p":
                rendered = _render_paragraph(el)
                if rendered:
                    lines.append(rendered)
            elif name == "li":
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"- {text}")

        return "\n\n".join(lines)


def _render_paragraph(p: Tag) -> str:
    """Render a <p> preserving <strong>Name:</strong> and **Votes to ...** markers.

    Returns an empty string for <p> tags nested in already-rendered blocks
    (e.g. <li><p>) to avoid double-emission.
    """
    # Skip <p> nested inside <li> — the li rendering already captures the text.
    parent = p.parent
    if parent is not None and parent.name == "li":
        return ""

    # Vote grouping header: "Votes to maintain Bank Rate at 4%"
    strong = p.find("strong")
    if strong and _is_vote_header(strong.get_text(strip=True)):
        return f"**{strong.get_text(strip=True)}**"

    # Individual member statement: <strong>Name:</strong> rationale
    if strong and strong.get_text(strip=True).endswith(":"):
        name = strong.get_text(strip=True).rstrip(":")
        # Everything after the strong tag
        rest = "".join(
            sibling.get_text(" ", strip=True) if isinstance(sibling, Tag) else str(sibling)
            for sibling in strong.next_siblings
        ).strip()
        return f"**{name}:** {rest}".rstrip()

    return p.get_text(" ", strip=True)


def _is_vote_header(text: str) -> bool:
    """Detect vote grouping headers like 'Votes to maintain Bank Rate at 4%'."""
    lowered = text.lower()
    return lowered.startswith("votes to ") and len(text) < 200
