"""Scraper for Bank of England speeches.

Target list hardcoded: 10 MPC member speeches from 2025-2026 selected for
policy relevance (forward guidance, dissent rationale, inflation outlook,
consumption/trade/financial-stability arguments that appear in our eval set).

Speaker name is in the sidebar (div.col3), outside the main content container,
so we extract it via _extract_page_metadata before narrowing to content.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from boe_rag.scraper.base import BaseScraper, is_in_ancestor

# Chart caption list items that leak from <ul> blocks adjacent to chart images.
# Example lines: "Source: ONS, Bank of England.", "(a) Outturn CPI data...",
# "Notes: ...", "Sources: BoE, ONS.". These are chart apparatus, not prose.
_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Sources?\s*:", re.IGNORECASE),
    re.compile(r"^\s*Notes?\s*:", re.IGNORECASE),
    re.compile(r"^\s*\([a-z]\)\s"),  # "(a) ", "(b) ", etc.
)


def _is_chart_caption(text: str) -> bool:
    """True if text looks like a chart caption/footnote line, not prose."""
    return any(pat.match(text) for pat in _CAPTION_PATTERNS)


# Selected from bankofengland.co.uk/sitemap/speeches on 2026-04-14.
# Criteria: MPC members, 2025-2026, policy-relevant content.
SPEECH_URLS: list[str] = [
    "https://www.bankofengland.co.uk/speech/2025/january/alan-taylor-speech-at-leeds-university-inflation-dynamics-and-outlook",
    "https://www.bankofengland.co.uk/speech/2025/january/sarah-breeden-speech-at-the-university-of-edinburgh",
    "https://www.bankofengland.co.uk/speech/2025/february/andrew-bailey-keynote-speech-university-of-chicago-booth-school-of-business",
    "https://www.bankofengland.co.uk/speech/2025/february/catherine-l-mann-speech-at-leeds-beckett-university-economic-prospects",
    "https://www.bankofengland.co.uk/speech/2025/february/swati-dhingra-speech-dow-lecture-niesr-trade-fragmentation-and-monetary-policy",
    "https://www.bankofengland.co.uk/speech/2025/may/clare-lombardelli-keynote-speech-at-the-bank-of-england-bank-watchers-conference",
    # Pill May 2025 is PDF-only; swap for Ramsden Jun 2025 (HTML speech on labour market).
    "https://www.bankofengland.co.uk/speech/2025/june/dave-ramsden-speech-and-panellist-at-the-barclays-cepr-monetary-policy-forum-2025",
    # Greene Jun 2025 is PDF-only; use Greene Feb 2025 "Not such an island after all" instead.
    "https://www.bankofengland.co.uk/speech/2025/february/megan-greene-speech-at-the-institute-of-directors",
    "https://www.bankofengland.co.uk/speech/2025/october/alan-taylor-remarks-and-fireside-chat-with-gillian-tett-at-cambridge",
    "https://www.bankofengland.co.uk/speech/2025/october/catherine-l-mann-keynote-speech-at-resolution-foundation",
]


class SpeechScraper(BaseScraper):
    """Scraper for BoE speech detail pages."""

    def _extract_page_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract speaker, title, date from outside the content container.

        Returns:
            (dict) Keys: speaker, title, published_date.
        """
        title_el = soup.select_one("h1[itemprop='name']")
        title = title_el.get_text(" ", strip=True) if title_el else ""

        # Speaker is in the right-hand sidebar.
        speaker_el = soup.select_one("div.col3 a.med-block-cta h3")
        speaker = speaker_el.get_text(" ", strip=True) if speaker_el else ""

        date_el = soup.select_one("div.published-date")
        # Collapse multi-line whitespace to single spaces.
        date = " ".join(date_el.get_text(" ", strip=True).split()) if date_el else ""

        return {"title": title, "speaker": speaker, "published_date": date}

    def _walk_content_tree(
        self,
        content: Tag,
        charts: list[str],
        tables: list[str],
    ) -> str:
        """Emit speech text with ## / ### headings and paragraphs."""
        lines: list[str] = []

        # Hero/summary paragraph lives outside div#output in some layouts.
        hero = content.find_parent("div")
        if hero is not None:
            hero_el = hero.select_one("div.hero-paragraph span.hero")
            if hero_el:
                summary = hero_el.get_text(" ", strip=True)
                if summary:
                    lines.append(summary)

        for el in content.descendants:
            if not isinstance(el, Tag):
                continue

            name = el.name

            if name in ("h1", "h2"):
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"## {text}")
            elif name == "h3":
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"### {text}")
            elif name == "p":
                if is_in_ancestor(el, tag_name="li"):
                    continue
                text = el.get_text(" ", strip=True)
                if text and not _is_chart_caption(text):
                    lines.append(text)
            elif name == "li":
                text = el.get_text(" ", strip=True)
                if text and not _is_chart_caption(text):
                    lines.append(f"- {text}")

        return "\n\n".join(lines)
