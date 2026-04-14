"""Base scraper for Bank of England publication pages.

Shared extraction pipeline: fetch → find content → extract chart/table markers
→ strip containers → walk content tree → normalise unicode. Subclasses override
the steps that differ per document type.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

from boe_rag.config import SCRAPE_DELAY_SECONDS, SCRAPE_TIMEOUT, SCRAPE_USER_AGENT

logger = logging.getLogger(__name__)


class ScraperError(Exception):
    """Raised when the page structure doesn't match expectations."""


def url_to_cache_name(url: str) -> str:
    """Build a human-readable cache filename from the last 3 URL segments.

    Args:
        url (str): Full URL, e.g. https://www.bankofengland.co.uk/a/b/c.

    Returns:
        (str) Cache filename, e.g. "a_b_c.html".
    """
    parts = url.rstrip("/").split("/")
    return "_".join(parts[-3:]) + ".html"


def fetch_page(url: str, cache_dir: Path) -> str | None:
    """Fetch a page with caching, rate limiting, and graceful error handling.

    Args:
        url (str): Target URL.
        cache_dir (Path): Directory for raw HTML cache files.

    Returns:
        (str | None) HTML on success, None on HTTP error, timeout, or connection failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / url_to_cache_name(url)
    if cache_path.exists():
        logger.debug("Cache hit: %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    time.sleep(SCRAPE_DELAY_SECONDS)
    headers = {"User-Agent": SCRAPE_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.warning("HTTP %s for %s — skipping", e.response.status_code, url)
        return None
    except (requests.ConnectionError, requests.Timeout):
        logger.warning("Connection/timeout for %s — retrying once", url)
        time.sleep(5)
        try:
            resp = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
            resp.raise_for_status()
        except Exception:
            logger.error("Retry failed for %s — skipping", url)
            return None

    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text


def normalise_text(text: str) -> str:
    """Normalise whitespace and unicode characters.

    Converts &nbsp; to regular space, curly quotes to straight, en/em-dashes
    to hyphens. Prevents invisible characters from poisoning embeddings.

    Args:
        text (str): Raw text from BeautifulSoup extraction.

    Returns:
        (str) Normalised text.
    """
    replacements = {
        "\xa0": " ",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": " - ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return unicodedata.normalize("NFKC", text)


class BaseScraper(ABC):
    """Base class for all BoE document scrapers.

    Subclasses implement `_walk_content_tree` to produce document-type-specific
    text with structural markers that the chunker parses (see spec 02 interface
    contract).
    """

    def scrape(self, html: str) -> tuple[str, dict]:
        """Extract clean text and page-level metadata from HTML.

        Args:
            html (str): Full page HTML.

        Returns:
            (tuple) (extracted_text, page_metadata).
        """
        soup = BeautifulSoup(html, "lxml")

        # Page-level metadata (e.g. speaker from sidebar) BEFORE narrowing to content.
        page_metadata = self._extract_page_metadata(soup)

        content = self._find_content(soup)

        # Extract chart/table info BEFORE stripping containers — order matters.
        chart_markers = self._extract_chart_titles(content)
        table_markers = self._extract_tables(content)

        # Strip elements inside the content container.
        strip_selectors = [
            "div.img-block",            # Chart containers (base64 images)
            "div.footnotes-container",  # Footnotes
            "nav.nav-chapters",         # Sidebar TOC (JS-populated)
            "div.pdf-form",             # "Convert to PDF" form
        ]
        for selector in strip_selectors:
            for el in content.select(selector):
                el.decompose()

        # Word paste artefacts.
        for a in content.find_all("a", id=lambda x: x and x.startswith("_Hlk")):
            a.decompose()

        raw_text = self._walk_content_tree(content, chart_markers, table_markers)
        return normalise_text(raw_text), page_metadata

    def _find_content(self, soup: BeautifulSoup) -> Tag:
        """Locate the first non-empty div#output — content root for MPR/FSR/Speech.

        Some pages (e.g. August 2025 MPR) have multiple div#output elements
        where the early ones are empty scaffolding. Pick the first one that
        actually contains text.

        MPC minutes do NOT have div#output and must override this method.
        """
        for candidate in soup.select("div#output"):
            if candidate.find(["h2", "p"]):
                return candidate
        raise ScraperError(
            "No non-empty div#output — page may be PDF-only or MPC pages should override"
        )

    def _extract_page_metadata(self, soup: BeautifulSoup) -> dict:
        """Override in subclasses needing metadata outside the content container."""
        return {}

    def _extract_chart_titles(self, content: Tag) -> list[str]:
        """Override in MPR/FSR to collect h3.img-title text before stripping."""
        return []

    def _extract_tables(self, content: Tag) -> list[str]:
        """Override in MPR/FSR to extract tables via pandas.read_html."""
        return []

    @abstractmethod
    def _walk_content_tree(
        self,
        content: Tag,
        charts: list[str],
        tables: list[str],
    ) -> str:
        """Subclass implements document-type-specific text extraction."""
        ...
