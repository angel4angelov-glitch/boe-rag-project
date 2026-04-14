"""Scraper for Bank of England Financial Stability Reports.

Structurally identical to MPR: section.page-section chapters, div.box-highlight
boxes, base64 chart images, real HTML tables. Same extraction logic applies.
"""

from __future__ import annotations

from boe_rag.scraper.mpr import MPRScraper


class FSRScraper(MPRScraper):
    """Scraper for Financial Stability Report pages.

    FSR markup matches MPR exactly. No method overrides needed — this class
    exists purely for clarity (one class per document type) and to allow
    document-type-specific customisation later without affecting MPR.
    """
