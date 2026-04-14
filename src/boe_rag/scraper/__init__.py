"""BoE document scrapers.

Public interface: per-type scraper classes (`MPCScraper`, `MPRScraper`,
`FSRScraper`, `SpeechScraper`) plus the fetch/cache helper used by the
scrape runner.
"""

from boe_rag.scraper.base import BaseScraper, ScraperError, fetch_page, url_to_cache_name
from boe_rag.scraper.fsr import FSRScraper
from boe_rag.scraper.mpc import MPCScraper
from boe_rag.scraper.mpr import MPRScraper
from boe_rag.scraper.runner import scrape_all
from boe_rag.scraper.speeches import SPEECH_URLS, SpeechScraper

__all__ = [
    "BaseScraper",
    "FSRScraper",
    "MPCScraper",
    "MPRScraper",
    "SPEECH_URLS",
    "ScraperError",
    "SpeechScraper",
    "fetch_page",
    "scrape_all",
    "url_to_cache_name",
]
