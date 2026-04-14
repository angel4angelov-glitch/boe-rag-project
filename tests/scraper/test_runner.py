"""Tests for the scrape runner's manifest behaviour on re-runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import Tag

from boe_rag.models import DocumentType
from boe_rag.scraper.base import BaseScraper
from boe_rag.scraper.runner import _process_document


class _StubScraper(BaseScraper):
    """Scraper stub whose scrape() bypasses parsing entirely."""

    def scrape(self, html: str) -> tuple[str, dict]:  # type: ignore[override]
        return "body text " * 5, {"title": "Extracted Title From HTML"}

    def _walk_content_tree(
        self, content: Tag, charts: list[str], tables: list[str]
    ) -> str:
        return ""


@pytest.fixture
def cached_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Seed the HTML cache so fetch_page returns a cached payload without HTTP."""
    from boe_rag.config import Paths
    from boe_rag.scraper.base import url_to_cache_name

    cache_dir = tmp_path / "html_cache"
    cache_dir.mkdir()
    url = "https://www.bankofengland.co.uk/test/2025/november-2025"
    (cache_dir / url_to_cache_name(url)).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(Paths, "HTML_CACHE", cache_dir)
    return tmp_path


def test_manifest_title_on_rerun_reflects_extracted_title(cached_html: Path) -> None:
    """On re-run (output exists), manifest row must use the extracted <h1> title."""
    url = "https://www.bankofengland.co.uk/test/2025/november-2025"
    out_dir = cached_html / "out"
    out_dir.mkdir()
    filename = "mpc_2025_11.txt"
    (out_dir / filename).write_text("pre-existing content", encoding="utf-8")

    row = _process_document(
        scraper=_StubScraper(),
        date="2025-11",
        url=url,
        document_type=DocumentType.MPC_MINUTES,
        filename=filename,
        out_dir=out_dir,
        default_title="Default November 2025 MPC Minutes",
    )

    assert row.title == "Extracted Title From HTML", row
    assert row.status == "ok"
    # Word count still measured from the existing file, not a re-write.
    assert row.word_count == len("pre-existing content".split())


class _SpeechStubScraper(BaseScraper):
    """Returns a speech-style metadata dict (title + speaker)."""

    def scrape(self, html: str) -> tuple[str, dict]:  # type: ignore[override]
        return "speech body", {
            "title": "Test Speech Title",
            "speaker": "Professor Alan Taylor",
        }

    def _walk_content_tree(self, content, charts, tables):
        return ""


def test_manifest_row_includes_normalised_speaker_for_speech(cached_html: Path) -> None:
    """Speech manifest rows carry speaker normalised to 'FirstName LastName'."""
    url = "https://www.bankofengland.co.uk/test/2025/november-2025"
    out_dir = cached_html / "out"
    out_dir.mkdir()

    row = _process_document(
        scraper=_SpeechStubScraper(),
        date="2025-01",
        url=url,
        document_type=DocumentType.SPEECH,
        filename="speech_taylor_2025_01.txt",
        out_dir=out_dir,
        default_title="default",
    )
    assert row.speaker == "Alan Taylor", row
    # Non-speech rows use empty string so manifest.csv has a stable schema.
    assert "speaker" in row.__dataclass_fields__


def test_manifest_row_empty_speaker_for_non_speech(cached_html: Path) -> None:
    url = "https://www.bankofengland.co.uk/test/2025/november-2025"
    out_dir = cached_html / "out"
    out_dir.mkdir()

    row = _process_document(
        scraper=_StubScraper(),
        date="2025-11",
        url=url,
        document_type=DocumentType.MPC_MINUTES,
        filename="mpc_2025_11.txt",
        out_dir=out_dir,
        default_title="November 2025 MPC Minutes",
    )
    assert row.speaker == ""
