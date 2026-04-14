"""Scrape runner: fetches target documents, applies the right scraper,
writes output to data/raw/, and maintains data/raw/manifest.csv.

Idempotent at two levels:
  - HTML fetch: skip if cached (handled by fetch_page)
  - Text output: skip if .txt already exists (handled here)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from boe_rag.config import Paths
from boe_rag.models import DocumentType
from boe_rag.scraper.base import BaseScraper, fetch_page
from boe_rag.scraper.fsr import FSRScraper
from boe_rag.scraper.mpc import MPCScraper
from boe_rag.scraper.mpr import MPRScraper
from boe_rag.scraper.speeches import SPEECH_URLS, SpeechScraper

logger = logging.getLogger(__name__)


# ── Target URL lists ─────────────────────────────────────────

MPC_URLS: list[tuple[str, str]] = [
    ("2025-06", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/june-2025"),
    ("2025-08", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/august-2025"),
    ("2025-09", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/september-2025"),
    ("2025-11", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/november-2025"),
    ("2025-12", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/december-2025"),
    ("2026-02", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/february-2026"),
    ("2026-03", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/march-2026"),
]

MPR_URLS: list[tuple[str, str]] = [
    ("2025-02", "https://www.bankofengland.co.uk/monetary-policy-report/2025/february-2025"),
    ("2025-05", "https://www.bankofengland.co.uk/monetary-policy-report/2025/may-2025"),
    ("2025-08", "https://www.bankofengland.co.uk/monetary-policy-report/2025/august-2025"),
    ("2025-11", "https://www.bankofengland.co.uk/monetary-policy-report/2025/november-2025"),
]

FSR_URLS: list[tuple[str, str]] = [
    ("2025-07", "https://www.bankofengland.co.uk/financial-stability-report/2025/july-2025"),
    ("2025-12", "https://www.bankofengland.co.uk/financial-stability-report/2025/december-2025"),
]


# ── Manifest row ────────────────────────────────────────────


@dataclass(frozen=True)
class ManifestRow:
    filename: str
    document_type: str
    date: str
    title: str
    source_url: str
    word_count: int
    status: str  # "ok" | "missing"


# ── Document processing ─────────────────────────────────────


def _process_document(
    scraper: BaseScraper,
    date: str,
    url: str,
    document_type: DocumentType,
    filename: str,
    out_dir: Path,
    default_title: str,
) -> ManifestRow:
    """Fetch, scrape, save, and return manifest metadata for one document."""
    output_path = out_dir / filename
    if output_path.exists():
        text = output_path.read_text(encoding="utf-8")
        word_count = len(text.split())
        logger.info("Skipping %s (already processed)", filename)
        return ManifestRow(
            filename=filename,
            document_type=str(document_type),
            date=date,
            title=default_title,
            source_url=url,
            word_count=word_count,
            status="ok",
        )

    html = fetch_page(url, Paths.HTML_CACHE)
    if html is None:
        logger.warning("Fetch failed for %s", url)
        return ManifestRow(
            filename=filename,
            document_type=str(document_type),
            date=date,
            title=default_title,
            source_url=url,
            word_count=0,
            status="missing",
        )

    text, metadata = scraper.scrape(html)
    title = metadata.get("title") or default_title
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    word_count = len(text.split())
    logger.info("Wrote %s (%d words)", filename, word_count)
    return ManifestRow(
        filename=filename,
        document_type=str(document_type),
        date=date,
        title=title,
        source_url=url,
        word_count=word_count,
        status="ok",
    )


def _month_label(date: str) -> str:
    """YYYY-MM → 'November 2025' for use in default titles."""
    year, month = date.split("-")
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return f"{month_names[int(month) - 1]} {year}"


def _speaker_slug(speaker: str) -> str:
    """'Andrew Bailey' → 'bailey' for filenames."""
    return speaker.split()[-1].lower() if speaker else "unknown"


# ── Entry point ─────────────────────────────────────────────


def scrape_all() -> list[ManifestRow]:
    """Scrape every target document and write data/raw/manifest.csv.

    Returns:
        (list[ManifestRow]) Rows for every attempted document (ok or missing).
    """
    rows: list[ManifestRow] = []

    # MPC minutes
    mpc_scraper = MPCScraper()
    mpc_dir = Paths.DATA_RAW / "mpc_minutes"
    for date, url in MPC_URLS:
        filename = f"mpc_{date.replace('-', '_')}.txt"
        title = f"{_month_label(date)} MPC Minutes"
        rows.append(_process_document(
            mpc_scraper, date, url, DocumentType.MPC_MINUTES, filename, mpc_dir, title,
        ))

    # Monetary Policy Reports
    mpr_scraper = MPRScraper()
    mpr_dir = Paths.DATA_RAW / "mpr"
    for date, url in MPR_URLS:
        filename = f"mpr_{date.replace('-', '_')}.txt"
        title = f"{_month_label(date)} Monetary Policy Report"
        rows.append(_process_document(
            mpr_scraper, date, url, DocumentType.MPR, filename, mpr_dir, title,
        ))

    # Financial Stability Reports
    fsr_scraper = FSRScraper()
    fsr_dir = Paths.DATA_RAW / "fsr"
    for date, url in FSR_URLS:
        filename = f"fsr_{date.replace('-', '_')}.txt"
        title = f"{_month_label(date)} Financial Stability Report"
        rows.append(_process_document(
            fsr_scraper, date, url, DocumentType.FSR, filename, fsr_dir, title,
        ))

    # Speeches — filename built from extracted speaker + URL year/month
    speech_scraper = SpeechScraper()
    speech_dir = Paths.DATA_RAW / "speeches"
    for url in SPEECH_URLS:
        # URL: .../speech/2025/november/alan-taylor-...
        parts = url.rstrip("/").split("/")
        year, month = parts[-3], parts[-2]
        month_num = str([
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ].index(month.lower()) + 1).zfill(2)
        date = f"{year}-{month_num}"

        # Need the speaker name to build the filename — fetch and peek first.
        html = fetch_page(url, Paths.HTML_CACHE)
        if html is None:
            rows.append(ManifestRow(
                filename=f"speech_unknown_{year}_{month_num}.txt",
                document_type=str(DocumentType.SPEECH),
                date=date, title="", source_url=url,
                word_count=0, status="missing",
            ))
            continue

        try:
            text, metadata = speech_scraper.scrape(html)
        except Exception as e:
            logger.warning("Speech scrape failed for %s: %s", url, e)
            slug = parts[-1]
            rows.append(ManifestRow(
                filename=f"speech_unparseable_{year}_{month_num}_{slug[:40]}.txt",
                document_type=str(DocumentType.SPEECH),
                date=date, title="", source_url=url,
                word_count=0, status="missing",
            ))
            continue

        speaker = metadata.get("speaker", "")
        title = metadata.get("title", "")
        filename = f"speech_{_speaker_slug(speaker)}_{year}_{month_num}.txt"
        output_path = speech_dir / filename

        if output_path.exists():
            logger.info("Skipping %s (already processed)", filename)
            word_count = len(output_path.read_text(encoding="utf-8").split())
        else:
            speech_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            word_count = len(text.split())
            logger.info("Wrote %s (%d words, speaker=%s)", filename, word_count, speaker)

        rows.append(ManifestRow(
            filename=filename,
            document_type=str(DocumentType.SPEECH),
            date=date,
            title=title,
            source_url=url,
            word_count=word_count,
            status="ok",
        ))

    _write_manifest(rows, Paths.DATA_RAW / "manifest.csv")
    return rows


def _write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write manifest.csv with one row per document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "document_type", "date", "title", "source_url", "word_count", "status"]
        )
        for row in rows:
            writer.writerow([
                row.filename, row.document_type, row.date, row.title,
                row.source_url, row.word_count, row.status,
            ])
    logger.info("Wrote manifest with %d rows to %s", len(rows), path)
