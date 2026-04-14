"""Tests for chunk_all orchestrator + JSON serialisation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from boe_rag.chunking.runner import chunk_all
from boe_rag.config import Paths


@pytest.fixture
def tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Minimal data/raw/ with one MPC, one speech, one manifest row."""
    raw = tmp_path / "raw"
    chunks = tmp_path / "chunks"
    (raw / "mpc_minutes").mkdir(parents=True)
    (raw / "speeches").mkdir(parents=True)
    (chunks / "baseline").mkdir(parents=True)
    (chunks / "enhanced").mkdir(parents=True)

    mpc_text = (
        "## Monetary Policy Summary\n\nThe MPC voted to maintain Bank Rate.\n\n"
        "### The immediate policy decision\n\n"
        "1: Members reviewed the outlook for inflation.\n\n"
        "**Votes to maintain Bank Rate at 4%**\n\n"
        "Five members voted in favour.\n\n"
        "**Andrew Bailey:** I considered the risks carefully and prefer to hold."
    )
    (raw / "mpc_minutes" / "mpc_2025_11.txt").write_text(mpc_text, encoding="utf-8")

    speech_text = (
        "## Speech\n\nHero summary paragraph on inflation and the outlook.\n\n"
        "### Where have we come from?\n\n"
        "Historical context on the inflation shock.\n\n"
        "### Where next?\n\n"
        "Forward-looking discussion about policy scenarios."
    )
    (raw / "speeches" / "speech_bailey_2025_02.txt").write_text(speech_text, encoding="utf-8")

    with (raw / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "document_type", "date", "title", "source_url", "word_count", "status"]
        )
        writer.writerow(
            [
                "mpc_2025_11.txt",
                "MPC_minutes",
                "2025-11",
                "November 2025 MPC Minutes",
                "https://example.com/mpc",
                "50",
                "ok",
            ]
        )
        writer.writerow(
            [
                "speech_bailey_2025_02.txt",
                "speech",
                "2025-02",
                "Bailey speech",
                "https://example.com/bailey",
                "40",
                "ok",
            ]
        )

    monkeypatch.setattr(Paths, "DATA_RAW", raw)
    monkeypatch.setattr(Paths, "DATA_CHUNKS", chunks)
    return tmp_path


def test_chunk_all_writes_enhanced_and_baseline_json(tiny_corpus: Path) -> None:
    summary = chunk_all()
    assert summary.documents_processed == 2
    assert summary.enhanced_chunk_count > 0
    assert summary.baseline_chunk_count > 0

    enhanced_dir = tiny_corpus / "chunks" / "enhanced"
    baseline_dir = tiny_corpus / "chunks" / "baseline"

    assert (enhanced_dir / "mpc_2025_11.json").exists()
    assert (enhanced_dir / "speech_bailey_2025_02.json").exists()
    assert (baseline_dir / "mpc_2025_11.json").exists()
    assert (baseline_dir / "speech_bailey_2025_02.json").exists()


def test_chunk_all_enhanced_json_schema(tiny_corpus: Path) -> None:
    chunk_all()
    payload = json.loads(
        (tiny_corpus / "chunks" / "enhanced" / "mpc_2025_11.json").read_text()
    )
    assert payload["document"] == "mpc_2025_11"
    assert payload["document_type"] == "MPC_minutes"
    assert payload["date"] == "2025-11"
    assert payload["title"] == "November 2025 MPC Minutes"
    assert payload["source_url"] == "https://example.com/mpc"
    assert payload["total_chunks"] == len(payload["chunks"])

    first = payload["chunks"][0]
    assert set(first.keys()) == {"chunk_id", "text", "metadata", "token_count"}
    md = first["metadata"]
    assert set(md.keys()) == {
        "document_type",
        "date",
        "section_category",
        "speaker",
        "source_url",
        "paragraph_range",
        "title",
    }
    # section_category is serialised as the StrEnum string value.
    assert isinstance(md["section_category"], str)


def test_chunk_all_baseline_json_has_no_metadata(tiny_corpus: Path) -> None:
    chunk_all()
    payload = json.loads(
        (tiny_corpus / "chunks" / "baseline" / "mpc_2025_11.json").read_text()
    )
    assert payload["document"] == "mpc_2025_11"
    for chunk in payload["chunks"]:
        assert set(chunk.keys()) == {"chunk_id", "text", "token_count"}


def test_chunk_all_skips_missing_status_rows(tiny_corpus: Path) -> None:
    """Manifest rows with status != 'ok' are skipped."""
    manifest = tiny_corpus / "raw" / "manifest.csv"
    rows = manifest.read_text().splitlines()
    # Flip the speech row to 'missing'.
    rows[2] = rows[2].replace(",ok", ",missing")
    manifest.write_text("\n".join(rows), encoding="utf-8")

    summary = chunk_all()
    assert summary.documents_processed == 1  # only the MPC row
