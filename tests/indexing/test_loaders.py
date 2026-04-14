"""Tests for JSON-to-record loaders used by the indexer."""

from __future__ import annotations

import json
from pathlib import Path

from boe_rag.indexing.chroma_store import load_baseline_chunks, load_enhanced_chunks


def _write_enhanced(dir_: Path, doc_id: str, chunks: list[dict]) -> None:
    payload = {
        "document": doc_id,
        "document_type": "MPC_minutes",
        "date": "2025-11",
        "source_url": "https://example.com",
        "title": "November 2025 MPC Minutes",
        "total_chunks": len(chunks),
        "total_tokens": sum(c["token_count"] for c in chunks),
        "chunks": chunks,
    }
    (dir_ / f"{doc_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _enhanced_chunk(chunk_id: str, speaker: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"body for {chunk_id}",
        "metadata": {
            "document_type": "MPC_minutes",
            "date": "2025-11",
            "section_category": "policy_discussion",
            "speaker": speaker,
            "source_url": "https://example.com",
            "paragraph_range": "1-3",
            "title": "November 2025 MPC Minutes",
        },
        "token_count": 42,
    }


def test_load_enhanced_chunks_flattens_across_files(tmp_path: Path) -> None:
    """Multiple JSON docs → one flat list of records in sorted file order."""
    _write_enhanced(tmp_path, "mpc_2025_11", [_enhanced_chunk("a1"), _enhanced_chunk("a2")])
    _write_enhanced(tmp_path, "mpc_2025_12", [_enhanced_chunk("b1")])

    records = load_enhanced_chunks(tmp_path)
    assert [r["id"] for r in records] == ["a1", "a2", "b1"]
    assert all("text" in r and "metadata" in r for r in records)


def test_load_enhanced_chunks_converts_none_speaker_to_empty_string(tmp_path: Path) -> None:
    """ChromaDB metadata rejects None; the loader must coerce to ''."""
    _write_enhanced(tmp_path, "mpc_test", [_enhanced_chunk("a1", speaker=None)])
    records = load_enhanced_chunks(tmp_path)
    assert records[0]["metadata"]["speaker"] == ""
    # Non-string types are not silently introduced.
    for v in records[0]["metadata"].values():
        assert isinstance(v, (str, int, float, bool)), v


def test_load_enhanced_chunks_preserves_speaker_when_present(tmp_path: Path) -> None:
    _write_enhanced(tmp_path, "mpc_test", [_enhanced_chunk("a1", speaker="Andrew Bailey")])
    records = load_enhanced_chunks(tmp_path)
    assert records[0]["metadata"]["speaker"] == "Andrew Bailey"


def test_load_baseline_chunks_has_no_metadata_key(tmp_path: Path) -> None:
    payload = {
        "document": "mpc_2025_11",
        "total_chunks": 2,
        "total_tokens": 100,
        "chunks": [
            {"chunk_id": "baseline_mpc_2025_11_001", "text": "aaa", "token_count": 50},
            {"chunk_id": "baseline_mpc_2025_11_002", "text": "bbb", "token_count": 50},
        ],
    }
    (tmp_path / "mpc_2025_11.json").write_text(json.dumps(payload), encoding="utf-8")

    records = load_baseline_chunks(tmp_path)
    assert len(records) == 2
    assert records[0]["id"] == "baseline_mpc_2025_11_001"
    assert records[0]["text"] == "aaa"
    assert "metadata" not in records[0], "baseline records must NOT carry metadata"


def test_load_enhanced_chunks_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_enhanced_chunks(tmp_path) == []
    assert load_baseline_chunks(tmp_path) == []
