"""Repro-metadata helpers: git info, compute_test_set_hash, run metadata block."""

from __future__ import annotations

import csv
from pathlib import Path

from boe_rag.evaluation.repro import (
    collect_run_metadata,
    compute_test_set_hash,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class TestTestSetHash:
    def test_returns_sha256_prefixed_hex(self, tmp_path: Path) -> None:
        p = tmp_path / "t.csv"
        _write_csv(p, [{"q": "a", "c": "x"}])
        h = compute_test_set_hash(p)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64

    def test_stable_across_identical_content(self, tmp_path: Path) -> None:
        rows = [{"q": "a", "c": "x"}, {"q": "b", "c": "y"}]
        p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
        _write_csv(p1, rows)
        _write_csv(p2, rows)
        assert compute_test_set_hash(p1) == compute_test_set_hash(p2)

    def test_changes_when_content_changes(self, tmp_path: Path) -> None:
        p = tmp_path / "t.csv"
        _write_csv(p, [{"q": "a", "c": "x"}])
        h1 = compute_test_set_hash(p)
        _write_csv(p, [{"q": "a", "c": "CHANGED"}])
        h2 = compute_test_set_hash(p)
        assert h1 != h2

    def test_insensitive_to_trailing_whitespace(self, tmp_path: Path) -> None:
        """Canonicalisation should ignore benign Excel-ish whitespace edits."""
        p1 = tmp_path / "clean.csv"
        p2 = tmp_path / "trailing.csv"
        p1.write_text("q,c\nhello,world\n", encoding="utf-8")
        p2.write_text("q,c\nhello,world  \n", encoding="utf-8")
        # Strip trailing whitespace on values before hashing — our canonical
        # form dumps each row as JSON with sorted keys after DictReader
        # parses; DictReader preserves trailing whitespace inside fields.
        # This test asserts our canonicaliser strips it.
        assert compute_test_set_hash(p1) == compute_test_set_hash(p2)


class TestCollectRunMetadata:
    def test_contains_required_fields(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ts.csv"
        _write_csv(csv_path, [{"question": "q", "category": "simple_factual",
                               "expected_answer": "a", "source_document": "d",
                               "source_paragraph": "p", "source_quote": "sq"}])
        meta = collect_run_metadata(test_set_path=csv_path)
        required = {
            "timestamp", "ragas_version",
            "generation_model", "grading_model", "embedding_model",
            "rerank_model", "evaluator_model", "evaluator_temperature",
            "test_set_hash", "n_queries",
        }
        assert required.issubset(meta.keys()), f"missing: {required - meta.keys()}"
        assert meta["n_queries"] == 1
        assert meta["test_set_hash"].startswith("sha256:")
        assert meta["evaluator_temperature"] == 0.0

    def test_git_fields_present_even_outside_repo(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ts.csv"
        _write_csv(csv_path, [{"question": "q", "category": "simple_factual",
                               "expected_answer": "a", "source_document": "d",
                               "source_paragraph": "p", "source_quote": "sq"}])
        meta = collect_run_metadata(test_set_path=csv_path)
        # git_sha may be None outside a repo — but the keys must exist so
        # the output JSON has a stable shape.
        assert "git_sha" in meta
        assert "git_dirty" in meta
