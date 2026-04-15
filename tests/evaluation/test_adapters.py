"""Test set loader + pipeline-result → SingleTurnSample adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from boe_rag.evaluation.adapters import (
    ABSTAIN_MESSAGE,
    SHOULD_ABSTAIN_IDS,
    is_abstain,
    load_pipeline_results,
    load_test_set,
    results_to_samples,
)


def _write_test_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["question", "category", "expected_answer",
                        "source_document", "source_paragraph", "source_quote"],
        )
        w.writeheader()
        w.writerows(rows)


class TestLoadTestSet:
    def test_yields_ids_q01_q02(self, tmp_path: Path) -> None:
        p = tmp_path / "ts.csv"
        _write_test_csv(p, [
            {"question": "Q1", "category": "simple_factual", "expected_answer": "A1",
             "source_document": "d", "source_paragraph": "p", "source_quote": "sq"},
            {"question": "Q2", "category": "comparative", "expected_answer": "A2",
             "source_document": "d", "source_paragraph": "p", "source_quote": "sq"},
        ])
        ts = load_test_set(p)
        assert list(ts.keys()) == ["q01", "q02"]
        assert ts["q01"]["question"] == "Q1"
        assert ts["q02"]["category"] == "comparative"

    def test_preserves_ground_truth_fields(self, tmp_path: Path) -> None:
        p = tmp_path / "ts.csv"
        _write_test_csv(p, [
            {"question": "Q", "category": "simple_factual", "expected_answer": "EA",
             "source_document": "sd", "source_paragraph": "sp", "source_quote": "SQ"},
        ])
        ts = load_test_set(p)
        row = ts["q01"]
        assert row["expected_answer"] == "EA"
        assert row["source_quote"] == "SQ"


class TestShouldAbstainIds:
    def test_q21_is_included(self) -> None:
        assert "q21" in SHOULD_ABSTAIN_IDS

    def test_is_frozenset_for_immutability(self) -> None:
        assert isinstance(SHOULD_ABSTAIN_IDS, frozenset)


class TestIsAbstain:
    def test_true_for_exact_abstain_message(self) -> None:
        assert is_abstain(ABSTAIN_MESSAGE) is True

    def test_false_for_real_answer(self) -> None:
        assert is_abstain("The MPC voted 7-2 to hold Bank Rate.") is False

    def test_false_for_empty(self) -> None:
        assert is_abstain("") is False

    def test_true_with_leading_trailing_whitespace(self) -> None:
        assert is_abstain("  " + ABSTAIN_MESSAGE + "\n") is True


class TestResultsToSamples:
    def _fixture_results(self) -> dict:
        return {
            "q01": {
                "question": "What was the vote?",
                "category": "simple_factual",
                "answer": "The vote was 7-2.",
                "sources": [
                    {"chunk_id": "c1", "score": 0.9,
                     "text": "Full chunk text explaining the vote split."},
                    {"chunk_id": "c2", "score": 0.8, "text": "Second chunk."},
                ],
                "pipeline_trace": ["retrieve", "generate"],
            }
        }

    def _fixture_test_set(self) -> dict:
        return {
            "q01": {
                "question": "What was the vote?",
                "category": "simple_factual",
                "expected_answer": "Seven to two.",
                "source_document": "mpc", "source_paragraph": "p1",
                "source_quote": "Seven members voted to maintain.",
            }
        }

    def test_builds_single_turn_sample_list(self) -> None:
        samples = results_to_samples(self._fixture_results(), self._fixture_test_set())
        assert len(samples) == 1
        qid, sample = samples[0]
        assert qid == "q01"
        assert sample.user_input == "What was the vote?"
        assert sample.response == "The vote was 7-2."
        assert sample.reference == "Seven to two."
        assert len(sample.retrieved_contexts) == 2
        assert "Full chunk text" in sample.retrieved_contexts[0]

    def test_falls_back_to_text_preview_when_text_missing(self) -> None:
        results = self._fixture_results()
        for src in results["q01"]["sources"]:
            src["text_preview"] = src.pop("text")[:200]
        samples = results_to_samples(results, self._fixture_test_set())
        qid, sample = samples[0]
        assert len(sample.retrieved_contexts) == 2
        assert sample.retrieved_contexts[0]  # non-empty

    def test_abstain_row_yields_empty_contexts(self) -> None:
        results = self._fixture_results()
        results["q01"]["answer"] = ABSTAIN_MESSAGE
        results["q01"]["sources"] = []
        samples = results_to_samples(results, self._fixture_test_set())
        qid, sample = samples[0]
        assert sample.retrieved_contexts == []
        assert sample.response == ABSTAIN_MESSAGE

    def test_skips_missing_query_ids(self) -> None:
        """If results JSON is missing a qid in test set, skip it (don't crash)."""
        results = self._fixture_results()
        ts = self._fixture_test_set()
        ts["q02"] = {**ts["q01"], "question": "Q2"}
        samples = results_to_samples(results, ts)
        assert [qid for qid, _ in samples] == ["q01"]


class TestLoadPipelineResults:
    def test_loads_and_validates_shape(self, tmp_path: Path) -> None:
        import json
        p = tmp_path / "results.json"
        payload = {
            "q01": {
                "question": "Q", "category": "simple_factual",
                "answer": "A", "sources": [],
                "pipeline_trace": [], "pipeline_name": "baseline",
                "chunks_retrieved": 0, "chunks_used": 0,
                "crag_rewrites": 0, "hallucination_retries": 0,
                "is_grounded": None, "metadata_filters_used": None,
            }
        }
        p.write_text(json.dumps(payload))
        loaded = load_pipeline_results(p)
        assert set(loaded.keys()) == {"q01"}
        assert loaded["q01"]["answer"] == "A"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pipeline_results(tmp_path / "nope.json")
