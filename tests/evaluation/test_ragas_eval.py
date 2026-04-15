"""Checkpoint/resume + abstain-skip behaviour for the RAGAS runner.

Tests stub each metric's ascore() to avoid any real API calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from ragas import SingleTurnSample
from ragas.metrics.result import MetricResult

from boe_rag.evaluation.ragas_eval import (
    CONTEXT_REQUIRED_METRICS,
    load_done_keys,
    run_ragas,
)


# ── Fixtures ────────────────────────────────────────────────


class StubMetric:
    """Minimal fake metric matching the ascore(self, **kwargs) contract."""

    def __init__(self, name: str, required_fields: tuple[str, ...],
                 fixed_value: float = 0.5, raises: Exception | None = None) -> None:
        self.name = name
        self._required = required_fields
        self._value = fixed_value
        self._raises = raises
        self.call_count = 0

    async def ascore(self, **kwargs) -> MetricResult:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        # Mimic RAGAS empty-field validation for context-requiring metrics
        if "retrieved_contexts" in self._required and not kwargs.get("retrieved_contexts"):
            raise ValueError("retrieved_contexts is missing.")
        return MetricResult(value=self._value)


@pytest.fixture
def faith_stub() -> StubMetric:
    return StubMetric(
        "faithfulness",
        required_fields=("user_input", "response", "retrieved_contexts"),
    )


@pytest.fixture
def rel_stub() -> StubMetric:
    return StubMetric("answer_relevancy", required_fields=("user_input", "response"))


@pytest.fixture
def sample_answered() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="What was the vote?",
        response="The vote was 7-2.",
        retrieved_contexts=["context chunk one", "context chunk two"],
        reference="Seven to two.",
    )


@pytest.fixture
def sample_abstain() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="What is the Fed's view?",
        response="This question does not appear to be answerable from the Bank of England document corpus.",
        retrieved_contexts=[],
        reference="Out of scope.",
    )


# ── Behaviour ──────────────────────────────────────────────


class TestAscoreDispatch:
    """Ensure only the kwargs each metric's ascore declares are passed."""

    def test_dispatches_only_declared_fields(self, rel_stub, sample_answered, tmp_path):
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[rel_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["score"] == 0.5


class TestAbstainSkip:
    def test_context_required_metric_skips_empty_contexts(
        self, faith_stub, sample_abstain, tmp_path
    ):
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_abstain],
            query_ids=["q21"],
            pipeline_name="enhanced",
            metrics=[faith_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["skipped"] == "abstain"
        assert lines[0]["score"] is None
        assert faith_stub.call_count == 0  # never invoked

    def test_non_context_metric_scores_abstain_row(
        self, rel_stub, sample_abstain, tmp_path
    ):
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_abstain],
            query_ids=["q21"],
            pipeline_name="enhanced",
            metrics=[rel_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert lines[0]["skipped"] is None
        assert lines[0]["score"] == 0.5
        assert rel_stub.call_count == 1

    def test_context_required_metrics_constant_is_set(self):
        assert "faithfulness" in CONTEXT_REQUIRED_METRICS
        assert "context_precision_with_reference" in CONTEXT_REQUIRED_METRICS
        assert "context_recall" in CONTEXT_REQUIRED_METRICS
        assert "answer_relevancy" not in CONTEXT_REQUIRED_METRICS


class TestResume:
    def test_skips_already_scored_keys(self, rel_stub, sample_answered, tmp_path):
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[rel_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        assert rel_stub.call_count == 1

        # Second invocation with resume=True: skip existing.
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[rel_stub],
            out_path=out,
            resume=True,
            concurrency=1,
        ))
        assert rel_stub.call_count == 1  # unchanged

    def test_load_done_keys_matches_written_lines(self, rel_stub, sample_answered, tmp_path):
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[rel_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        done = load_done_keys(out)
        assert ("p", "q01", "answer_relevancy") in done

    def test_no_resume_wipes_file(self, rel_stub, sample_answered, tmp_path):
        out = tmp_path / "r.jsonl"
        out.write_text(json.dumps(
            {"pipeline": "p", "query_id": "q01", "metric": "answer_relevancy",
             "score": 0.1, "skipped": None, "err": None, "ts": "old"}
        ) + "\n")
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[rel_stub],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        # resume=False truncates and re-scores
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["score"] == 0.5  # stub value, not the 0.1 from before


class TestErrorPath:
    def test_persists_error_as_null_score(self, sample_answered, tmp_path):
        failing = StubMetric(
            "faithfulness",
            required_fields=("user_input", "response", "retrieved_contexts"),
            raises=RuntimeError("boom"),
        )
        out = tmp_path / "r.jsonl"
        asyncio.run(run_ragas(
            samples=[sample_answered],
            query_ids=["q01"],
            pipeline_name="p",
            metrics=[failing],
            out_path=out,
            resume=False,
            concurrency=1,
        ))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert lines[0]["score"] is None
        assert "RuntimeError" in lines[0]["err"]
        assert "boom" in lines[0]["err"]
