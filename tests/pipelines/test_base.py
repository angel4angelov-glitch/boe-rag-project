"""Tests for the BasePipeline abstract interface."""

from __future__ import annotations

import pytest

from boe_rag.models import PipelineResult
from boe_rag.pipelines.base import BasePipeline


def test_basepipeline_cannot_be_instantiated_directly() -> None:
    """Abstract — subclass MUST implement run()."""
    with pytest.raises(TypeError):
        BasePipeline()  # type: ignore[abstract]


def test_basepipeline_subclass_without_run_is_abstract() -> None:
    class _Incomplete(BasePipeline):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_basepipeline_subclass_with_run_can_be_instantiated() -> None:
    class _Stub(BasePipeline):
        def run(self, query: str) -> PipelineResult:
            return PipelineResult(
                answer="ok",
                sources=(),
                pipeline_name="stub",
                chunks_retrieved=0,
                chunks_used=0,
                model="none",
                crag_rewrites=0,
                hallucination_retries=0,
                is_grounded=None,
                metadata_filters_used=None,
                pipeline_trace=("stub",),
            )

    pipeline = _Stub()
    result = pipeline.run("anything")
    assert isinstance(result, PipelineResult)
    assert result.pipeline_name == "stub"
