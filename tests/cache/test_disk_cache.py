"""CachedPipeline wrapper: cache hits, misses, invalidation, roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from boe_rag.cache import CachedPipeline
from boe_rag.models import PipelineResult, RetrievedDocument


# ── Fixtures ─────────────────────────────────────────────


class FakePipeline:
    """Spy pipeline — records calls, returns a canned PipelineResult."""

    def __init__(self, answer: str = "canned") -> None:
        self.calls: list[str] = []
        self._answer = answer

    def run(self, query: str) -> PipelineResult:
        self.calls.append(query)
        return PipelineResult(
            answer=f"{self._answer}: {query}",
            sources=(
                RetrievedDocument(chunk_id="c1", text="ctx", score=0.5, metadata=None),
            ),
            pipeline_name="enhanced",
            chunks_retrieved=1, chunks_used=1,
            model="claude-sonnet-4-20250514",
            crag_rewrites=0, hallucination_retries=0,
            is_grounded=True, metadata_filters_used=None,
            pipeline_trace=("retrieve", "generate"),
        )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# ── Behaviour ────────────────────────────────────────────


class TestCachedPipeline:
    def test_miss_calls_inner(self, cache_dir: Path) -> None:
        inner = FakePipeline()
        cached = CachedPipeline(inner, cache_dir=cache_dir)
        r = cached.run("question A")
        assert inner.calls == ["question A"]
        assert r.answer == "canned: question A"

    def test_hit_skips_inner(self, cache_dir: Path) -> None:
        inner = FakePipeline()
        cached = CachedPipeline(inner, cache_dir=cache_dir)
        cached.run("question A")
        cached.run("question A")
        assert len(inner.calls) == 1  # second call served from cache

    def test_different_question_is_miss(self, cache_dir: Path) -> None:
        inner = FakePipeline()
        cached = CachedPipeline(inner, cache_dir=cache_dir)
        cached.run("question A")
        cached.run("question B")
        assert inner.calls == ["question A", "question B"]

    def test_pipeline_result_roundtrips(self, cache_dir: Path) -> None:
        inner = FakePipeline(answer="roundtrip")
        cached = CachedPipeline(inner, cache_dir=cache_dir)
        first = cached.run("Q")
        second = cached.run("Q")
        assert first.answer == second.answer
        assert first.chunks_retrieved == second.chunks_retrieved
        assert first.sources[0].chunk_id == second.sources[0].chunk_id
        assert first.pipeline_trace == second.pipeline_trace

    def test_key_invalidation_on_prompt_change(self, cache_dir: Path) -> None:
        """Changing the prompts-module hash should miss even for same question."""
        inner = FakePipeline()
        cached = CachedPipeline(inner, cache_dir=cache_dir, version_tag="v1")
        cached.run("Q")

        cached_v2 = CachedPipeline(inner, cache_dir=cache_dir, version_tag="v2")
        cached_v2.run("Q")
        assert len(inner.calls) == 2  # v2 ignores v1's cache entry

    def test_persists_across_instances(self, cache_dir: Path) -> None:
        """Two CachedPipeline instances on the same dir share the cache."""
        cached_1 = CachedPipeline(FakePipeline(), cache_dir=cache_dir)
        cached_1.run("shared")

        spy = FakePipeline()
        cached_2 = CachedPipeline(spy, cache_dir=cache_dir)
        cached_2.run("shared")
        assert spy.calls == []  # served from cache on disk

    def test_clear_evicts_all(self, cache_dir: Path) -> None:
        inner = FakePipeline()
        cached = CachedPipeline(inner, cache_dir=cache_dir)
        cached.run("X")
        cached.clear()
        cached.run("X")
        assert len(inner.calls) == 2  # clear() forced miss
