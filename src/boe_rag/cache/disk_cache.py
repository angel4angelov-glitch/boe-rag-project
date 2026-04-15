"""Diskcache-backed wrapper for ``BasePipeline``-compatible objects."""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Protocol

import diskcache

from boe_rag.config import GENERATION_MODEL
from boe_rag.models import PipelineResult

logger = logging.getLogger(__name__)


class _PipelineLike(Protocol):
    pipeline_name: str

    def run(self, query: str) -> PipelineResult: ...  # noqa: D102


def _default_version_tag() -> str:
    """SHA-256 prefix of the prompts module source.

    Any edit to ``boe_rag/pipelines/prompts.py`` changes this hash,
    which changes the cache key, which forces a miss — so we never
    serve an answer generated from an outdated prompt.
    """
    prompts_path = Path(__file__).resolve().parent.parent / "pipelines" / "prompts.py"
    try:
        h = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
        return f"prompts:{h[:12]}"
    except OSError:  # pragma: no cover — prompts file always present
        return "prompts:unknown"


class CachedPipeline:
    """Wrap a pipeline with a disk-backed response cache.

    Attributes:
        pipeline_name: Mirror of the inner pipeline's name so
            ``PipelineResult.pipeline_name`` round-trips correctly.
    """

    def __init__(
        self,
        inner: _PipelineLike,
        *,
        cache_dir: Path | str = ".cache/pipeline",
        version_tag: str | None = None,
        size_limit_bytes: int = 1024 * 1024 * 1024,  # 1 GB
    ) -> None:
        self._inner = inner
        self._cache = diskcache.Cache(
            directory=str(cache_dir),
            size_limit=size_limit_bytes,
        )
        self._version_tag = version_tag or _default_version_tag()
        # Best-effort mirror — not all pipeline-like objects expose it.
        self.pipeline_name = getattr(inner, "pipeline_name", "unknown")

    def _cache_key(self, query: str) -> str:
        inner_name = getattr(self._inner, "pipeline_name", "unknown")
        material = f"{inner_name}|{GENERATION_MODEL}|{self._version_tag}|{query}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def run(self, query: str) -> PipelineResult:
        key = self._cache_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("cache HIT: %s", query[:60])
            return pickle.loads(cached)

        logger.debug("cache MISS: %s", query[:60])
        result = self._inner.run(query)
        self._cache.set(key, pickle.dumps(result))
        return result

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()
