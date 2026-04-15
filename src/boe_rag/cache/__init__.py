"""On-disk pipeline response cache.

Wraps any object with a ``run(query: str) -> PipelineResult`` method —
returns the cached result when the same query has been seen before.

Key invalidation is deliberate:
  - **pipeline_name** (baseline / enhanced) — different pipelines,
    different answers, different cache namespace.
  - **generation model** — upgrading Claude versions invalidates.
  - **version_tag** — hash of the ``boe_rag.pipelines.prompts`` module
    source by default. Any prompt edit invalidates the whole cache.
  - **question** — the actual user query text.

Use case: demo safety. Running the same question twice during a live
walkthrough shouldn't burn API credits or risk a 429. Also useful
during report iteration when copy-tuning the same queries.

Not intended for production: no TTL, no per-user namespacing, no
cache-stampede protection.
"""

from boe_rag.cache.disk_cache import CachedPipeline

__all__ = ["CachedPipeline"]
