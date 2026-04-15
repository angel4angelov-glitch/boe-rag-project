"""Observability helpers: LangSmith tracing decorator for pipeline runs.

Why a wrapper instead of raw ``@traceable``:

1. **Filters out ``self``** from the trace inputs. Without a custom
   ``process_inputs``, calling ``@traceable`` on an instance method
   dumps the entire pipeline object into the trace payload — including
   ChromaDB clients, LLM config, prompts. Noisy; not actionable.

2. **Tags + metadata** are set consistently. Every pipeline run gets
   ``tags=["pipeline:baseline|enhanced"]`` + ``metadata`` with the
   model, so the LangSmith dashboard filters are meaningful.

3. **No-op when langsmith is absent or tracing is disabled.** The
   decorator still works if someone installs the core without
   langsmith (it's a LangChain transitive dep so unlikely) or if
   ``LANGSMITH_TRACING=false`` (which is the default in tests).
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

try:
    from langsmith import traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover — langsmith is a transitive langchain dep
    _LANGSMITH_AVAILABLE = False

    def traceable(*_args: Any, **_kwargs: Any) -> Callable:  # type: ignore[misc]
        """No-op fallback when langsmith isn't importable."""
        def _decorator(fn: Callable) -> Callable:
            return fn
        return _decorator


def _process_inputs(inputs: dict) -> dict:
    """Drop ``self`` and expose the query verbatim.

    The @traceable decorator passes the method's bound args as a dict.
    We want ``{"query": "..."}`` in the trace payload, not the pipeline
    instance with all its clients.
    """
    out = {k: v for k, v in inputs.items() if k != "self"}
    return out


def traced_run(*, pipeline_name: str) -> Callable:
    """Decorator for ``BaselinePipeline.run`` / ``EnhancedPipeline.run``.

    Args:
        pipeline_name: "baseline" or "enhanced". Used as the trace name
            and as a tag so LangSmith dashboards can group by pipeline.

    Returns:
        A decorator that wraps the target method with ``@traceable``
        when langsmith is available, otherwise a no-op passthrough.
    """
    name = f"{pipeline_name.capitalize()}Pipeline.run"
    tags = [f"pipeline:{pipeline_name}"]

    def _decorator(fn: Callable) -> Callable:
        traced = traceable(
            name=name,
            tags=tags,
            metadata={"pipeline_name": pipeline_name},
            process_inputs=_process_inputs,
        )(fn)

        @wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return traced(*args, **kwargs)
        return _wrapper

    return _decorator
