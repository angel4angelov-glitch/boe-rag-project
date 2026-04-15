"""Pytest config: force-disable LangSmith tracing across the whole suite.

Without this, running pytest with LANGSMITH_TRACING=true set in the
shell would spam the dashboard with garbage traces on every CI run and
blow the 5k/month free-tier quota in an afternoon.

Two things to be careful about:

1. ``langsmith.utils.get_env_var`` is ``@lru_cache(maxsize=100)`` —
   once the SDK has read LANGSMITH_TRACING, a later ``monkeypatch.setenv``
   won't take effect unless we clear the cache.
2. The env var must be set BEFORE any langchain/langsmith import fires
   on the module being tested. A session-scoped fixture at collection
   time is the safest place.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_langsmith_in_tests() -> None:
    """Force LangSmith tracing off for the entire test session.

    Pops any ambient setting, writes the explicit off-value, clears the
    langsmith env-var cache if the SDK is installed. Safe to run even
    when langsmith is absent (import guarded).
    """
    for key in (
        "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING",
    ):
        os.environ.pop(key, None)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    try:
        from langsmith.utils import get_env_var
        get_env_var.cache_clear()
    except (ImportError, AttributeError):
        pass
