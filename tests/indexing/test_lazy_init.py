"""Tests for lazy initialisation — no side effects at import time."""

from __future__ import annotations

import importlib
import os
import sys


def test_import_does_not_create_client_or_lookup_api_key(
    monkeypatch,
) -> None:
    """Importing chroma_store must not hit disk, network, or env."""
    # Wipe any cached import so the module is freshly loaded.
    sys.modules.pop("boe_rag.indexing.chroma_store", None)

    # If the import lazily read OPENAI_API_KEY, removing it would still allow
    # successful import; only the first call to _get_embedding_fn should raise.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = importlib.import_module("boe_rag.indexing.chroma_store")

    # Singletons remain None until first use.
    assert module._client is None, "client must be lazy"
    assert module._embedding_fn is None, "embedding fn must be lazy"


def test_get_embedding_fn_raises_without_api_key(monkeypatch) -> None:
    """First call surfaces a clear error when OPENAI_API_KEY is missing."""
    sys.modules.pop("boe_rag.indexing.chroma_store", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # load_dotenv reads .env files — point it at an empty dir so nothing
    # auto-populates the missing key.
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    module = importlib.import_module("boe_rag.indexing.chroma_store")

    import pytest
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        module._get_embedding_fn()
