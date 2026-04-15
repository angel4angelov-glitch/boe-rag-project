"""Unit tests for boe_rag.observability.traced_run.

Tests don't need real LangSmith; conftest.py forces tracing OFF for the
whole test session. What we test here:
  - Decorator preserves the wrapped function's return value
  - Decorator preserves functools metadata (__name__, __doc__)
  - _process_inputs filters out ``self`` from trace payloads
  - Decorator works on bound methods (not just functions)
"""

from __future__ import annotations

from boe_rag.observability import _process_inputs, traced_run


class TestProcessInputs:
    def test_drops_self_key(self) -> None:
        out = _process_inputs({"self": "pipeline_obj", "query": "q"})
        assert out == {"query": "q"}

    def test_preserves_other_keys(self) -> None:
        out = _process_inputs({"self": "x", "a": 1, "b": 2})
        assert out == {"a": 1, "b": 2}

    def test_no_self_noop(self) -> None:
        out = _process_inputs({"query": "q", "k": 5})
        assert out == {"query": "q", "k": 5}


class TestTracedRun:
    def test_decorator_preserves_return_value(self) -> None:
        @traced_run(pipeline_name="test")
        def run(self, query: str) -> str:
            return f"answer: {query}"

        result = run(None, "hello")
        assert result == "answer: hello"

    def test_decorator_preserves_function_name(self) -> None:
        @traced_run(pipeline_name="test")
        def run(self, query: str) -> str:
            """A nice docstring."""
            return query

        # functools.wraps propagation
        assert run.__name__ == "run"
        assert run.__doc__ == "A nice docstring."

    def test_decorator_works_on_class_method(self) -> None:
        """Mirrors how BaselinePipeline / EnhancedPipeline use it."""

        class FakePipeline:
            @traced_run(pipeline_name="fake")
            def run(self, query: str) -> dict:
                return {"q": query, "p": "fake"}

        out = FakePipeline().run("test query")
        assert out == {"q": "test query", "p": "fake"}

    def test_decorator_passes_through_exceptions(self) -> None:
        @traced_run(pipeline_name="test")
        def run(self, query: str) -> str:
            raise RuntimeError("kaboom")

        try:
            run(None, "x")
        except RuntimeError as e:
            assert str(e) == "kaboom"
        else:
            raise AssertionError("expected RuntimeError")
