"""Abstract pipeline interface — both pipelines implement the same surface.

Evaluation code (spec 07) iterates over ``BasePipeline`` instances and
calls ``run(query)`` without caring whether the implementation is the
naive baseline or the full CRAG enhanced pipeline. Keeping the contract
narrow (one method, one return type) is what makes the comparison
apples-to-apples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from boe_rag.models import PipelineResult


class BasePipeline(ABC):
    """Abstract interface for RAG pipelines."""

    @abstractmethod
    def run(self, query: str) -> PipelineResult:
        """Execute the pipeline on a query and return a structured result."""
