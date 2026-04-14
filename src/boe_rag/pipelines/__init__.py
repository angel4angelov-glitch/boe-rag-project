"""Pipelines package: BasePipeline ABC + concrete baseline / enhanced impls."""

from boe_rag.pipelines.base import BasePipeline
from boe_rag.pipelines.baseline import BaselinePipeline
from boe_rag.pipelines.prompts import BASELINE_PROMPT, ENHANCED_GENERATION_PROMPT

__all__ = [
    "BASELINE_PROMPT",
    "ENHANCED_GENERATION_PROMPT",
    "BasePipeline",
    "BaselinePipeline",
]
