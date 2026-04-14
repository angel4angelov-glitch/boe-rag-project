"""Tests for prompt templates.

Asserts the placeholders the pipelines depend on are present and that the
deliberate naive/enhanced prompt distinction is preserved (the enhanced
prompt's citation/grounding rules are part of the evaluation delta).
"""

from __future__ import annotations

from boe_rag.pipelines.prompts import BASELINE_PROMPT, ENHANCED_GENERATION_PROMPT


def test_baseline_prompt_has_required_placeholders() -> None:
    assert "{context}" in BASELINE_PROMPT
    assert "{question}" in BASELINE_PROMPT


def test_baseline_prompt_formats_cleanly() -> None:
    out = BASELINE_PROMPT.format(context="some retrieved text", question="What was the vote?")
    assert "some retrieved text" in out
    assert "What was the vote?" in out
    # Placeholders fully substituted — no curly-brace residue.
    assert "{" not in out
    assert "}" not in out


def test_enhanced_prompt_has_required_placeholders() -> None:
    assert "{context}" in ENHANCED_GENERATION_PROMPT
    assert "{question}" in ENHANCED_GENERATION_PROMPT


def test_enhanced_prompt_carries_domain_rules_baseline_lacks() -> None:
    """The enhanced prompt's citation/grounding rules ARE the delta."""
    enhanced_lower = ENHANCED_GENERATION_PROMPT.lower()
    baseline_lower = BASELINE_PROMPT.lower()

    # Enhanced must instruct citation, grounding, and domain language.
    assert "cite" in enhanced_lower
    assert "bank of england" in enhanced_lower or "boe" in enhanced_lower
    # Baseline must NOT instruct citation — its naivety is part of the design.
    assert "cite" not in baseline_lower
