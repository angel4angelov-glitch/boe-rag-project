"""Tests for prompt templates.

Asserts the placeholders the pipelines depend on are present and that the
deliberate naive/enhanced prompt distinction is preserved (the enhanced
prompt's citation/grounding rules are part of the evaluation delta).
"""

from __future__ import annotations

from boe_rag.pipelines.prompts import (
    ANALYZE_QUERY_PROMPT,
    BASELINE_PROMPT,
    ENHANCED_GENERATION_PROMPT,
    GRADING_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    HALLUCINATION_RETRY_PROMPT,
    REWRITE_QUERY_PROMPT,
)


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


# ── Enhanced CRAG pipeline prompts (spec 06) ────────────────


def test_analyze_query_prompt_lists_allowed_filter_fields() -> None:
    """The LLM needs to know EXACTLY which metadata fields it can filter on."""
    p = ANALYZE_QUERY_PROMPT.lower()
    assert "document_type" in p
    assert "date" in p
    assert "section_category" in p
    assert "speaker" in p
    # And it must know the valid document_type values.
    assert "mpc_minutes" in p or "mpc minutes" in p
    assert "mpr" in p
    assert "fsr" in p
    assert "speech" in p


def test_analyze_query_prompt_interpolates_question() -> None:
    out = ANALYZE_QUERY_PROMPT.format(question="What was the MPC vote in November 2025?")
    assert "What was the MPC vote in November 2025?" in out
    assert "{question}" not in out


def test_grading_prompt_has_document_and_question_placeholders() -> None:
    assert "{document}" in GRADING_PROMPT
    assert "{question}" in GRADING_PROMPT
    # Binary grading: must mention the two possible answers.
    p = GRADING_PROMPT.lower()
    assert "yes" in p and "no" in p


def test_rewrite_prompt_interpolates_question() -> None:
    out = REWRITE_QUERY_PROMPT.format(question="MPC vote?")
    assert "MPC vote?" in out
    assert "{question}" not in out
    # Must instruct the LLM about the rewrite goal.
    assert "rewrite" in out.lower() or "reformulate" in out.lower()


def test_hallucination_check_prompt_has_context_and_answer() -> None:
    assert "{context}" in HALLUCINATION_CHECK_PROMPT
    assert "{answer}" in HALLUCINATION_CHECK_PROMPT
    p = HALLUCINATION_CHECK_PROMPT.lower()
    assert "yes" in p and "no" in p


def test_hallucination_retry_prompt_is_stricter_than_first_pass() -> None:
    """Second generation must explicitly reference the groundedness failure."""
    p = HALLUCINATION_RETRY_PROMPT.lower()
    # Must carry the generation placeholders forward.
    assert "{context}" in HALLUCINATION_RETRY_PROMPT
    assert "{question}" in HALLUCINATION_RETRY_PROMPT
    # Must communicate that the prior answer failed verification.
    assert "previous" in p or "earlier" in p or "failed" in p
