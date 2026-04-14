"""Tests for category assignment + token counting."""

from __future__ import annotations

from boe_rag.chunking.metadata import assign_category, count_tokens, normalise_speaker
from boe_rag.models import DocumentType, SectionCategory


# ── Speaker normalisation ───────────────────────────────────


def test_normalise_speaker_drops_middle_initial() -> None:
    assert normalise_speaker("Catherine L Mann") == "Catherine Mann"
    assert normalise_speaker("Catherine L. Mann") == "Catherine Mann"


def test_normalise_speaker_drops_honorifics() -> None:
    assert normalise_speaker("Professor Alan Taylor") == "Alan Taylor"
    assert normalise_speaker("Dr Sarah Breeden") == "Sarah Breeden"
    assert normalise_speaker("Prof. Megan Greene") == "Megan Greene"


def test_normalise_speaker_passes_simple_names_through() -> None:
    assert normalise_speaker("Andrew Bailey") == "Andrew Bailey"
    assert normalise_speaker("Huw Pill") == "Huw Pill"


def test_normalise_speaker_handles_empty_and_whitespace() -> None:
    assert normalise_speaker("") == ""
    assert normalise_speaker("   ") == ""


def test_normalise_speaker_keeps_hyphenated_surnames_intact() -> None:
    """Hyphenated last names must not lose their hyphen during normalisation."""
    assert normalise_speaker("Mary Smith-Jones") == "Mary Smith-Jones"


# ── Token counting ──────────────────────────────────────────


def test_count_tokens_empty_string() -> None:
    assert count_tokens("") == 0


def test_count_tokens_stable_ordering() -> None:
    """Same text twice returns same count — tokenizer is deterministic."""
    text = "The MPC voted to maintain Bank Rate at 4%."
    assert count_tokens(text) == count_tokens(text)
    # Sanity: a ~10-word sentence should be ~10-15 tokens, not 100s.
    assert 8 < count_tokens(text) < 20


# ── Structural section types (definitive mappings) ──────────


def test_vote_section_type_maps_to_voting() -> None:
    assert assign_category("", "vote", DocumentType.MPC_MINUTES) is SectionCategory.VOTING


def test_member_section_type_maps_to_individual_statement() -> None:
    cat = assign_category("Andrew Bailey", "member", DocumentType.MPC_MINUTES)
    assert cat is SectionCategory.INDIVIDUAL_STATEMENT


def test_box_section_type_maps_to_box_analysis() -> None:
    cat = assign_category("Box A: Firms' costs", "box", DocumentType.MPR)
    assert cat is SectionCategory.BOX_ANALYSIS


def test_structural_overrides_beat_heading_keywords() -> None:
    """Even if the heading contains 'inflation', a 'vote' section is VOTING."""
    cat = assign_category("Inflation outlook", "vote", DocumentType.MPC_MINUTES)
    assert cat is SectionCategory.VOTING


# ── MPR heading-based classification ────────────────────────


def test_mpr_inflation_subsection() -> None:
    cat = assign_category("1.1: Inflation", "h3", DocumentType.MPR)
    assert cat is SectionCategory.INFLATION


def test_mpr_activity_subsection_is_demand_output() -> None:
    cat = assign_category("1.2: Activity", "h3", DocumentType.MPR)
    assert cat is SectionCategory.DEMAND_OUTPUT


def test_mpr_global_subsection_is_global_economy() -> None:
    cat = assign_category("1.3: Global and financial conditions", "h3", DocumentType.MPR)
    assert cat is SectionCategory.GLOBAL_ECONOMY


def test_mpr_labour_market_heading() -> None:
    cat = assign_category("The labour market has continued to soften", "h3", DocumentType.MPR)
    assert cat is SectionCategory.LABOUR_MARKET


def test_mpr_forward_guidance_on_outlook_or_projection() -> None:
    assert assign_category("3: Outlook and risks", "h2", DocumentType.MPR) is SectionCategory.FORWARD_GUIDANCE
    assert assign_category("Key policy judgement 1", "h3", DocumentType.MPR) is SectionCategory.FORWARD_GUIDANCE
    assert assign_category("3.1: Central projection", "h3", DocumentType.MPR) is SectionCategory.FORWARD_GUIDANCE


def test_mpr_unknown_heading_defaults_to_policy_discussion() -> None:
    cat = assign_category("Monetary Policy Summary", "h2", DocumentType.MPR)
    assert cat is SectionCategory.POLICY_DISCUSSION


# ── FSR heading-based classification ────────────────────────


def test_fsr_global_vulnerabilities_is_risk_assessment() -> None:
    cat = assign_category("2: Global vulnerabilities", "h2", DocumentType.FSR)
    assert cat is SectionCategory.RISK_ASSESSMENT


def test_fsr_banking_resilience_is_financial_stability() -> None:
    cat = assign_category("4: UK banking sector resilience", "h2", DocumentType.FSR)
    assert cat is SectionCategory.FINANCIAL_STABILITY


def test_fsr_household_debt_is_financial_stability() -> None:
    cat = assign_category("3: UK household and corporate debt vulnerabilities", "h2", DocumentType.FSR)
    assert cat is SectionCategory.FINANCIAL_STABILITY


def test_fsr_unknown_heading_defaults_to_financial_stability() -> None:
    """FSR catch-all is financial_stability (not policy_discussion)."""
    cat = assign_category("Annex 1: Macroprudential policy decisions", "h2", DocumentType.FSR)
    assert cat is SectionCategory.FINANCIAL_STABILITY


# ── MPC heading-based classification ────────────────────────


def test_mpc_discussion_is_policy_discussion() -> None:
    cat = assign_category("The Committee's discussions", "h3", DocumentType.MPC_MINUTES)
    assert cat is SectionCategory.POLICY_DISCUSSION


def test_mpc_default_is_policy_discussion() -> None:
    cat = assign_category("Anything else", "h2", DocumentType.MPC_MINUTES)
    assert cat is SectionCategory.POLICY_DISCUSSION


# ── Speech heading-based classification ─────────────────────


def test_speech_default_is_speech_main() -> None:
    cat = assign_category("Where have we come from?", "h3", DocumentType.SPEECH)
    assert cat is SectionCategory.SPEECH_MAIN


def test_speech_outlook_heading_is_forward_guidance() -> None:
    cat = assign_category(
        "Where next? The scenarios for policy in 2025", "h3", DocumentType.SPEECH
    )
    assert cat is SectionCategory.FORWARD_GUIDANCE
