"""Section category assignment + token counting.

Category assignment is a function of:
  1. Structural section type (definitive: vote/member/box override headings).
  2. Heading keywords, per document type (MPR/FSR/MPC/Speech have different
     keyword tables because the same word means different things in different
     documents, e.g. 'markets' is financial-stability in FSR, policy-discussion
     in MPR).

Keyword tables are ordered — the first matching rule wins. This keeps the
logic readable and debuggable (unlike regex trees or fuzzy scoring).

Token counting uses tiktoken's cl100k_base — the encoding for
text-embedding-3-small, so reported counts match what the embedding model
will actually see.
"""

from __future__ import annotations

import tiktoken

from boe_rag.models import DocumentType, SectionCategory

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in text."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))


# Ordered (substring, category) rules — first match wins.
# All keys are lowercase; compared against heading.lower().
_MPR_RULES: tuple[tuple[str, SectionCategory], ...] = (
    # Forward guidance: policy-maker framing of the future.
    ("outlook", SectionCategory.FORWARD_GUIDANCE),
    ("projection", SectionCategory.FORWARD_GUIDANCE),
    ("scenario", SectionCategory.FORWARD_GUIDANCE),
    ("judgement", SectionCategory.FORWARD_GUIDANCE),
    ("risks", SectionCategory.FORWARD_GUIDANCE),
    # Labour market before 'inflation' because 'wage inflation' shouldn't
    # classify as INFLATION (wage section).
    ("labour", SectionCategory.LABOUR_MARKET),
    ("employment", SectionCategory.LABOUR_MARKET),
    ("unemployment", SectionCategory.LABOUR_MARKET),
    ("inflation", SectionCategory.INFLATION),
    ("global", SectionCategory.GLOBAL_ECONOMY),
    ("international", SectionCategory.GLOBAL_ECONOMY),
    ("world", SectionCategory.GLOBAL_ECONOMY),
    ("external", SectionCategory.GLOBAL_ECONOMY),
    ("activity", SectionCategory.DEMAND_OUTPUT),
    ("demand", SectionCategory.DEMAND_OUTPUT),
    ("consumption", SectionCategory.DEMAND_OUTPUT),
    ("gdp", SectionCategory.DEMAND_OUTPUT),
    ("household", SectionCategory.DEMAND_OUTPUT),
)

_FSR_RULES: tuple[tuple[str, SectionCategory], ...] = (
    # UK-specific stability sections first — they often also contain
    # 'vulnerabilities' but belong to FINANCIAL_STABILITY, not RISK_ASSESSMENT.
    ("household", SectionCategory.FINANCIAL_STABILITY),
    ("corporate", SectionCategory.FINANCIAL_STABILITY),
    ("bank", SectionCategory.FINANCIAL_STABILITY),
    ("stress test", SectionCategory.FINANCIAL_STABILITY),
    ("resilience", SectionCategory.FINANCIAL_STABILITY),
    ("capital", SectionCategory.FINANCIAL_STABILITY),
    ("market-based", SectionCategory.FINANCIAL_STABILITY),
    ("structural change", SectionCategory.FINANCIAL_STABILITY),
    # Global / systemic risk framing.
    ("global", SectionCategory.RISK_ASSESSMENT),
    ("financial market", SectionCategory.RISK_ASSESSMENT),
    ("risk environment", SectionCategory.RISK_ASSESSMENT),
    ("vulnerabilit", SectionCategory.RISK_ASSESSMENT),
)

_SPEECH_RULES: tuple[tuple[str, SectionCategory], ...] = (
    ("outlook", SectionCategory.FORWARD_GUIDANCE),
    ("scenario", SectionCategory.FORWARD_GUIDANCE),
    ("where next", SectionCategory.FORWARD_GUIDANCE),
    ("forward", SectionCategory.FORWARD_GUIDANCE),
)


def _match_rules(
    heading_lower: str, rules: tuple[tuple[str, SectionCategory], ...]
) -> SectionCategory | None:
    """Return the first rule's category whose substring is in heading_lower, or None."""
    for keyword, category in rules:
        if keyword in heading_lower:
            return category
    return None


def assign_category(
    heading: str,
    section_type: str,
    doc_type: DocumentType,
) -> SectionCategory:
    """Map a parsed section to a SectionCategory.

    Resolution order:
      1. Structural section_type (vote / member / box) — definitive.
      2. Heading-keyword match from the document-type's rule table.
      3. Per-doc-type default (MPC/MPR: POLICY_DISCUSSION; FSR:
         FINANCIAL_STABILITY; Speech: SPEECH_MAIN).

    Args:
        heading: The section heading text (may be empty).
        section_type: One of {"h2","h3","box","vote","member","text"}.
        doc_type: The parent document's type.
    """
    if section_type == "vote":
        return SectionCategory.VOTING
    if section_type == "member":
        return SectionCategory.INDIVIDUAL_STATEMENT
    if section_type == "box":
        return SectionCategory.BOX_ANALYSIS

    heading_lower = heading.lower()

    if doc_type is DocumentType.MPR:
        return _match_rules(heading_lower, _MPR_RULES) or SectionCategory.POLICY_DISCUSSION
    if doc_type is DocumentType.FSR:
        return _match_rules(heading_lower, _FSR_RULES) or SectionCategory.FINANCIAL_STABILITY
    if doc_type is DocumentType.SPEECH:
        return _match_rules(heading_lower, _SPEECH_RULES) or SectionCategory.SPEECH_MAIN
    # DocumentType.MPC_MINUTES — discussion is the catch-all.
    return SectionCategory.POLICY_DISCUSSION
