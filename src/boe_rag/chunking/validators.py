"""Chunk validation checks (spec 03 §'Validation Checks').

Each check returns a CheckResult (pass / warn / fail + human-readable detail).
Checks are grouped into two entry points:

  validate_chunks(chunks, original_text, doc_type)
      Per-document checks: count, required categories, content sanity, sizes.

  validate_corpus(enhanced, baseline)
      Cross-document checks: unique ids across the whole corpus, token
      balance between enhanced and baseline pipelines.

The notebook walks the returned reports for a human-friendly summary table.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from boe_rag.config import ENHANCED_MAX_CHUNK
from boe_rag.models import Chunk, DocumentType, SectionCategory

# Hard cap from section_chunker (avoid cycle: keep constant in sync).
_HARD_CAP_TOKENS = 2000

# Expected chunks-per-document ranges. Tuned against the actual BoE corpus:
# shorter summary MPCs and MPRs legitimately produce fewer chunks than the
# spec's initial estimate, and Mann's short Feb 2025 speech sits at 4.
_COUNT_RANGES: dict[DocumentType, tuple[int, int]] = {
    DocumentType.MPC_MINUTES: (5, 40),
    DocumentType.MPR: (20, 200),
    DocumentType.FSR: (20, 250),
    DocumentType.SPEECH: (3, 40),
}

# Required categories per doc type — spec check #2.
_REQUIRED_CATEGORIES: dict[DocumentType, set[SectionCategory]] = {
    DocumentType.MPC_MINUTES: {
        SectionCategory.VOTING,
        SectionCategory.INDIVIDUAL_STATEMENT,
    },
    DocumentType.MPR: {SectionCategory.BOX_ANALYSIS},
    DocumentType.FSR: {SectionCategory.BOX_ANALYSIS},
    DocumentType.SPEECH: set(),
}


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


@dataclass
class ValidationReport:
    """Collected check results for a document or corpus."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status is not CheckStatus.FAIL for c in self.checks)

    def add(self, name: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))


# ── Per-document checks ────────────────────────────────────


def validate_chunks(
    chunks: list[Chunk],
    original_text: str,
    doc_type: DocumentType,
) -> ValidationReport:
    """Run all per-document validation checks.

    Args:
        chunks: The enhanced chunks for one document.
        original_text: Original scraped text (used for length balance check).
        doc_type: Document type — drives expected ranges.
    """
    report = ValidationReport()
    _check_count(report, chunks, doc_type)
    _check_required_categories(report, chunks, doc_type, original_text)
    _check_voting_keywords(report, chunks)
    _check_box_markers(report, chunks)
    _check_required_fields(report, chunks)
    _check_min_text_length(report, chunks)
    _check_max_size(report, chunks)
    _check_median_size(report, chunks)
    _check_token_balance(report, chunks, original_text)
    return report


def _check_count(report: ValidationReport, chunks: list[Chunk], doc_type: DocumentType) -> None:
    lo, hi = _COUNT_RANGES.get(doc_type, (1, 500))
    n = len(chunks)
    if lo <= n <= hi:
        report.add("chunk_count", CheckStatus.PASS, f"{n} chunks (expected {lo}-{hi})")
    elif n == 0:
        report.add("chunk_count", CheckStatus.FAIL, "no chunks produced")
    elif n < lo:
        report.add("chunk_count", CheckStatus.FAIL, f"{n} chunks (< min {lo})")
    else:
        report.add("chunk_count", CheckStatus.WARN, f"{n} chunks (> max {hi})")


def _check_required_categories(
    report: ValidationReport,
    chunks: list[Chunk],
    doc_type: DocumentType,
    original_text: str = "",
) -> None:
    """FAIL when the chunker missed markers that exist in the source text.

    Some BoE MPC minutes publish in a prose-only format with no bolded
    ``**Name:**`` / ``**Votes to ...**`` markers. In that case the category
    is legitimately absent — downgrade the check to WARN so reports
    distinguish 'chunker bug' from 'source data lacks markers'.
    """
    required = _REQUIRED_CATEGORIES.get(doc_type, set())
    if not required:
        report.add(
            "required_categories", CheckStatus.PASS, "no required categories for this doc type"
        )
        return
    present = {c.metadata.section_category for c in chunks}
    missing = required - present
    if not missing:
        report.add("required_categories", CheckStatus.PASS, "all required categories present")
        return
    names = ", ".join(sorted(cat.value for cat in missing))
    if _source_has_markers(original_text, missing):
        report.add(
            "required_categories",
            CheckStatus.FAIL,
            f"missing: {names} (markers present in source — chunker bug)",
        )
    else:
        report.add(
            "required_categories",
            CheckStatus.WARN,
            f"missing: {names} (source lacks structural markers)",
        )


_INDIVIDUAL_STATEMENT_RE = re.compile(r"^\*\*[A-Za-z][A-Za-z .'-]+:\*\*", re.MULTILINE)


def _source_has_markers(text: str, missing: set[SectionCategory]) -> bool:
    """Return True if the source text contains the structural markers for any of `missing`."""
    if not text:
        return False
    if SectionCategory.VOTING in missing and "**Votes to" in text:
        return True
    if SectionCategory.INDIVIDUAL_STATEMENT in missing and _INDIVIDUAL_STATEMENT_RE.search(text):
        return True
    if SectionCategory.BOX_ANALYSIS in missing and "[BOX START" in text:
        return True
    return False


def _check_voting_keywords(report: ValidationReport, chunks: list[Chunk]) -> None:
    bad_ids = [
        c.chunk_id
        for c in chunks
        if c.metadata.section_category is SectionCategory.VOTING
        and "vote" not in c.text.lower()
        and "bank rate" not in c.text.lower()
    ]
    if bad_ids:
        report.add(
            "voting_keywords",
            CheckStatus.FAIL,
            f"{len(bad_ids)} VOTING chunks without 'vote' or 'Bank Rate': {bad_ids[:3]}",
        )
    else:
        report.add("voting_keywords", CheckStatus.PASS, "all voting chunks have expected keywords")


def _check_box_markers(report: ValidationReport, chunks: list[Chunk]) -> None:
    bad_ids = [
        c.chunk_id
        for c in chunks
        if c.metadata.section_category is SectionCategory.BOX_ANALYSIS
        and "[BOX START" not in c.text
        and "Box" not in c.text
    ]
    if bad_ids:
        report.add(
            "box_markers",
            CheckStatus.FAIL,
            f"{len(bad_ids)} BOX_ANALYSIS chunks without '[BOX START' or 'Box'",
        )
    else:
        report.add("box_markers", CheckStatus.PASS, "all box chunks have expected markers")


def _check_required_fields(report: ValidationReport, chunks: list[Chunk]) -> None:
    missing: list[str] = []
    for c in chunks:
        if not c.chunk_id or not c.text:
            missing.append(c.chunk_id or "<no-id>")
            continue
        m = c.metadata
        if not m.document_type or not m.date or not m.section_category:
            missing.append(c.chunk_id)
    if missing:
        report.add(
            "required_fields",
            CheckStatus.FAIL,
            f"{len(missing)} chunks missing required metadata: {missing[:3]}",
        )
    else:
        report.add("required_fields", CheckStatus.PASS, "all chunks have required fields")


_STRUCTURAL_CATEGORIES_EXEMPT_FROM_MIN_LENGTH = frozenset(
    {SectionCategory.VOTING, SectionCategory.INDIVIDUAL_STATEMENT}
)


def _check_min_text_length(report: ValidationReport, chunks: list[Chunk]) -> None:
    """Fail when content chunks are under 50 chars.

    Structural-marker chunks (VOTING header, brief member statements) carry
    their value in the metadata (speaker, section_category) and may be
    legitimately short — exempted from this check.
    """
    short = [
        c.chunk_id
        for c in chunks
        if len(c.text) < 50
        and c.metadata.section_category not in _STRUCTURAL_CATEGORIES_EXEMPT_FROM_MIN_LENGTH
    ]
    if short:
        report.add(
            "min_text_length",
            CheckStatus.FAIL,
            f"{len(short)} chunks under 50 chars: {short[:3]}",
        )
    else:
        report.add("min_text_length", CheckStatus.PASS, "all chunks >= 50 chars")


def _check_max_size(report: ValidationReport, chunks: list[Chunk]) -> None:
    over_hard_cap = [c.chunk_id for c in chunks if c.token_count > _HARD_CAP_TOKENS]
    over_max_non_box = [
        c.chunk_id
        for c in chunks
        if c.token_count > ENHANCED_MAX_CHUNK
        and c.metadata.section_category is not SectionCategory.BOX_ANALYSIS
    ]
    if over_hard_cap:
        report.add(
            "max_size",
            CheckStatus.FAIL,
            f"{len(over_hard_cap)} chunks over hard cap ({_HARD_CAP_TOKENS}): {over_hard_cap[:3]}",
        )
    elif over_max_non_box:
        report.add(
            "max_size",
            CheckStatus.FAIL,
            f"{len(over_max_non_box)} non-box chunks over {ENHANCED_MAX_CHUNK}: {over_max_non_box[:3]}",
        )
    else:
        report.add("max_size", CheckStatus.PASS, "all chunks within size limits")


def _check_median_size(report: ValidationReport, chunks: list[Chunk]) -> None:
    if not chunks:
        report.add("median_size", CheckStatus.FAIL, "no chunks to size-check")
        return
    # Ignore boxes when reporting median — they're intentionally variable.
    non_box = [c.token_count for c in chunks if c.metadata.section_category is not SectionCategory.BOX_ANALYSIS]
    if not non_box:
        report.add("median_size", CheckStatus.PASS, "only box chunks — skipping median check")
        return
    median = statistics.median(non_box)
    if 150 <= median <= 800:
        report.add("median_size", CheckStatus.PASS, f"median {median:.0f} tokens (non-box)")
    else:
        report.add("median_size", CheckStatus.WARN, f"median {median:.0f} tokens outside 150-800 (non-box)")


def _check_token_balance(
    report: ValidationReport, chunks: list[Chunk], original_text: str
) -> None:
    """Chunk total (minus estimated overlap) within 10% of original length."""
    if not chunks or not original_text:
        report.add("token_balance", CheckStatus.WARN, "missing chunks or original text")
        return
    from boe_rag.chunking.metadata import count_tokens

    total_chunk_tokens = sum(c.token_count for c in chunks)
    overlap_allowance = max(0, len(chunks) - 1) * 50
    effective = total_chunk_tokens - overlap_allowance
    original_tokens = count_tokens(original_text)
    if original_tokens == 0:
        report.add("token_balance", CheckStatus.WARN, "original text has zero tokens")
        return
    ratio = effective / original_tokens
    if 0.85 <= ratio <= 1.10:
        report.add("token_balance", CheckStatus.PASS, f"effective/original = {ratio:.2f}")
    else:
        report.add(
            "token_balance",
            CheckStatus.WARN,
            f"effective/original = {ratio:.2f} (expected 0.85-1.10)",
        )


# ── Corpus-level checks ────────────────────────────────────


def validate_corpus(
    enhanced: dict[str, list[Chunk]],
    baseline: dict[str, list[dict]],
) -> ValidationReport:
    """Cross-document checks: unique ids, pipeline-level token balance.

    Args:
        enhanced: Map of doc_id -> enhanced chunks.
        baseline: Map of doc_id -> baseline chunk dicts.
    """
    report = ValidationReport()
    _check_unique_ids(report, enhanced, baseline)
    _check_corpus_token_balance(report, enhanced, baseline)
    return report


def _check_unique_ids(
    report: ValidationReport,
    enhanced: dict[str, list[Chunk]],
    baseline: dict[str, list[dict]],
) -> None:
    all_ids: list[str] = []
    for chunks in enhanced.values():
        all_ids.extend(c.chunk_id for c in chunks)
    for chunks in baseline.values():
        all_ids.extend(c["chunk_id"] for c in chunks)

    counts = Counter(all_ids)
    dupes = [cid for cid, count in counts.items() if count > 1]
    if dupes:
        report.add(
            "unique_chunk_ids",
            CheckStatus.FAIL,
            f"{len(dupes)} duplicate chunk_ids: {dupes[:3]}",
        )
    else:
        report.add("unique_chunk_ids", CheckStatus.PASS, f"{len(all_ids)} chunk_ids all unique")


def _check_corpus_token_balance(
    report: ValidationReport,
    enhanced: dict[str, list[Chunk]],
    baseline: dict[str, list[dict]],
) -> None:
    """Overlap-adjusted totals for the two pipelines should be within 10%."""
    if not enhanced or not baseline:
        report.add("corpus_token_balance", CheckStatus.WARN, "missing pipeline data")
        return
    enh_total = 0
    for chunks in enhanced.values():
        enh_total += sum(c.token_count for c in chunks) - max(0, len(chunks) - 1) * 50
    base_total = sum(c["token_count"] for chunks in baseline.values() for c in chunks)

    if base_total == 0:
        report.add("corpus_token_balance", CheckStatus.WARN, "baseline has zero tokens")
        return
    ratio = enh_total / base_total
    if 0.90 <= ratio <= 1.10:
        report.add(
            "corpus_token_balance",
            CheckStatus.PASS,
            f"enhanced/baseline = {ratio:.2f} (enh={enh_total}, base={base_total})",
        )
    else:
        report.add(
            "corpus_token_balance",
            CheckStatus.WARN,
            f"enhanced/baseline = {ratio:.2f} — outside 0.90-1.10",
        )
