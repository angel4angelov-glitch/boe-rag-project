"""Domain types and data models.

Enums and dataclasses together — domain types are foundational,
everything else imports from here.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


# ── Domain enums ──────────────────────────────────────────────
# StrEnum: serialises to string for ChromaDB metadata,
# but catches typos at definition time (not at query time).


class DocumentType(StrEnum):
    MPR = "MPR"
    FSR = "FSR"
    MPC_MINUTES = "MPC_minutes"
    SPEECH = "speech"


class SectionCategory(StrEnum):
    GLOBAL_ECONOMY = "global_economy"
    INFLATION = "inflation"
    LABOUR_MARKET = "labour_market"
    DEMAND_OUTPUT = "demand_output"
    POLICY_DISCUSSION = "policy_discussion"
    VOTING = "voting"
    INDIVIDUAL_STATEMENT = "individual_statement"
    BOX_ANALYSIS = "box_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FINANCIAL_STABILITY = "financial_stability"
    FORWARD_GUIDANCE = "forward_guidance"
    SPEECH_MAIN = "speech_main"


# ── Dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkMetadata:
    document_type: DocumentType
    date: str  # "2025-11" (YYYY-MM)
    section_category: SectionCategory
    speaker: str | None  # for speeches / individual statements
    source_url: str
    paragraph_range: str  # MPC: "15-18", MPR: "Box A", Speech: "1"
    title: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int


@dataclass(frozen=True)
class RetrievedDocument:
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata | None  # None for baseline (no metadata stored)


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    sources: Sequence[RetrievedDocument]
    pipeline_name: str  # "baseline" | "enhanced"
    chunks_retrieved: int
    chunks_used: int
    model: str
    crag_rewrites: int  # 0 for baseline
    hallucination_retries: int  # 0 for baseline
    is_grounded: bool | None  # None for baseline
    metadata_filters_used: dict | None
    pipeline_trace: Sequence[str]
    # Rerank ordering — populated by the enhanced pipeline to let
    # evaluation quantify reranker impact. Both empty on baseline,
    # both empty on enhanced when rerank was skipped (≤1 graded doc).
    pre_rerank_ids: Sequence[str] = ()
    post_rerank_ids: Sequence[str] = ()
