"""Enhanced section-aware chunker.

Two-pass pipeline:
  Pass 1 — parse_document(text): walks the document line-by-line, using the
           structural markers defined in spec 02 (## / ### / N: / **Name:** /
           **Votes** / [BOX START]...[BOX END]) to produce a flat list of
           RawSection objects.
  Pass 2 — chunk_document(...): categorises each RawSection, merges small
           adjacent sections of the same category, splits oversized sections
           at paragraph boundaries, applies intra-section overlap, and emits
           Chunk objects with full ChunkMetadata.

This module is separate from chunk_all (runner.py) so pass-1 and pass-2 can
be tested in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from boe_rag.chunking.metadata import assign_category, count_tokens, normalise_speaker
from boe_rag.config import ENHANCED_MAX_CHUNK, ENHANCED_MIN_CHUNK, ENHANCED_OVERLAP
from boe_rag.models import Chunk, ChunkMetadata, DocumentType, SectionCategory

# Regex patterns — interface contract from spec 02.
H2_PATTERN = re.compile(r"^## (.+)$")
H3_PATTERN = re.compile(r"^### (.+)$")
PARA_PATTERN = re.compile(r"^(\d+)[:.]\s(.+)")
VOTE_PATTERN = re.compile(r"^\*\*Votes to .+\*\*$")
MEMBER_PATTERN = re.compile(r"^\*\*([A-Za-z .'-]+):\*\*\s*(.+)")
BOX_START = re.compile(r"^\[BOX START: (.+)\]$")
BOX_END = re.compile(r"^\[BOX END\]$")

# Hard cap even for boxes (spec acceptance criterion #7).
HARD_CAP_TOKENS = 2000


# ── Pass 1: structural parsing ──────────────────────────────


@dataclass
class RawSection:
    """A contiguous region of document text with a detected structural type.

    Attributes:
        heading: Human-readable label (heading text for h2/h3, speaker for
            member, box title for box, vote-header line for vote).
        lines: The raw text lines (including the marker line itself for
            h2/h3/vote/member/box) in document order.
        section_type: One of ``{"h2","h3","box","vote","member","text"}``.
            ``"text"`` is the initial preamble before any marker.
        paragraph_numbers: Numeric paragraph labels (``N:`` format) seen in
            this section — used for MPC paragraph_range metadata and for
            splitting oversized discussion sections at paragraph boundaries.
        speaker: MPC member name when section_type == "member".
        parent_h2 / parent_h3: Enclosing chapter / sub-chapter heading text
            for nesting context. ``h3`` sections carry their parent ``h2``;
            non-heading sections carry whatever was most recently seen.
    """

    heading: str
    lines: list[str] = field(default_factory=list)
    section_type: str = "text"
    paragraph_numbers: list[int] = field(default_factory=list)
    speaker: str | None = None
    parent_h2: str = ""
    parent_h3: str = ""


def parse_document(text: str) -> list[RawSection]:
    """Split document text into raw sections by structural markers.

    Blank lines and non-marker text lines belong to the current open section
    and are preserved in ``.lines`` so pass 2 can recover paragraph
    boundaries. Box regions are atomic — every line between ``[BOX START:]``
    and ``[BOX END]`` is captured inside a single ``box`` section, unaffected
    by other markers that may appear inside the box body.
    """
    sections: list[RawSection] = []
    current = RawSection(heading="")
    in_box = False
    current_h2 = ""
    current_h3 = ""

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        # Box atomicity first — nothing inside a box is parsed as a marker.
        if in_box:
            current.lines.append(line)
            if BOX_END.match(line):
                in_box = False
                _flush(sections, current)
                current = RawSection(heading="", parent_h2=current_h2, parent_h3=current_h3)
            continue

        if (m := BOX_START.match(line)) is not None:
            _flush(sections, current)
            current = RawSection(
                heading=m.group(1),
                lines=[line],
                section_type="box",
                parent_h2=current_h2,
                parent_h3=current_h3,
            )
            in_box = True
            continue

        if (m := H2_PATTERN.match(line)) is not None:
            _flush(sections, current)
            current_h2 = m.group(1)
            current_h3 = ""
            current = RawSection(
                heading=current_h2,
                lines=[line],
                section_type="h2",
                parent_h2="",
                parent_h3="",
            )
            continue

        if (m := H3_PATTERN.match(line)) is not None:
            _flush(sections, current)
            current_h3 = m.group(1)
            current = RawSection(
                heading=current_h3,
                lines=[line],
                section_type="h3",
                parent_h2=current_h2,
                parent_h3="",
            )
            continue

        if VOTE_PATTERN.match(line):
            _flush(sections, current)
            current = RawSection(
                heading=line,
                lines=[line],
                section_type="vote",
                parent_h2=current_h2,
                parent_h3=current_h3,
            )
            continue

        if (m := MEMBER_PATTERN.match(line)) is not None:
            _flush(sections, current)
            current = RawSection(
                heading=m.group(1),
                lines=[line],
                section_type="member",
                speaker=m.group(1),
                parent_h2=current_h2,
                parent_h3=current_h3,
            )
            continue

        # Numbered paragraph — append to current section and record the number.
        if (m := PARA_PATTERN.match(line)) is not None:
            current.paragraph_numbers.append(int(m.group(1)))

        current.lines.append(line)

    _flush(sections, current)
    return sections


def _flush(sections: list[RawSection], current: RawSection) -> None:
    """Append ``current`` to ``sections`` unless it is empty whitespace."""
    if current.heading or any(line.strip() for line in current.lines):
        sections.append(current)


# ── Pass 2: categorise → merge → split → build chunks ──────


_KEEP_AT_ANY_SIZE = frozenset({"vote", "member", "box"})
_DROP_IF_TOKENS_BELOW = 15


def chunk_document(
    text: str,
    document_type: DocumentType,
    date: str,
    source_url: str,
    title: str,
    doc_id: str | None = None,
    speaker: str | None = None,
) -> list[Chunk]:
    """Build section-aware Chunk objects from a processed document text.

    The pipeline is:
      1. parse_document → list[RawSection]
      2. assign SectionCategory to each
      3. merge small consecutive same-category sections
      4. split sections that exceed size caps (paragraph → token fallback)
      5. apply 50-token overlap within a split non-box section
      6. drop near-empty non-structural chunks
      7. wrap each final piece in a Chunk with full ChunkMetadata

    Args:
        text: Processed document text (scraper output).
        document_type: Enum for the source document.
        date: 'YYYY-MM' from manifest.
        source_url: Source URL from manifest.
        title: Human-readable title from manifest.
        doc_id: Optional document identifier (e.g. 'speech_bailey_2025_02')
            included in each chunk_id so multiple docs of the same type/month
            don't collide (e.g. four Feb-2025 speeches).

    Returns:
        (list[Chunk]) Chunks ready to serialise or embed.
    """
    sections = parse_document(text)
    if not sections:
        return []

    categorised: list[_CategorisedSection] = [
        _CategorisedSection(
            raw=s,
            category=assign_category(s.heading, s.section_type, document_type),
        )
        for s in sections
    ]

    merged = _merge_small_adjacent(categorised)

    chunks: list[Chunk] = []
    speech_seq = 0
    for cs in merged:
        pieces = _split_oversized(cs)
        for local_idx, piece_text in enumerate(pieces):
            # Drop near-empty non-structural chunks (heading-only sections,
            # lonely vote-header lines with no body). Structural markers
            # carry metadata even when the text is tiny.
            if (
                cs.raw.section_type not in _KEEP_AT_ANY_SIZE
                and count_tokens(piece_text) < _DROP_IF_TOKENS_BELOW
            ):
                continue

            if document_type is DocumentType.SPEECH:
                speech_seq += 1
                para_range = str(speech_seq)
            else:
                para_range = _paragraph_range(cs, document_type, local_idx, len(pieces))

            chunk_speaker = _resolve_chunk_speaker(
                cs, document_type=document_type, manifest_speaker=speaker
            )
            metadata = ChunkMetadata(
                document_type=document_type,
                date=date,
                section_category=cs.category,
                speaker=chunk_speaker,
                source_url=source_url,
                paragraph_range=para_range,
                title=title,
            )
            chunk_id = _chunk_id(doc_id, document_type, date, cs.category, len(chunks) + 1)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece_text,
                    metadata=metadata,
                    token_count=count_tokens(piece_text),
                )
            )

    return chunks


@dataclass
class _CategorisedSection:
    """A RawSection with its assigned SectionCategory and cached token count."""

    raw: RawSection
    category: SectionCategory

    @property
    def text(self) -> str:
        return "\n".join(self.raw.lines).strip()

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


_NON_MERGEABLE_CATEGORIES = frozenset(
    {
        SectionCategory.BOX_ANALYSIS,
        SectionCategory.VOTING,
        SectionCategory.INDIVIDUAL_STATEMENT,
    }
)


def _merge_small_adjacent(
    sections: list[_CategorisedSection],
) -> list[_CategorisedSection]:
    """Merge consecutive same-category sections when BOTH are under 150 tokens.

    Prevents one-sentence chunks (e.g. a standalone h3 heading immediately
    followed by a two-line paragraph). Structural-marker categories
    (BOX_ANALYSIS, VOTING, INDIVIDUAL_STATEMENT) are never merged — each
    marker deserves a distinct chunk for metadata filtering to work.
    """
    if not sections:
        return []

    merged: list[_CategorisedSection] = [sections[0]]
    for current in sections[1:]:
        previous = merged[-1]
        if (
            previous.category is current.category
            and previous.category not in _NON_MERGEABLE_CATEGORIES
            and previous.tokens < 150
            and current.tokens < 150
        ):
            combined = RawSection(
                heading=previous.raw.heading or current.raw.heading,
                lines=previous.raw.lines + current.raw.lines,
                section_type=previous.raw.section_type,
                paragraph_numbers=previous.raw.paragraph_numbers
                + current.raw.paragraph_numbers,
                speaker=previous.raw.speaker or current.raw.speaker,
                parent_h2=previous.raw.parent_h2 or current.raw.parent_h2,
                parent_h3=previous.raw.parent_h3 or current.raw.parent_h3,
            )
            merged[-1] = _CategorisedSection(raw=combined, category=previous.category)
        else:
            merged.append(current)
    return merged


def _split_oversized(section: _CategorisedSection) -> list[str]:
    """Split a section into one or more text pieces respecting size limits.

    Rules:
      - Box analyses stay whole below the hard cap (2000 tokens). Over the
        hard cap they split at paragraph boundaries with the ``[BOX START:]``
        marker propagated onto every piece and NO overlap applied.
      - Regular sections split at paragraph boundaries when over the 1200
        max. Monolithic sections (one giant paragraph) fall back to
        token-level splitting so no piece exceeds the cap.
      - Overlap: 50 tokens between consecutive pieces of the SAME non-box
        section (intra-section overlap only).
    """
    text = section.text
    if not text:
        return []

    tokens = section.tokens
    is_box = section.category is SectionCategory.BOX_ANALYSIS
    cap_for_keep_whole = HARD_CAP_TOKENS if is_box else ENHANCED_MAX_CHUNK
    if tokens <= cap_for_keep_whole:
        return [text]

    # Leave head-room for the box-marker prepend (≈20 tokens for the marker
    # line) or the non-box overlap prepend (50 tokens) so the prepend never
    # pushes a piece over ENHANCED_MAX_CHUNK.
    marker_budget = 60 if is_box else ENHANCED_OVERLAP
    split_target = ENHANCED_MAX_CHUNK - marker_budget

    paragraphs = _paragraph_blocks(text)
    if len(paragraphs) > 1:
        pieces = _greedy_pack(paragraphs, target=split_target)
    else:
        pieces = [text]

    # Guarantee: every piece is under the split_target, even monolithic
    # content that paragraph-boundary splitting could not reduce.
    pieces = _enforce_cap(pieces, split_target)

    if is_box:
        return _propagate_box_marker(pieces, section.raw.heading)
    return _apply_overlap(pieces, overlap_tokens=ENHANCED_OVERLAP)


def _enforce_cap(pieces: list[str], max_tokens: int) -> list[str]:
    """Token-split any piece whose tokens still exceed ``max_tokens``.

    Used when paragraph-boundary splitting leaves a monolithic paragraph
    above the cap — falls back to tiktoken-aware recursive splitting.
    """
    out: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            out.append(piece)
            continue
        out.extend(_token_split(piece, max_tokens))
    return out


def _token_split(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks bounded by token count using cl100k_base.

    RecursiveCharacterTextSplitter rounds up on separator boundaries and
    can overshoot the nominal ``chunk_size`` by a few tokens. We apply a
    small safety margin so every returned piece reliably stays at or below
    ``max_tokens``.
    """
    safety_margin = 50
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max(1, max_tokens - safety_margin),
        chunk_overlap=0,
    )
    return splitter.split_text(text)


def _propagate_box_marker(pieces: list[str], heading: str) -> list[str]:
    """Prepend '[BOX START: heading] (continued)' to every piece after the first.

    The first piece already contains the original ``[BOX START: ...]`` line.
    Continuation markers keep the section_category and paragraph_range
    metadata meaningful for downstream consumers.
    """
    if not pieces:
        return pieces
    marker = f"[BOX START: {heading}] (continued)"
    return [pieces[0]] + [
        piece if piece.startswith("[BOX START:") else f"{marker}\n\n{piece}"
        for piece in pieces[1:]
    ]


def _paragraph_blocks(text: str) -> list[str]:
    """Split text on blank-line paragraph boundaries."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b]


def _greedy_pack(paragraphs: list[str], *, target: int) -> list[str]:
    """Greedily pack paragraphs into pieces each <= target tokens.

    Keeps paragraph boundaries intact (never splits mid-paragraph). If a
    single paragraph exceeds ``target``, it becomes its own piece.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if buffer and buffer_tokens + para_tokens > target:
            pieces.append("\n\n".join(buffer))
            buffer = [para]
            buffer_tokens = para_tokens
        else:
            buffer.append(para)
            buffer_tokens += para_tokens

    if buffer:
        pieces.append("\n\n".join(buffer))
    return pieces


def _apply_overlap(pieces: list[str], *, overlap_tokens: int) -> list[str]:
    """Prepend the last ``overlap_tokens`` tokens of piece N to piece N+1."""
    if overlap_tokens <= 0 or len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = _tail_tokens(pieces[i - 1], overlap_tokens)
        if tail:
            out.append(f"{tail}\n\n{pieces[i]}")
        else:
            out.append(pieces[i])
    return out


def _tail_tokens(text: str, n: int) -> str:
    """Return the text corresponding to the last ``n`` tokens of ``text``."""
    from boe_rag.chunking.metadata import _ENCODER  # local import avoids cycle

    tokens = _ENCODER.encode(text)
    if len(tokens) <= n:
        return text
    return _ENCODER.decode(tokens[-n:])


def _paragraph_range(
    section: _CategorisedSection,
    doc_type: DocumentType,
    local_idx: int,
    total_pieces: int,
) -> str:
    """Build the ``paragraph_range`` metadata string per document type."""
    raw = section.raw
    if doc_type is DocumentType.MPC_MINUTES:
        if raw.paragraph_numbers:
            lo, hi = min(raw.paragraph_numbers), max(raw.paragraph_numbers)
            return f"{lo}-{hi}" if lo != hi else str(lo)
        if raw.section_type == "vote":
            return "Vote"
        if raw.section_type == "member" and raw.speaker:
            return raw.speaker
        return raw.heading or raw.parent_h3 or raw.parent_h2

    if doc_type in (DocumentType.MPR, DocumentType.FSR):
        if raw.section_type == "box":
            return _box_label(raw.heading)
        h2_ref = _chapter_ref(raw.parent_h2 or raw.heading)
        h3_ref = _subsection_ref(raw.parent_h3 or raw.heading)
        parts = [p for p in (h2_ref, h3_ref) if p]
        if parts and total_pieces > 1:
            parts.append(f"pt{local_idx + 1}")
        return " ".join(parts) if parts else raw.heading

    return raw.heading


def _box_label(heading: str) -> str:
    """'Box A - Firms' costs' -> 'Box A'. Falls back to the full heading."""
    head = heading.split(":")[0].split("-")[0].strip()
    return head or heading


_CHAPTER_NUM_RE = re.compile(r"^(\d+)\s*:")
_SUBSECTION_NUM_RE = re.compile(r"^(\d+\.\d+)\s*:")


def _chapter_ref(h2: str) -> str:
    """'1: The economic outlook' -> 'Ch.1'. Empty if no leading number."""
    if (m := _CHAPTER_NUM_RE.match(h2)) is not None:
        return f"Ch.{m.group(1)}"
    return ""


def _subsection_ref(h3: str) -> str:
    """'1.1: Inflation' -> 's1.1'. Empty if no dotted number prefix."""
    if (m := _SUBSECTION_NUM_RE.match(h3)) is not None:
        return f"s{m.group(1)}"
    return ""


def _resolve_chunk_speaker(
    cs: _CategorisedSection,
    *,
    document_type: DocumentType,
    manifest_speaker: str | None,
) -> str | None:
    """Return the canonical speaker string (or None) for a chunk.

    Resolution rules:
      * MPC member statements take the speaker from the parsed
        ``**Name:**`` marker, normalised to "FirstName LastName".
      * All other MPC chunks (vote headers, discussion) have no speaker.
      * Every chunk from a SPEECH document inherits the manifest's speaker
        — even sub-sections (``### Where have we come from?``) belong to
        the same speaker. The manifest value is normalised upstream.
      * MPR / FSR chunks never carry a speaker.
    """
    if cs.raw.section_type == "member" and cs.raw.speaker:
        return normalise_speaker(cs.raw.speaker)
    if document_type is DocumentType.SPEECH and manifest_speaker:
        return manifest_speaker
    return None


def _chunk_id(
    doc_id: str | None,
    doc_type: DocumentType,
    date: str,
    category: SectionCategory,
    index: int,
) -> str:
    """Build a chunk id unique within the corpus.

    When ``doc_id`` is provided (e.g. ``'speech_bailey_2025_02'``), it forms
    the prefix so two docs in the same month/category never collide. When
    omitted (test fixtures), falls back to ``{doc_type}_{date}``.
    """
    prefix = doc_id if doc_id else f"{doc_type.value}_{date}"
    return f"{prefix}_{category.value}_{index:03d}"
