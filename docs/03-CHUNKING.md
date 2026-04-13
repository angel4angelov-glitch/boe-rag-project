# 03 — Section-Aware Chunking

## Objective
Split raw BoE documents into semantically meaningful chunks with rich metadata. This is the **most critical component** — chunk quality determines the evaluation delta between baseline and enhanced pipelines.

## Depends on
- 01-PROJECT-SETUP (`Chunk`, `ChunkMetadata`, `DocumentType`, `SectionCategory` from `boe_rag.models`)
- 02-DATA-INGESTION (processed `.txt` files in `data/raw/`, `manifest.csv`)

## Deliverables
- [ ] `src/boe_rag/chunking/base_chunker.py` — fixed-size naive chunker (baseline)
- [ ] `src/boe_rag/chunking/section_chunker.py` — section-aware chunker (enhanced)
- [ ] `src/boe_rag/chunking/metadata.py` — section category detection logic
- [ ] `src/boe_rag/chunking/validators.py` — chunk QA checks
- [ ] JSON files in `data/chunks/baseline/` and `data/chunks/enhanced/`
- [ ] Validation report in Notebook 01 showing chunk distribution

---

## Why This Is The Hardest Part

The entire thesis of the project is: **section-aware chunking with metadata beats naive fixed-size chunking**. If your chunks are bad, your enhanced pipeline won't outperform baseline, and your evaluation delta collapses.

The Snowflake chunking study (2025) found metadata enrichment produced larger gains than switching retrieval algorithms. This is where your marks come from.

---

## Data Model (from 01-PROJECT-SETUP — do NOT redefine)

The chunker produces `Chunk` objects using the types already defined in `boe_rag.models`:

```python
# Already defined in models.py — imported, not redefined
from boe_rag.models import Chunk, ChunkMetadata, DocumentType, SectionCategory

# Example: MPC chunk with paragraph range
chunk = Chunk(
    chunk_id="MPC_minutes_2025-11_voting_001",
    text="Seven members voted to maintain Bank Rate at 4.75%...",
    metadata=ChunkMetadata(
        document_type=DocumentType.MPC_MINUTES,
        date="2025-11",
        section_category=SectionCategory.VOTING,
        speaker=None,
        source_url="https://www.bankofengland.co.uk/...",
        paragraph_range="19-20",       # MPC: numbered paragraph range
        title="November 2025 MPC Minutes",
    ),
    token_count=312,
)

# Example: MPR chunk with section reference (no paragraph numbers)
chunk = Chunk(
    chunk_id="MPR_2025-11_box_analysis_001",
    text="[BOX START: Box A - Developments in firms' costs...",
    metadata=ChunkMetadata(
        document_type=DocumentType.MPR,
        date="2025-11",
        section_category=SectionCategory.BOX_ANALYSIS,
        speaker=None,
        source_url="https://www.bankofengland.co.uk/...",
        paragraph_range="Box A",       # MPR/FSR: section/box reference
        title="November 2025 Monetary Policy Report",
    ),
    token_count=890,
)
```

`DocumentType` and `SectionCategory` are `StrEnum` — they serialise to strings for ChromaDB but catch typos at definition time.

### `paragraph_range` convention by document type
| Document type | `paragraph_range` value | Example |
|--------------|------------------------|---------|
| MPC minutes | Numbered paragraph range | `"15-18"`, `"22"` |
| MPR | Chapter/section reference | `"Ch.1 s1.1"`, `"Box A"` |
| FSR | Chapter/section reference | `"Ch.4"`, `"Box C"`, `"Annex 2"` |
| Speech | Sequential chunk number | `"1"`, `"2"`, `"3"` |

This field provides traceability back to the source document. For MPC it enables paragraph-level citation; for MPR/FSR it enables section-level citation.

---

## Inputs

### From the scraper (02)
- Processed `.txt` files in `data/raw/` with structural markers
- `manifest.csv` providing `document_type`, `date`, `title`, `source_url` per file

### Structural markers (interface contract from 02)

The chunker parses these markers to detect section boundaries. This is the API.

| Marker | What it means | How the chunker uses it |
|--------|--------------|------------------------|
| `## Heading text` | H2 section/chapter | **Primary split point** for MPR/FSR chapters and MPC major sections |
| `### Sub-heading text` | H3 sub-section | **Secondary split point** within chapters |
| `N: paragraph text` | Numbered MPC paragraph | Paragraph range tracking for metadata |
| `**Name:** text` | MPC member statement | → `SectionCategory.INDIVIDUAL_STATEMENT` + `speaker` field |
| `**Votes to ...**` | Vote grouping header | → `SectionCategory.VOTING` |
| `[BOX START: Box X - title]` | MPR/FSR box start | Keep entire box as single chunk → `SectionCategory.BOX_ANALYSIS` |
| `[BOX END]` | Box end boundary | Box boundary |
| `[CHART: ...]` | Chart placeholder | Preserved in chunk text (not a split point) |
| `[TABLE: ...]` | Table text | Preserved in chunk text (not a split point) |

---

## Token Counting

All token counts use `tiktoken` with the `cl100k_base` encoding (the tokenizer for `text-embedding-3-small`, the embedding model).

```python
import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))
```

Every chunk size limit, min/max threshold, and reported token count uses this function. No word-count approximations.

---

## Section Category Detection Strategy

### Priority order (highest to lowest)

The chunker uses **structural markers first, headings second, paragraph numbers as fallback**. Keywords are NOT used for primary classification — they're too noisy ("inflation" and "GDP" appear throughout MPC minutes, not just in their "home" section).

**1. Structural markers (definitive)**
- `**Votes to ...**` → `VOTING`
- `**Name:** text` → `INDIVIDUAL_STATEMENT` (+ set `speaker`)
- `[BOX START: ...]` → `BOX_ANALYSIS`

**2. Heading text (reliable for MPR/FSR/speeches)**
- `## 1: Current economic conditions` → map chapter number to category
- `### 1.1: Inflation` → `INFLATION`
- `## Financial Stability Report Summary` → `RISK_ASSESSMENT`
- For speeches: use `SPEECH_MAIN` as default, `FORWARD_GUIDANCE` if heading contains "outlook" or "policy"

**3. MPC paragraph grouping (heading-based, NOT hard-coded ranges)**

The original PLAN.md assumed fixed paragraph ranges (1-5 = global, 6-10 = inflation, etc.). This is **unreliable** — paragraph topic boundaries vary between meetings. Instead:

```
Strategy: Use the ### sub-headings in MPC minutes as the PRIMARY grouping signal.

Verified MPC heading structure (from November 2025):
  ### The Committee's discussions     ← policy_discussion (general umbrella)
  ### The immediate policy decision   ← voting + individual_statement
  ### Operational considerations      ← (skip or tag as policy_discussion)

Within "The Committee's discussions", paragraphs are grouped by the
FIRST numbered paragraph that changes topic. But since MPC minutes
have only 2-3 ### headings for 20+ paragraphs, we need sub-grouping.
```

**Sub-grouping within "The Committee's discussions":**
- Parse each numbered paragraph
- Use the heading `### The Committee's discussions` as the section start
- Default category: `POLICY_DISCUSSION`
- Override to `VOTING` or `INDIVIDUAL_STATEMENT` only when structural markers appear
- For paragraph-level topic tags (inflation vs labour market), use a lightweight keyword check as **secondary metadata** but NOT as the split boundary

**Why this is simpler than the original plan:** We don't try to assign every paragraph to a topic category. We let the heading structure drive splitting and use `POLICY_DISCUSSION` as the catch-all for the discussion section. The voting and individual statement sections are detected by structural markers, which are 100% reliable. This sacrifices granularity on the discussion paragraphs but avoids fragile paragraph-range assumptions.

**Impact on evaluation narrative:** This means MPC metadata filtering can distinguish `POLICY_DISCUSSION` vs `VOTING` vs `INDIVIDUAL_STATEMENT` — 3 categories, not 12. The granular categories (`INFLATION`, `LABOUR_MARKET`, `GLOBAL_ECONOMY`, `DEMAND_OUTPUT`) are primarily used for MPR/FSR chapter mappings. This is still a significant win over baseline (zero metadata), and the evaluation queries are designed to test this: "What was the MPC vote?" targets `VOTING`, "What did Lombardelli argue?" targets `INDIVIDUAL_STATEMENT`. Queries that need inflation-specific MPC paragraphs rely on embedding similarity rather than metadata filtering — which is honest and acknowledged.

---

## Chunking Rules by Document Type

### MPC Minutes

```
Input markers: ##, ###, N:, **Name:**, **Votes to...**

1. Split on ## headings:
   - "Monetary Policy Summary" → one chunk (SectionCategory.POLICY_DISCUSSION)
   - "Minutes of the MPC meeting..." → start of minutes body

2. Split on ### headings within minutes:
   - "The Committee's discussions" → start discussion section
   - "The immediate policy decision" → start voting section
   - Other ### headings → split points

3. Within discussion section:
   - Group consecutive paragraphs into chunks of 300-800 tokens
   - Split ONLY at paragraph boundaries (N: markers)
   - Never split mid-paragraph
   - Default category: POLICY_DISCUSSION
   - Track paragraph numbers for metadata (paragraph_range: "2-5")

4. Voting section (detected by **Votes to...** marker):
   - Keep the full vote tally + proposition as one chunk → VOTING
   - Each **Name:** block = one chunk → INDIVIDUAL_STATEMENT (speaker = Name)

5. If any chunk exceeds 800 tokens AND contains multiple paragraphs,
   split at the paragraph boundary closest to the midpoint.
```

### Monetary Policy Reports

```
Input markers: ##, ###, [BOX START/END], [CHART:], [TABLE:]

1. Split on ## chapter headings:
   - Each chapter heading starts a new section
   - Map chapter to category by number:
     Ch.1 → INFLATION (or GLOBAL_ECONOMY depending on MPR structure)
     Ch.2 → DEMAND_OUTPUT
     Ch.3 → POLICY_DISCUSSION (projections)
     (Mapping finalised after inspecting actual MPR chapter titles)

2. Split on ### sub-section headings within chapters:
   - Each ### starts a new chunk
   - If a ### section exceeds 1000 tokens, split at paragraph boundaries

3. Box analyses ([BOX START] to [BOX END]):
   - Keep as ONE chunk regardless of size → BOX_ANALYSIS
   - Box integrity > size limits
   - Even 1500-token boxes stay intact — they're the content baseline
     can't retrieve properly, so they're high-value

4. Charts and tables:
   - [CHART: ...] markers stay in the chunk text as context (not split points)
   - [TABLE: ...] markers + table text stay with surrounding paragraph

5. Target chunk size: 400-800 tokens for regular sections
```

### Financial Stability Reports

```
Same strategy as MPR with FSR-specific category mappings:

- Ch.1 → RISK_ASSESSMENT (global vulnerabilities)
- Ch.2 → FINANCIAL_STABILITY (UK household/corporate debt)
- Ch.3-4 → FINANCIAL_STABILITY (banking, market-based finance)
- Boxes → BOX_ANALYSIS (same rule: keep intact)
- Annexes → split on headings, tag as document_type-appropriate category

(Category mapping finalised after inspecting actual FSR chapter titles)
```

### Speeches

```
Input markers: ##, ###

1. Summary paragraph (first paragraph or hero text) → one chunk, SPEECH_MAIN
2. Split body on ## and ### headings
3. If a section exceeds 600 tokens, split at paragraph boundaries
4. Default category: SPEECH_MAIN
5. All chunks tagged with speaker name (from manifest.csv)
6. Target chunk size: 300-600 tokens
```

---

## Enhanced Chunker: Parsing Algorithm

The enhanced chunker uses a **two-pass line-by-line parser**:

```python
import re
from dataclasses import dataclass

# Regex patterns matching the interface contract from 02
H2_PATTERN      = re.compile(r"^## (.+)$")
H3_PATTERN      = re.compile(r"^### (.+)$")
PARA_PATTERN    = re.compile(r"^(\d+)[:.]\s(.+)")
VOTE_PATTERN    = re.compile(r"^\*\*Votes to .+\*\*$")
MEMBER_PATTERN  = re.compile(r"^\*\*([A-Za-z .'-]+):\*\*\s*(.+)")
BOX_START       = re.compile(r"^\[BOX START: (.+)\]$")
BOX_END         = re.compile(r"^\[BOX END\]$")


@dataclass
class RawSection:
    """Intermediate representation: a section of text with a detected type."""
    heading: str
    lines: list[str]
    section_type: str           # "h2", "h3", "box", "vote", "member", "para"
    paragraph_numbers: list[int]
    speaker: str | None = None


def parse_document(text: str) -> list[RawSection]:
    """Pass 1: Split document into raw sections by structural markers."""
    sections: list[RawSection] = []
    current = RawSection(heading="", lines=[], section_type="text", paragraph_numbers=[])
    in_box = False

    for line in (l.rstrip() for l in text.split("\n")):
        # Box regions: accumulate everything between BOX START and BOX END
        if (m := BOX_START.match(line)):
            _flush(sections, current)
            current = RawSection(heading=m.group(1), lines=[line], section_type="box", paragraph_numbers=[])
            in_box = True
            continue
        if BOX_END.match(line):
            current.lines.append(line)
            in_box = False
            _flush(sections, current)
            current = RawSection(heading="", lines=[], section_type="text", paragraph_numbers=[])
            continue
        if in_box:
            current.lines.append(line)
            continue

        # H2 heading: major split
        if (m := H2_PATTERN.match(line)):
            _flush(sections, current)
            current = RawSection(heading=m.group(1), lines=[], section_type="h2", paragraph_numbers=[])
            continue

        # H3 heading: minor split
        if (m := H3_PATTERN.match(line)):
            _flush(sections, current)
            current = RawSection(heading=m.group(1), lines=[], section_type="h3", paragraph_numbers=[])
            continue

        # Vote grouping header
        if VOTE_PATTERN.match(line):
            _flush(sections, current)
            current = RawSection(heading=line, lines=[line], section_type="vote", paragraph_numbers=[])
            continue

        # Individual member statement
        if (m := MEMBER_PATTERN.match(line)):
            _flush(sections, current)
            current = RawSection(heading=m.group(1), lines=[line], section_type="member",
                                 paragraph_numbers=[], speaker=m.group(1))
            continue

        # Numbered paragraph
        if (m := PARA_PATTERN.match(line)):
            current.paragraph_numbers.append(int(m.group(1)))

        current.lines.append(line)

    _flush(sections, current)
    return sections


def _flush(sections: list[RawSection], current: RawSection) -> None:
    if current.lines or current.heading:
        sections.append(current)
```

**Pass 2** takes the `list[RawSection]` and:
1. **Map category**: Each section → `SectionCategory` using `metadata.assign_category(heading, section_type, doc_type)`
2. **Merge small sections**: Consecutive sections of the same category where BOTH are under 150 tokens → merge into one. This prevents one-sentence chunks while preserving intentional boundaries.
3. **Split oversized sections**:
   - MPC: split at `N:` paragraph markers (tracked in `paragraph_numbers`)
   - MPR/FSR: split at `\n\n` blank-line paragraph boundaries within the section text
   - Speeches: split at `\n\n` blank-line boundaries
   - Split target: closest boundary to the midpoint that keeps both halves above 100 tokens
4. **Apply overlap**: 50-token overlap between consecutive chunks within the same category (see Overlap Behaviour section)
5. **Construct `Chunk` objects**: Combine section text + metadata from manifest + computed `paragraph_range` + `count_tokens()`

This parser handles all four document types — the structural markers are the same (from spec 02). The per-type differences are in:
- Category mapping (Step 1) → lives in `metadata.py`
- Split boundary detection (Step 3) → `N:` markers for MPC, `\n\n` for everything else

---

## Module Responsibilities

| Module | What goes in it |
|--------|----------------|
| `section_chunker.py` | The two-pass parser above, `parse_document()`, section merging/splitting, overlap, `Chunk` construction. The main entry point: `chunk_document(text, manifest_row) -> list[Chunk]` |
| `base_chunker.py` | The baseline `RecursiveCharacterTextSplitter` wrapper. Entry point: `chunk_document_baseline(text, doc_id) -> list[dict]` |
| `metadata.py` | `SectionCategory` assignment logic: heading-to-category mappings per document type, `count_tokens()`, `assign_category(heading, section_type, doc_type) -> SectionCategory` |
| `validators.py` | All 14 validation checks as callable functions. Entry point: `validate_chunks(chunks, original_text) -> ValidationReport` |

---

## Baseline Chunker (Deliberately Naive)

The baseline exists to **lose**. It must be genuinely naive:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# IMPORTANT: from_tiktoken_encoder for TOKEN-based splitting, not character-based.
# RecursiveCharacterTextSplitter(chunk_size=500) would give 500 CHARACTERS ≈ 125 tokens.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,       # 500 tokens
    chunk_overlap=0,      # no overlap
)

chunks = splitter.split_text(document_text)
# No metadata. No section awareness. No structural markers parsed.
```

**What makes it naive:**
- Fixed 500-token chunks with no overlap
- Splits anywhere — mid-sentence, mid-paragraph, mid-box
- Zero metadata (no `document_type`, `date`, `section_category`, `speaker`)
- Structural markers (`##`, `[BOX START]`, etc.) treated as regular text, not boundaries
- The same splitter for all document types

---

## Chunk Size Parameters

| Parameter | Baseline | Enhanced |
|-----------|----------|----------|
| Strategy | Fixed 500-token | Section-aware (heading/marker-based) |
| Token counting | `cl100k_base` | `cl100k_base` |
| Target size | 500 tokens | 300-800 tokens (varies by doc type) |
| Min size | N/A (fixed) | 100 tokens (allow small voting chunks) |
| Max size | 500 tokens | 1200 tokens (boxes can be large) |
| Overlap | 0 | 50 tokens at chunk boundaries (within same section) |
| Split boundaries | Anywhere | Paragraph/heading boundaries only |
| Metadata | None | Full `ChunkMetadata` |

### Overlap behaviour (enhanced only)
- When splitting consecutive paragraphs within the same section, the last 50 tokens of chunk N are duplicated at the start of chunk N+1
- Overlap does NOT cross section boundaries (no overlap between the end of "Committee's discussions" and start of "Immediate policy decision")
- Overlap does NOT apply to box analyses (they are standalone chunks)
- Implementation: after splitting, prepend the tail of the previous chunk's text to the next chunk

---

## Output Format

Each document produces a JSON file. The structure mirrors the `Chunk` dataclass with nested `ChunkMetadata`:

```json
{
  "document": "mpc_2025_11",
  "document_type": "MPC_minutes",
  "date": "2025-11",
  "source_url": "https://www.bankofengland.co.uk/...",
  "title": "November 2025 MPC Minutes",
  "chunks": [
    {
      "chunk_id": "MPC_minutes_2025-11_voting_001",
      "text": "Seven members voted to maintain Bank Rate at 4.75%...",
      "metadata": {
        "document_type": "MPC_minutes",
        "date": "2025-11",
        "section_category": "voting",
        "speaker": null,
        "source_url": "https://...",
        "paragraph_range": "19-20",
        "title": "November 2025 MPC Minutes"
      },
      "token_count": 312
    }
  ],
  "total_chunks": 24,
  "total_tokens": 8400
}
```

Baseline chunks are simpler (no metadata nesting):
```json
{
  "document": "mpc_2025_11",
  "chunks": [
    {
      "chunk_id": "baseline_mpc_2025_11_001",
      "text": "...first 500 tokens...",
      "token_count": 500
    }
  ]
}
```

---

## Validation Checks

Run after chunking every document:

### Distribution checks
1. Print chunks per document — expect 15-30 for MPC, 40-80 for MPR/FSR, 8-20 for speeches
2. Print chunks per `SectionCategory` — `VOTING` and `INDIVIDUAL_STATEMENT` must be non-empty for MPC minutes. `BOX_ANALYSIS` must be non-empty for MPR/FSR.
3. If one document produces <5 chunks or >100 chunks, investigate

### Content checks
4. Sample 3 random chunks from each document type — display in notebook for visual inspection
5. Verify all voting chunks contain "voted" or "Bank Rate"
6. Verify all box analysis chunks contain `[BOX START` or "Box"
7. Verify no chunk is pure boilerplate (navigation, copyright, disclaimers)

### Integrity checks
8. Every `Chunk` has non-null: `chunk_id`, `text`, `metadata.section_category`, `metadata.document_type`, `metadata.date`
9. No duplicate `chunk_id` values across all documents
10. All chunk texts are non-empty and >50 characters
11. Total unique text across all enhanced chunks from a document (after deduplicating overlap regions) should be within 5% of original text length. If >10% gap, content was dropped. Measure by summing token counts and subtracting estimated overlap: `total_tokens - (num_chunks - 1) * 50`.

### Size checks
12. Enhanced chunks: median 300-600 tokens, no chunk >1200 tokens (except boxes, with a hard cap at 2000)
13. Baseline chunks: all chunks within 450-550 token range (RecursiveCharacterTextSplitter doesn't produce exact sizes)
14. Count overlap-adjusted total tokens per document in both pipelines (enhanced total minus `(num_chunks - 1) * 50` for overlap) — should be within 10% of each other (same source text, different splitting)

---

## Known Risks

| Risk | Status | Mitigation |
|------|--------|-----------|
| MPC paragraph numbering inconsistent (colon vs period) | **Verified** — regex `^\d+[:.]\s` handles both | Tested against November 2025 minutes |
| MPR boxes rendered as images not text | **Resolved** — verified boxes are HTML `div.box-highlight` | No action needed |
| MPC paragraph topic boundaries vary between meetings | **Open** — paragraph ranges from PLAN.md are unverified | Use heading-based grouping as primary strategy, not hard-coded ranges |
| Section category detection fails silently | Open | Validation check #2 catches empty categories; default to `POLICY_DISCUSSION` rather than crash |
| MPR chapter-to-category mapping differs between reports | Open | Inspect all 4 MPR chapter titles before finalising the mapping constant |
| FSR chapter-to-category mapping differs between reports | Open | Inspect both FSR chapter titles before finalising the mapping constant |

---

## Acceptance Criteria

1. Every document in `data/raw/` has a corresponding JSON in both `data/chunks/enhanced/` and `data/chunks/baseline/`
2. All 14 validation checks pass
3. Enhanced chunks use `Chunk` and `ChunkMetadata` dataclasses from `boe_rag.models` (with `StrEnum` types, not raw strings)
4. Baseline chunks use `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with 500-token chunks and zero metadata
5. Token counting uses `tiktoken` `cl100k_base` encoding throughout
6. Chunker reads document metadata from `manifest.csv` (not inferred from filenames)
7. No chunk exceeds 2000 tokens (hard cap, even for boxes)
8. Box analyses are never split (each `[BOX START]...[BOX END]` = one chunk)
