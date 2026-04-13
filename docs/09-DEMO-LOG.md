# 09 — Demo Log

## Objective
5-8 annotated query examples showing the system in action. Each example demonstrates a specific design decision or capability.

## Depends on
05-BASELINE-PIPELINE, 06-ENHANCED-PIPELINE (both working)

## Deliverables
- [ ] Demo log document (PDF, Word, or within a notebook)
- [ ] 5-8 examples with query → retrieval → answer → commentary

---

## Required Examples

### Example 1: Baseline fails, enhanced succeeds
**Query**: "What specific consumption weakness scenario did Box D in the November 2025 MPR describe?"
**Purpose**: Demonstrates section-aware chunking + metadata filtering. Baseline retrieves random MPR chunks; enhanced filters to `section_category: box_analysis` and retrieves Box D intact.
**Show**: Baseline chunks (wrong content) vs enhanced chunks (Box D text). Answer quality difference.

### Example 2: Both succeed (baseline isn't useless)
**Query**: "What was the MPC vote split in February 2026?"
**Purpose**: Simple factual query. Vote text appears in many chunks. Shows baseline works for easy queries — the enhanced pipeline's value is on harder ones.
**Show**: Both answers correct. Note enhanced adds citation to specific paragraph.

### Example 3: CRAG loop triggers
**Query**: "What structural labour market changes did Clare Lombardelli highlight in November 2025?"
**Purpose**: First retrieval may miss Lombardelli-specific chunks. Grading flags irrelevance, query is rewritten, re-retrieval succeeds.
**Show**: `pipeline_trace` showing the rewrite path. Retrieved chunks before and after rewrite.

### Example 4: Both fail (honest about limitations)
**Query**: "What was the exact GDP growth figure for Q3 2025 cited on page 23 of the November MPR?"
**Purpose**: Numerical precision + page-specific retrieval. RAG systems struggle with exact numbers and page references.
**Show**: Both answers approximate or wrong. Commentary on why (embedding similarity doesn't capture numerical precision, chunks don't preserve page numbers).

### Example 5: Out-of-scope handled gracefully
**Query**: "What is the Federal Reserve's view on interest rates?"
**Purpose**: No BoE documents should be relevant. Enhanced pipeline's grading should flag all chunks as irrelevant.
**Show**: Enhanced pipeline detects no relevant documents, returns "This question is outside the scope of the BoE document corpus." Baseline returns a confused answer mixing BoE and Fed content.

### Example 6: Reranking impact visible
**Query**: "How did the BoE's assessment of global risks evolve between the July 2025 and December 2025 FSRs?"
**Purpose**: Show document order before and after Cohere rerank.
**Show**: Top-5 chunks before rerank (generic FSR paragraphs) vs after rerank (specific global risk assessment sections moved to top). Answer quality improvement.

### Examples 7-8: Optional extras
- Hallucination check catching an unsupported claim
- Metadata filter narrowing search to speeches by a specific MPC member

---

## Format Per Example

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE N: [Title]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUERY: [exact query string]

BASELINE:
  Retrieved chunks (summarised):
    1. [chunk_id] — [first 100 chars]...
    2. ...
  Answer: [full baseline answer]

ENHANCED:
  Pipeline trace: analyze_query → retrieve → grade → rerank → generate → hallucination_check
  Metadata filters: {document_type: "MPR", section_category: "box_analysis"}
  Retrieved chunks (summarised):
    1. [chunk_id] — [first 100 chars]... (rerank score: 0.94)
    2. ...
  CRAG rewrites: 0
  Hallucination check: grounded
  Answer: [full enhanced answer]

COMMENTARY (2-3 sentences):
[Connect to design decisions. Why did enhanced succeed where baseline
failed? Which technique made the difference?]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Acceptance Criteria

1. At least 6 examples present (5 required + 1 extra)
2. Each example has: query, baseline output, enhanced output, commentary
3. At least one example shows CRAG rewrite triggering
4. At least one example shows both pipelines failing (honesty)
5. At least one example shows reranking changing document order
6. Commentary connects each example to a specific design decision
