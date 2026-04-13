# 08 — Report (1500 Words)

## Objective
1500-word report. Strict word limit. Every sentence must earn its place.

## Depends on
07-EVALUATION (RAGAS results and comparison tables)

## Deliverables
- [ ] Report document (PDF or Word, per submission requirements)
- [ ] Exactly 1500 words (±50)

---

## Structure and Word Budget

| Section | Words | Purpose | What to write |
|---------|-------|---------|---------------|
| **1. Domain Justification** | 200 | Why this domain needs RAG | Table 1: LLM failures on BoE questions. Evidence that Claude/GPT-4 hallucinate or miss deep content. Cite ChatDNB, RBA PubCHAT as precedent. The argument: "LLMs give confidently wrong answers — RAG grounds every claim in a verifiable source." |
| **2. System Design** | 300 | What you built and why | One integrated pipeline diagram. Three techniques, each tied to a failure mode: (1) section-aware chunking → fixes structure destruction, (2) CRAG grading + rewrite → fixes retrieval failure, (3) Cohere rerank → fixes relevance ordering. Reference Snowflake finding on metadata > algorithm changes. |
| **3. Evaluation Results** | 300 | Proof it works | Table 2: baseline vs enhanced RAGAS scores. Interpret the delta — where is it biggest and why. Per-category breakdown. Highlight queries where CRAG loop triggered and whether rewrites improved results. |
| **4. Failure Analysis** | 300 | Honest about limitations | Where the system still breaks: numerical extraction, cross-document temporal reasoning, chart/table content. Connect to literature: FinanceBench shared vector RAG = 19%, PageIndex = 98.7%. Acknowledge the gap. |
| **5. Future Improvements** | 200 | What's next | Finance-specific embeddings (Fin-E5), temporal-aware retrieval (TMMHybridRAG), multi-document tracking, PageIndex-style tree navigation. Each grounded in a specific limitation from Section 4. |
| **6. Reflection** | 200 | Domain-informed insight | Why getting MPC language wrong matters for rates positioning. "Services price inflation persistence" vs "underlying inflationary pressure" is a meaningful distinction for gilt traders. Connect to LDI and portfolio management experience. |

---

## Writing Rules

1. **No filler.** Every sentence must contain information. Delete "In this report we describe..." and similar throat-clearing.
2. **Tables count toward word limit** — use them strategically, they're dense.
3. **One pipeline diagram** — reference it in System Design, don't describe every node in prose.
4. **Cite properly** — CRAG (Yan et al., 2024), Self-RAG (Asai et al., 2023), FinanceBench (Islam et al., 2023), Snowflake (2025), PageIndex (Zhang & Tang, 2025).
5. **Numbers, not adjectives.** "Context precision improved from 0.40 to 0.75" not "retrieval quality improved significantly."
6. **The report contains thinking; the notebooks contain implementation.** Don't describe code in the report.
7. **Section 4 (Failure Analysis) is where marks are.** Markers know enhanced > baseline. They want to see you understand WHY it fails and WHERE.

---

## Table 1: LLM Baseline Failures (Section 1)

| Question | Claude (web search) | GPT-4 (no retrieval) | RAG system |
|----------|--------------------|--------------------|------------|
| Box D consumption scenario | "I cannot access the specific Box D content" | Hallucinated content | Correct, with citation |
| Dec 2025 FSR LDI risks | Partial | Fabricated LDI risks not in FSR | Correct, with citation |
| MPC vote split Feb 2026 | Correct (from news) | Approximate | Exact, with paragraph ref |
| ... | ... | ... | ... |

Populate with actual test results. Screenshots of LLM failures saved as evidence.

---

## Table 2: RAGAS Comparison (Section 3)

| Metric | Baseline | Enhanced | Δ |
|--------|----------|----------|---|
| Faithfulness | | | |
| Answer Relevancy | | | |
| Context Precision | | | |
| Context Recall | | | |

Populate from 07-EVALUATION results.

---

## Key Citations to Include

- CRAG: Yan et al. (2024), arXiv:2401.15884
- Self-RAG: Asai et al. (2023), arXiv:2310.11511
- RAG survey: Gao et al. (2024), arXiv:2312.10997
- CB-LMs: Gambacorta et al. (2024), BIS WP 1215
- FinanceBench: Islam et al. (2023), arXiv:2311.11944
- PageIndex: Zhang & Tang (2025)
- Snowflake chunking study (2025)
- ChatDNB: De Nederlandsche Bank, "Initiative of the Year" 2024
- RBA PubCHAT: Reserve Bank of Australia, ECONDAT 2025

---

## Acceptance Criteria

1. Word count: 1450-1550 words
2. All 6 sections present with correct word budgets (±30 per section)
3. Table 1 populated with actual LLM failure evidence
4. Table 2 populated with actual RAGAS scores from evaluation
5. At least 8 citations from the key citations list
6. No code in the report — only design decisions and results
7. AI disclosure statement included (required by assignment)
