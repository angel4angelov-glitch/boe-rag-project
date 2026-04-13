# RAG Assignment Plan — Handoff Document

## Context
MSc Financial Technology, University of Warwick. Module IB9AU0 Generative AI and AI Applications. Individual Assignment 2. 50% weighting. 1500-word report + notebooks + demo log. AI use is REQUIRED and must be disclosed.

## The Idea (One Sentence)
A Corrective RAG system over Bank of England policy analysis documents (MPRs, FSRs, MPC minutes, speeches) that retrieves paragraph-level answers from source documents — something neither web search nor LLM memory can reliably do.

---

## Domain Justification

### Why this domain needs RAG
- LLMs with web search give journalist-level summaries but cannot access deep content (Box analyses on page 47, scenario quantitative assumptions, individual member voting rationales)
- LLMs without web search hallucinate specifics — GPT-4 fabricated December 2025 LDI risks that don't exist in the FSR
- The danger isn't "LLMs can't answer" — it's "LLMs give confidently wrong answers indistinguishable from correct ones"
- RAG grounds every answer in a verifiable source paragraph with citation

### Evidence (Table 1 of report)
- We tested 5 deep questions against Claude and GPT-4 without retrieval
- Claude (with web search) admitted it couldn't extract Box D's quantitative calibration
- GPT-4 hallucinated LDI risks in the December 2025 FSR that weren't there
- Screenshots saved as evidence

### Who would use this
- Rates traders parsing MPC announcements on decision day
- LDI portfolio managers tracking BoE risk assessments affecting gilt yields
- Macro hedge fund analysts tracking shifts in MPC reaction function
- Central bank researchers (BoE itself is building internal tools — ChatDNB, RBA PubCHAT — but nothing is public)

### Novelty
- No public RAG system exists for BoE policy documents (confirmed via GitHub search)
- Central banks are building internal RAG tools (ChatDNB won "Initiative of the Year" 2024) but none are public-facing
- All existing NLP work on central bank text is classification/sentiment, not QA
- The student (Angel) has professional background in LDI and gilt portfolios — reflection will be genuinely domain-informed

---

## Corpus

### Documents to ingest
| Document type | Count | Pages each | Total pages | Source format |
|---|---|---|---|---|
| Monetary Policy Reports | 4 (Q1-Q4 2025 or 2025-2026) | ~80 | ~320 | HTML + PDF |
| Financial Stability Reports | 2 (July 2025, Dec 2025) | ~60 | ~120 | HTML + PDF |
| MPC Minutes | 8 (2025-2026 cycle) | ~20 | ~160 | HTML |
| Speeches (selected) | ~20 | ~10 | ~200 | HTML |
| **Total** | **~34** | | **~800 pages** | |

### Data sourcing
- Primary: HTML versions from bankofengland.co.uk (cleaner than PDFs, structured paragraphs)
- Fallback: PDF versions via PyMuPDF (NOT PyPDF2) for documents without HTML
- MPC minutes URL pattern: `https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/YYYY/month-YYYY`
- MPR URL pattern: `https://www.bankofengland.co.uk/monetary-policy-report/YYYY/month-YYYY`
- FSR URL pattern: `https://www.bankofengland.co.uk/financial-stability-report/YYYY/month-YYYY`

---

## Architecture

### Stack
| Component | Choice | Reason |
|---|---|---|
| Generation + Grading | Claude (via Anthropic API) | Available, high quality, good at following grounding instructions |
| Embeddings | OpenAI text-embedding-3-small OR Cohere embed | Cost-effective, widely used |
| Vector store | ChromaDB | Simple, rubric requires vector store, no infrastructure overhead |
| Reranking | Cohere rerank API | Purpose-built, outperforms general LLMs on reranking |
| Reranking fallback | sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2 | Local, no API dependency |
| Orchestration | LangGraph | Industry standard for CRAG flows, state machine for feedback loops |
| Evaluation | RAGAS | Standard framework, measures faithfulness, relevance, context precision/recall |

### Repos — specific roles

| Repo | URL | Role |
|---|---|---|
| **LangGraph Adaptive RAG (PRIMARY FORK)** | `https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/` | Main codebase to adapt. Has CRAG orchestration: document grading, query rewriting, hallucination checking, ChromaDB, all using current LangGraph APIs. This is the skeleton. |
| **vinodkrane/GenerativeAI-Lab/agentic-rag** | `https://github.com/vinodkrane/GenerativeAI-Lab/tree/main/agentic-rag` | Reference for RAGAS evaluation wiring. March 2026, already has faithfulness/relevance/context precision/recall metrics integrated. Copy the evaluation pattern. |
| **Grecil/Corrective-RAG** | `https://github.com/Grecil/Corrective-RAG` | Clean, minimal CRAG implementation with LangChain + LangGraph. Good for understanding the grading/rewriting flow if the Adaptive RAG tutorial is too complex. Study, don't fork. |
| **0xshre/rag-evaluation** | `https://github.com/0xshre/rag-evaluation` | ChromaDB + RAGAS pipeline reference. Shows how to wire RAGAS metrics into a ChromaDB-based system specifically. Useful for Notebook 3. |

### What to fork vs write from scratch
- **Fork/adapt:** LangGraph Adaptive RAG orchestration (CRAG flow: retrieve → grade → rewrite → generate → hallucination check). This is proven production code — don't rewrite it.
- **Write from scratch (this is YOUR contribution):** BoE document scraper, section-aware chunker with metadata tagging, BoE-specific metadata schema, evaluation test set with ground truth, domain-specific grading prompts tuned for monetary policy language
- **Adapt from vinodkrane/0xshre:** RAGAS evaluation pipeline, embedding/indexing configuration for ChromaDB

---

## Two Pipelines

### Baseline (deliberately naive)
- Fixed 500-token chunks, no overlap, no metadata
- Top-5 cosine similarity retrieval from ChromaDB
- Direct generation from retrieved chunks, no grading, no reranking
- Vanilla prompt: "Answer the question based on the following context: {context}"

### Enhanced (CRAG pipeline)
1. **Section-aware semantic chunking** — split along BoE document structure (voting sections, box analyses, individual member statements stay intact). Each chunk tagged with metadata: `meeting_date`, `document_type`, `section_category`, `speaker` (if applicable)
2. **Metadata-filtered retrieval** — use ChromaDB metadata filters before vector search (e.g., question about voting only searches chunks tagged `section: voting`)
3. **CRAG correction loop** (via LangGraph):
   - Retrieve top-k chunks
   - LLM grades each chunk for relevance (binary: relevant/not relevant)
   - If all irrelevant → rewrite query → re-retrieve (max 1 retry)
   - If relevant → proceed to reranking
4. **Cohere rerank** — reorder relevant chunks so most pertinent is first
5. **Generation** — Claude generates answer grounded in reranked context, with citation to source paragraph
6. **Hallucination check** — LLM verifies answer is supported by retrieved documents (binary: grounded/not grounded)

### Advanced techniques (rubric requires "at least one")
We have three, each addressing a documented failure mode:
1. **Semantic/hierarchical chunking** (rubric list) → addresses structure destruction
2. **Self-refinement loop with query rewriting** (rubric list) → addresses retrieval failure
3. **Re-ranking model** (rubric list) → addresses relevance ordering

---

## Section-Aware Chunking Logic

### MPC Minutes structure (consistent across meetings)
- Paragraphs 1-5: Global economic conditions
- Paragraphs 6-10: Inflation and price developments
- Paragraphs 11-14: Labour market and demand
- Paragraphs 15-18: Policy discussion and strategy
- Paragraphs 19+: Individual member votes and statements
- Pattern to detect voting: regex for "X members preferred to maintain/reduce Bank Rate"
- Pattern to detect individual statements: member names followed by colon

### MPR structure
- Sections clearly headed (Chapter 1: Economic outlook, Chapter 2: Inflation, etc.)
- Box analyses labelled "Box A", "Box B", etc. — keep each box as a single chunk
- Fan chart descriptions and scenario analyses — keep intact

### FSR structure
- Sections on global risks, UK banking resilience, market-based finance, NBFI
- Boxes and annexes — keep as single chunks

### Metadata schema per chunk
```python
{
    "document_type": "MPR" | "FSR" | "MPC_minutes" | "speech",
    "meeting_date": "2025-11",  # or publication date
    "section_category": "voting" | "inflation" | "labour_market" | "global" | "box_analysis" | "individual_statement" | "risk_assessment",
    "speaker": "Clare Lombardelli" | null,  # for individual statements
    "source_url": "https://...",
    "paragraph_numbers": "15-18"  # for traceability
}
```

### Validation
- After chunking, print chunk counts per document and per section category
- Eyeball sample chunks from each document type
- If one document produces 5 chunks and another 50, something broke

---

## Evaluation

### Test set: 15-20 queries across 4 categories

**Simple factual (5):**
1. What was the MPC vote split in February 2026?
2. What was Brent crude price cited in the March 2026 MPC minutes?
3. What was the CPI inflation rate cited in the November 2025 minutes?
4. When did the MPC last cut Bank Rate before March 2026?
5. What was the household saving ratio cited in the November 2025 MPR?

**Comparative/temporal (5):**
6. How did the February 2026 MPR near-term inflation projection differ from November 2025?
7. How did the MPC's language on inflation persistence change between November 2025 and February 2026?
8. Compare the voting patterns across the November 2025, December 2025, and February 2026 meetings.
9. How did the BoE's assessment of global risks evolve between the July 2025 and December 2025 FSRs?
10. What changed in Catherine Mann's policy stance between November 2025 and February 2026?

**Deep context (5):**
11. What specific consumption weakness scenario did Box D in the November 2025 MPR describe?
12. What structural labour market changes did Clare Lombardelli highlight in November 2025?
13. What risks from US corporate defaults did the December 2025 FSR identify?
14. What was the MPC's assessment of second-round effects from the Middle East energy shock in March 2026?
15. What asymmetric policy risk argument did Lombardelli make in November 2025?

**Edge cases / adversarial (4):**
16. What is the Federal Reserve's view on interest rates? (out of scope — should not retrieve BoE docs)
17. What does the BoE think about cryptocurrency regulation? (ambiguous — may retrieve tangential content)
18. Summarise the entire November 2025 MPR. (too broad — tests system behaviour on vague queries)
19. What was the exact GDP growth figure for Q3 2025 cited on page 23 of the November MPR? (numerical precision test)

### Ground truth
- For each question, manually identify the source paragraph(s) in original documents
- Record paragraph number / page / exact quote
- Store as CSV: `question, expected_answer, source_document, source_paragraph, source_quote`
- Reference answers use exact quotes from source documents, not paraphrases

### RAGAS metrics
- **Context precision**: of retrieved chunks, what fraction were relevant?
- **Context recall**: of all relevant chunks that exist, what fraction were retrieved?
- **Faithfulness**: is the generated answer supported by the retrieved context?
- **Answer relevance**: does the answer actually address the question?

### Comparison
- Run all queries through baseline pipeline → record RAGAS scores
- Run all queries through enhanced pipeline → record RAGAS scores
- Present as side-by-side table
- Also record: number of CRAG rewrites triggered, number of hallucination checks failed
- Design baseline to be genuinely naive (fixed chunks, no metadata, no grading) so delta is real

---

## Report Structure (1500 words, strict)

| Section | Words | Content |
|---|---|---|
| Domain justification | 200 | Why BoE policy docs need RAG. Table 1 showing LLM failures. Cite ChatDNB, RBA PubCHAT as internal precedents. |
| System design | 300 | One integrated pipeline, three techniques. Each tied to a documented failure mode (structure destruction, retrieval failure, relevance ordering). Reference PageIndex structural insight. |
| Evaluation results | 300 | Tables comparing baseline vs enhanced RAGAS scores. Interpret the delta. Highlight where CRAG loop triggered and whether rewrites improved results. |
| Failure analysis | 300 | Where the system still breaks: numerical extraction, cross-document temporal queries, chart/table content. Connect to literature (FinanceBench findings, FinSheet-Bench). |
| Future improvements | 200 | Finance-specific embeddings (Fin-E5), temporal-aware retrieval (TMMHybridRAG), multi-document tracking across meetings, PageIndex-style tree navigation. |
| Reflection | 200 | Domain-informed: why getting MPC language wrong matters for rates positioning. Why "services price inflation persistence" vs "underlying inflationary pressure" is a meaningful distinction. Connect to LDI and gilt portfolio management. |

---

## Demo Log Structure (5-8 examples)

1. **Baseline fails, enhanced succeeds** — deep MPR box analysis question
2. **Both succeed** — simple vote split question (shows baseline isn't useless)
3. **CRAG loop triggers** — ambiguous question where first retrieval was wrong, rewrite fixed it
4. **Both fail** — numerical precision question (honest about limitations)
5. **Edge case** — out-of-scope question handled gracefully
6. **Enhanced succeeds with reranking impact** — show retrieval order before/after rerank

Each example: query → retrieved chunks (summarised) → generated answer → 2-3 sentences commentary connecting to design decisions.

---

## Notebook Structure (3 notebooks, not 5)

### Notebook 1: Data Ingestion & Indexing
- Scrape BoE HTML documents
- Section-aware chunking with metadata tagging
- Validation: print chunk counts, display sample chunks
- Embed chunks
- Store in ChromaDB with metadata

### Notebook 2: Baseline & Enhanced Pipelines
- Baseline: naive fixed-chunk retrieval → generate
- Enhanced: CRAG flow via LangGraph (retrieve → grade → rewrite → rerank → generate → hallucination check)
- Both pipelines callable with same interface for fair comparison

### Notebook 3: Evaluation
- Load test questions + ground truth from CSV
- Run both pipelines on all questions
- Compute RAGAS scores
- Generate comparison tables and analysis
- Log CRAG-specific metrics (rewrites triggered, hallucination checks)

---

## Key Citations

### RAG techniques
- CRAG: Yan et al. (2024), arXiv:2401.15884
- Self-RAG: Asai et al. (2023), arXiv:2310.11511
- RAG survey: Gao et al. (2024), arXiv:2312.10997

### Central bank NLP
- CB-LMs: Gambacorta et al. (2024), BIS Working Papers No 1215
- IMF WP 2025/109: Silva, Moriya & Veyrune — largest LLM analysis of central bank communication
- CentralBankRoBERTa: Pfeifer & Marohl (2023), Journal of Finance and Data Science

### Financial RAG
- PageIndex: Zhang & Tang (2025) — vectorless reasoning-based RAG, 98.7% on FinanceBench
- FinSage: Wang et al. (2025), arXiv:2504.14493 — multi-path retrieval with DPO-trained reranker
- FinanceBench: Islam et al. (2023), arXiv:2311.11944 — shared vector store RAG scores 19%
- Snowflake chunking study (2025) — metadata enrichment larger gain than switching retrieval algorithms

### Central bank RAG systems (internal)
- ChatDNB: De Nederlandsche Bank, Central Banking "Initiative of the Year" 2024
- RBA PubCHAT: Reserve Bank of Australia, presented at ECONDAT 2025
- RAISE: Banca d'Italia, AML/CFT compliance RAG
- Fed Philadelphia: "Locally Hosted RAG Systems for Economists", ECONDAT 2025

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| LLMs already answer the questions | Tested — they hallucinate or give incomplete answers. Evidence saved. |
| PDF parsing nightmare | Use HTML versions from BoE website, not PDFs |
| CRAG loop never triggers | Include adversarial queries designed to cause retrieval failure |
| Cohere API down | Local cross-encoder fallback (ms-marco-MiniLM-L-6-v2) |
| Small evaluation delta | Baseline deliberately naive; queries target specific failure modes |
| Another student picks same domain | Our corpus includes deep MPR/FSR content, not just minutes |
| Report too descriptive | Word budget enforced; notebooks contain implementation, report contains thinking |
| Ground truth quality | Reference answers use exact quotes from source documents with page/paragraph numbers |
| Section-aware chunking breaks on edge cases | Validate parser against every document before running pipeline |

---

## AI Disclosure (required by assignment)

"AI tools were used throughout this project as required by the assignment brief. Specifically:
- Claude (Anthropic) was used for code generation, debugging, and code review across all notebooks
- The RAG pipeline orchestration was adapted from LangGraph's Adaptive RAG reference implementation
- Claude served as the generation LLM within the RAG pipeline itself
- The section-aware chunking logic, BoE document parser, evaluation test set design, and all analytical interpretation are original work
- All design decisions, domain justification, and reflection draw on the student's professional experience in institutional fixed income and LDI portfolio management"

---

## First Steps (in order)
1. Create project directory structure
2. Scrape BoE HTML documents into `data/` folder
3. Build and validate section-aware chunker
4. Index into ChromaDB
5. Wire up baseline pipeline
6. Wire up enhanced CRAG pipeline with LangGraph
7. Write test questions CSV with ground truth
8. Run RAGAS evaluation
9. Write report (1500 words)
10. Build demo log
11. Package as student_number.zip
