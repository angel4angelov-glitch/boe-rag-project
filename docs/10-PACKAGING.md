# 10 — Packaging & Submission

## Objective
Package everything into `student_number.zip` per assignment requirements. Clean, professional, nothing missing, nothing extra.

## Depends on
All previous specs completed.

## Deliverables
- [ ] Final zip file ready for submission
- [ ] AI disclosure statement included
- [ ] All notebooks run cleanly from top to bottom

---

## Submission Structure

```
student_number.zip
├── notebooks/
│   ├── 01_data_ingestion.ipynb    # Notebook 1: scrape + chunk + index
│   ├── 02_pipelines.ipynb         # Notebook 2: baseline + enhanced
│   └── 03_evaluation.ipynb        # Notebook 3: RAGAS comparison
├── src/                            # All source modules
│   ├── scraper/
│   ├── chunking/
│   ├── indexing/
│   ├── pipelines/
│   └── evaluation/
├── data/
│   ├── raw/                        # Scraped documents (or sample subset)
│   ├── chunks/                     # Processed chunks
│   └── evaluation_results/         # RAGAS scores, comparison tables
├── report.pdf                      # 1500-word report
├── demo_log.pdf                    # 5-8 annotated examples
├── requirements.txt                # Pinned dependencies
├── .env.example                    # API key template (no actual keys!)
├── README.md                       # Brief: what it is, how to run it
└── ai_disclosure.md                # Required AI usage statement
```

---

## Pre-Submission Checklist

### Notebooks
- [ ] Each notebook runs top-to-bottom without errors (Kernel → Restart & Run All)
- [ ] Outputs are visible (don't clear outputs before submission — markers want to see results)
- [ ] No hardcoded API keys anywhere in notebook cells
- [ ] Notebook 3 shows the comparison tables and RAGAS scores inline

### Code
- [ ] All imports resolve
- [ ] No absolute paths (use relative paths or pathlib)
- [ ] `.env.example` has all required keys listed with placeholder values
- [ ] No `__pycache__/`, `.ipynb_checkpoints/`, or `.env` in the zip

### Report
- [ ] Word count: 1450-1550
- [ ] Tables populated with actual results (not placeholders)
- [ ] AI disclosure statement present
- [ ] Saved as PDF

### Demo Log
- [ ] At least 6 examples
- [ ] Each has query → baseline → enhanced → commentary
- [ ] Saved as PDF

### Data
- [ ] `evaluation_results/` included with actual RAGAS scores
- [ ] `chunks/` included (so markers can inspect chunk quality)
- [ ] If raw data is too large (>50MB), include a representative sample + the scraper code
- [ ] `manifest.csv` included

---

## AI Disclosure Statement

```markdown
# AI Disclosure

AI tools were used throughout this project as required by the assignment brief.

**Tools used:**
- Claude (Anthropic): Code generation, debugging, code review across all notebooks
- Claude Sonnet: Generation LLM within the RAG pipeline and RAGAS evaluation LLM
- OpenAI text-embedding-3-small: Document embedding
- Cohere rerank-v3.5: Document reranking

**Adapted from external sources:**
- LangGraph Adaptive RAG tutorial: CRAG orchestration skeleton (graph structure, node pattern)
- RAGAS documentation: Evaluation metric wiring

**Original work:**
- BoE document scraper and section-aware chunking logic
- BoE-specific metadata schema and section category taxonomy
- Domain-tuned prompts for document grading and generation
- Evaluation test set with manually verified ground truth
- All analytical interpretation, domain justification, and reflection

**Student background:**
All design decisions, domain justification, and reflection draw on the student's
professional experience in institutional fixed income and LDI portfolio management.
```

---

## README.md (Brief)

```markdown
# BoE Policy RAG System

Corrective RAG system over Bank of England policy documents (MPRs, FSRs, MPC minutes, speeches).

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in API keys
3. Run notebooks in order: 01 → 02 → 03

## Structure
- `notebooks/` — 3 Jupyter notebooks (ingestion, pipelines, evaluation)
- `src/` — Source modules
- `data/` — Scraped documents, chunks, evaluation results
- `report.pdf` — 1500-word report
- `demo_log.pdf` — Annotated demo examples

## Requirements
- Python 3.11+
- API keys: Anthropic, OpenAI, Cohere
```

---

## Zip Command

```bash
cd /path/to/project
zip -r student_number.zip \
  notebooks/ src/ data/ \
  report.pdf demo_log.pdf \
  requirements.txt .env.example README.md ai_disclosure.md \
  -x "*.pyc" -x "__pycache__/*" -x ".ipynb_checkpoints/*" -x ".env" -x "chroma_db/*"
```

---

## Acceptance Criteria

1. Zip file contains all items in the structure above
2. No API keys or secrets in any file
3. No `__pycache__`, `.env`, or `.ipynb_checkpoints` in zip
4. Notebooks run cleanly with visible outputs
5. Zip size < 100MB (exclude large data files if needed)
6. AI disclosure is present and accurate
