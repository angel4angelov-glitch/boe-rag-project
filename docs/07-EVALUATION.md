# 07 — Evaluation

## Objective
Run both pipelines on the same test set, compute RAGAS metrics, produce comparison tables. This is where you prove the enhanced pipeline is better.

## Depends on
05-BASELINE-PIPELINE, 06-ENHANCED-PIPELINE (both callable)

## Deliverables
- [ ] `src/evaluation/test_set.csv` — 15-20 questions with ground truth
- [ ] `src/evaluation/ragas_eval.py` — RAGAS evaluation runner
- [ ] Comparison table: baseline vs enhanced RAGAS scores
- [ ] CRAG-specific metrics: rewrites triggered, hallucination checks failed
- [ ] All results saved to `data/evaluation_results/`

---

## Test Set Design

### 15-20 queries across 4 categories

**Simple factual (5)** — both pipelines should handle these; delta comes from citation quality
1. What was the MPC vote split in February 2026?
2. What was the Brent crude price cited in the March 2026 MPC minutes?
3. What was the CPI inflation rate cited in the November 2025 minutes?
4. When did the MPC last cut Bank Rate before March 2026?
5. What was the household saving ratio cited in the November 2025 MPR?

**Comparative/temporal (5)** — enhanced should win via metadata filtering + better chunks
6. How did the February 2026 MPR near-term inflation projection differ from November 2025?
7. How did the MPC's language on inflation persistence change between November 2025 and February 2026?
8. Compare the voting patterns across the November 2025, December 2025, and February 2026 meetings.
9. How did the BoE's assessment of global risks evolve between the July 2025 and December 2025 FSRs?
10. What changed in Catherine Mann's policy stance between November 2025 and February 2026?

**Deep context (5)** — enhanced should win decisively; baseline will retrieve wrong chunks
11. What specific consumption weakness scenario did Box D in the November 2025 MPR describe?
12. What structural labour market changes did Clare Lombardelli highlight in November 2025?
13. What risks from US corporate defaults did the December 2025 FSR identify?
14. What was the MPC's assessment of second-round effects from the Middle East energy shock in March 2026?
15. What asymmetric policy risk argument did Lombardelli make in November 2025?

**Edge cases / adversarial (4)** — test system robustness
16. What is the Federal Reserve's view on interest rates? (out of scope — should abstain)
17. What does the BoE think about cryptocurrency regulation? (ambiguous — may retrieve tangential)
18. Summarise the entire November 2025 MPR. (too broad — tests behaviour on vague queries)
19. What was the exact GDP growth figure for Q3 2025 cited on page 23 of the November MPR? (numerical precision)

---

## Ground Truth CSV Schema

```csv
id,category,question,expected_answer,source_document,source_section,source_paragraph,source_quote
1,simple_factual,"What was the MPC vote split in February 2026?","Seven members voted to maintain Bank Rate at 4.5% and two members preferred a reduction of 0.25 percentage points to 4.25%",mpc_2026_02,voting,22,"Seven members voted to maintain..."
```

**Rules for ground truth**:
- `expected_answer`: Complete factual answer using exact language from source
- `source_quote`: Verbatim quote from the source document (for context_recall)
- Every answer must be manually verified against the original BoE document
- For edge cases (16-18): `expected_answer` describes expected system behaviour, not a factual answer

---

## RAGAS Metrics (v0.2+ API)

### Metrics to compute
| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is the answer supported by retrieved context? | Detects hallucination |
| **Answer Relevancy** | Does the answer address the question? | Detects off-topic responses |
| **Context Precision** | Of retrieved chunks, what fraction were useful? | Measures retrieval quality |
| **Context Recall** | Of relevant chunks that exist, what fraction were retrieved? | Measures retrieval completeness |

### RAGAS v0.2+ wiring

```python
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [...],
    "answer": [...],          # pipeline output
    "contexts": [...],        # list of list of strings (retrieved chunks)
    "reference": [...],       # ground truth answer
}
dataset = Dataset.from_dict(eval_data)

# Configure RAGAS to use Claude for evaluation LLM
evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-20250514"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
]

result = evaluate(dataset=dataset, metrics=metrics)
```

**Important**: RAGAS v0.2 renamed columns and classes. Do NOT use the old `from ragas.metrics import faithfulness` lowercase imports — those are v0.1.

---

## CRAG-Specific Metrics (Beyond RAGAS)

Track these from the enhanced pipeline's metadata:

| Metric | Source | What it tells you |
|--------|--------|-------------------|
| Rewrite trigger rate | `crag_rewrites > 0` count / total queries | How often initial retrieval fails |
| Rewrite success rate | Queries where rewrite improved grading | Whether rewriting actually helps |
| Hallucination flag rate | `is_grounded == False` count / total queries | How often generation hallucinates |
| Metadata filter usage rate | `metadata_filters_used != None` / total queries | How often query analysis activates filters |
| Avg chunks after grading | mean of `chunks_after_grading` | How aggressively grading filters |
| Rerank order change | % of queries where top-1 chunk changed after rerank | Whether reranking is doing anything |

---

## Expected Results Shape

### Comparison table (Table 2 of report)
| Metric | Baseline | Enhanced | Delta |
|--------|----------|----------|-------|
| Faithfulness | ~0.65 | ~0.85 | +0.20 |
| Answer Relevancy | ~0.70 | ~0.85 | +0.15 |
| Context Precision | ~0.40 | ~0.75 | +0.35 |
| Context Recall | ~0.50 | ~0.80 | +0.30 |

**Context Precision should show the biggest delta** — this is where metadata filtering and section-aware chunking pay off. Baseline retrieves 5 random-ish chunks; enhanced retrieves 5 highly targeted ones.

**If your delta is <0.10 on any metric**: something is wrong with either the baseline (too good) or the enhanced pipeline (not working). Debug before writing the report.

---

## Per-Category Breakdown

Also compute RAGAS scores per category:

| Category | N | Baseline Faith. | Enhanced Faith. | ... |
|----------|---|-----------------|-----------------|-----|
| Simple factual | 5 | | | |
| Comparative/temporal | 5 | | | |
| Deep context | 5 | | | |
| Edge cases | 4 | | | |

**Expected**: Deep context and comparative queries should show the largest deltas. Simple factual may show small deltas (both pipelines can find "vote split" text).

---

## Output Files

Save everything to `data/evaluation_results/`:
```
data/evaluation_results/
├── baseline_results.json      # per-query: answer, sources, metadata
├── enhanced_results.json      # per-query: answer, sources, metadata
├── ragas_baseline.json        # RAGAS scores per query
├── ragas_enhanced.json        # RAGAS scores per query
├── comparison_table.csv       # side-by-side summary
└── crag_metrics.json          # enhanced-only: rewrite/hallucination stats
```

---

## Acceptance Criteria

1. Test set CSV has 15-20 questions with manually verified ground truth
2. Both pipelines run on all questions without errors
3. RAGAS scores computed for all 4 metrics on both pipelines
4. Enhanced pipeline outperforms baseline on all metrics (if not, debug)
5. CRAG-specific metrics show the correction loop is actually triggering
6. Per-category breakdown shows where the delta is largest
7. All results saved to `data/evaluation_results/`
