# B2 — Expand test set from 25 → 40 queries

## Goal
Add 15 more hand-labelled queries to `data/test_set.csv`, bringing
total to N=40. At N=40 effective ≈ 37 (after abstains), the paired
Wilcoxon crosses into ~75% power for delta=0.15 — currently we're at
~60%.

## Why
- **Statistical credibility**: N=25 is honest but weak. N=40 lets us
  claim "at α=0.05 the enhanced pipeline's Context Precision improvement
  reaches significance" (currently p_holm = 0.38).
- **Per-category balance**: edge_case categories have N=1 each, which
  is anecdote. Five more edge cases makes meaningful per-category
  statistics possible.
- **Depth**: can add more speech-focused questions (6 → 10) to
  stress-test the speaker metadata filtering.

## Risk: LOW code / HIGH time
Zero code changes. Risk is human — hand-labelling 15 good ground-truth
queries requires reading the source documents, extracting exact quotes,
validating categories. Rushed labels = garbage ground truth = garbage
RAGAS scores.

## Scope
**Modified files**:
- `data/test_set.csv` — 15 new rows

**No code changes required.** `test_set_hash` will change; the report's
repro metadata tracks this.

## Steps
1. Sketch 15 candidate queries balanced across categories:
   - 3 more `simple_factual` (hitting different document types)
   - 3 more `comparative` (deliberately multi-document to stress-test
     the filter fix from B1)
   - 5 more `deep_context` (varied topics)
   - 1 more `edge_case_out_of_scope` (e.g. "What did the SNB do?")
   - 1 more `edge_case_ambiguous` (e.g. "climate risk disclosure")
   - 1 more `edge_case_too_broad` (e.g. "Summarise all 2025 speeches")
   - 1 more `edge_case_numerical` (exact figure from a different doc)
2. For each, find the source document in `data/raw/`, extract the
   verbatim quote for `source_quote`, write `expected_answer` using the
   exact BoE language.
3. Rule: every expected_answer must cite a specific quote from a
   specific document. No paraphrasing.
4. Validate categories are consistent with the existing 7 buckets.
5. Re-run: pipelines (~13 min, $2), RAGAS (~7 min, $0.50). Total: ~$2.50.
6. Update report tables with N=40 numbers.

## Test plan
- `python -c "from boe_rag.evaluation import load_test_set; print(len(load_test_set('data/test_set.csv')))"` → 40
- All 203 existing tests pass (test set size isn't hard-coded anywhere).
- Re-run spec 07 aggregate, verify Wilcoxon p-values / CIs change as
  expected (with more N, CIs tighten).
- Spot-check 3 new queries run end-to-end with sensible answers.

## Rollback
`git checkout main -- data/test_set.csv`. Results files for N=25
remain in git history on a tag.

## Guardrail
Tag the current state BEFORE starting:
```bash
git tag -a v0.1-eval-n25 -m "Eval with N=25 test set"
git push origin v0.1-eval-n25
```
So the report can reference either the N=25 or N=40 results cleanly.

## Effort: 3-4 hours labelling + ~25 min re-eval + $2.50 spend

## Branch: `feat/test-set-n40`
