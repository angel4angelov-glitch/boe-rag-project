# A5 — Sonnet consistency check on worst-delta queries

## Goal
Rescore 3-5 queries (the ones with the biggest baseline-vs-enhanced
delta) using Claude Sonnet 4 as the judge instead of gpt-4o-mini.
Compare scores. If consistent, methodology paragraph in the report gets
a sentence. If divergent, that's a finding too.

## Why
- **Report — Methodology section**: "We evaluated with GPT-4o-mini to
  avoid self-grading bias, and validated consistency against Claude
  Sonnet 4 on N=5 worst-delta queries. Scores agreed within ±0.08 /
  differed materially in X/5 cases." Either result is publishable.
- **Low-cost hardening**: ~$1, 15 minutes. Cheapest report win
  available.

## Risk: ZERO
Uses the existing `--evaluator-provider anthropic --evaluator-model
claude-sonnet-4-20250514` flag. Writes to a **separate** JSONL so the
headline gpt-4o-mini results aren't overwritten. No code changes.

## Scope
**New files**:
- `data/evaluation_results/ragas_{baseline,enhanced}_sonnet_check.jsonl`
- `scripts/compare_evaluators.py` — compute score-divergence table

**Modified files**: none.

## Steps
1. Identify worst-delta queries from `comparison_table.csv` +
   per-category breakdown. Proposed set:
   - Biggest Faithfulness loss for enhanced: find `q??` with most
     negative per-query delta.
   - Biggest Precision win for enhanced: find `q??` with biggest
     positive delta.
   - q06, q10, q24 — enhanced abstains (interesting edge)
   - q21 — should abstain but didn't
2. Save those qids to a list.
3. Run RAGAS on that subset with Sonnet, writing to suffix path:
   ```bash
   python scripts/run_ragas.py \
       --pipeline both \
       --evaluator-provider anthropic \
       --evaluator-model claude-sonnet-4-20250514 \
       --subset-ids q06,q10,q21,q24,qXX \
       --out-suffix sonnet_check
   ```
   (Requires adding `--subset-ids` + `--out-suffix` flags to
   `run_ragas.py`. ~30 min of CLI work.)
4. Write `compare_evaluators.py`:
   - Load both JSONLs (gpt-4o-mini + Sonnet)
   - Per (qid, metric) row: compute abs diff
   - Output `evaluator_divergence.csv` with columns: qid, metric,
     gpt4o_mini, sonnet, |diff|
   - Print: "Max divergence: 0.X on (qid, metric). Mean abs diff: 0.X."
5. One paragraph in the report's Methodology section.

## Test plan
- CLI flags work (unit test on argparse parsing).
- `compare_evaluators.py` produces the divergence CSV.
- Visual: max divergence < 0.15 is "consistent"; > 0.2 is "material
  disagreement — worth a paragraph".

## Rollback
Delete the new JSONLs and the comparison script. Headline numbers
unchanged.

## Effort: 30-45 minutes ($1 spend)

## Branch: `feat/evaluator-consistency`
