# B1 — Fix q21 abstain failure

## Goal
Flip `abstain_correctness` from 0/3 to 1/3 (or better). Specifically,
make the enhanced pipeline abstain on q21 ("What is the Federal
Reserve's view on interest rates?") which is out-of-corpus.

## Why
- **The single biggest narrative landmine** in the current results.
  If the report says "we implemented a corrective RAG with abstain
  behaviour" but the eval shows `abstain_correctness = 0.0`, the
  examiner will circle it.
- **With the fix**: the Methodology section gets to say "we added an
  explicit out-of-corpus scope check in `analyze_query`, improving
  abstain correctness from 0/3 to 1/3."
- **Even if the fix turns q21 into a correct abstain but breaks
  another query**: the improvement is still reportable.

## Risk: MEDIUM
Modifies `ANALYZE_QUERY_PROMPT` (used by every query) or adds a new
scope-check node. Changes to the analyze_query prompt have already
caused regressions during spec 06 iteration — the fix needs TDD + full
re-eval to verify.

## Root cause hypothesis
`ANALYZE_QUERY_PROMPT` currently extracts filters but has no
"is this about the Bank of England?" gate. Claude happily extracts
filters for a Fed question, the pipeline retrieves tangential BoE
chunks (probably from FSR discussing Fed), generates an answer that
cites those chunks. Plausible → `is_grounded=True`.

## Two fix options

### Option A: Add an out-of-corpus field to `QueryFilters`
```python
class QueryFilters(BaseModel):
    ...
    out_of_corpus: bool = Field(
        default=False,
        description=(
            "True if the question is about a topic OUTSIDE the Bank of "
            "England document corpus (e.g. Federal Reserve, ECB, crypto "
            "regulation, US Treasury). The corpus contains ONLY BoE MPRs, "
            "FSRs, MPC minutes, and MPC member speeches from Nov 2024 "
            "onward."
        ),
    )
```
Then in `analyze_query` node:
```python
if filters.out_of_corpus:
    return {"answer": ABSTAIN_MESSAGE,
            "pipeline_trace": trace + ["abstain_out_of_corpus"]}
```
Route `analyze_query → abstain` conditionally.

**Pros**: single-node change. No new graph nodes.
**Cons**: couples routing to the Pydantic output.

### Option B: Add a dedicated `check_corpus_scope` node
Runs after `analyze_query`. Separate LLM call asking "is this question
answerable from BoE documents?". Routes to abstain if no.

**Pros**: cleaner separation, unit-testable in isolation.
**Cons**: one extra LLM call per query (~$0.002 × 25 = negligible).

**Recommendation: Option A.** Low cost, tight scope.

## Scope (Option A)
**Modified files**:
- `src/boe_rag/pipelines/nodes.py` — extend `QueryFilters`, update
  analyze_query node to emit `abstain` when `out_of_corpus=True`.
- `src/boe_rag/pipelines/prompts.py` — add an out-of-corpus rule to
  `ANALYZE_QUERY_PROMPT`.
- `src/boe_rag/pipelines/enhanced.py` — conditional edge from
  `analyze_query` to `abstain`.
- `tests/pipelines/test_nodes.py` — new tests for the scope field.

## Steps
1. **Red test first**: write a unit test asserting that a Fed question
   through `analyze_query` produces `out_of_corpus=True` and the node
   emits `{"answer": ABSTAIN_MESSAGE}`.
2. Extend `QueryFilters` with the field.
3. Update `ANALYZE_QUERY_PROMPT` with a new rule:
   > 0. First, determine whether the question is answerable from BoE
   >    documents. If the question asks about the Federal Reserve, ECB,
   >    other central banks, US Treasury, cryptocurrency, or topics
   >    outside UK monetary/financial policy, set out_of_corpus=true
   >    and omit all other filters.
4. Update `analyze_query` node: if `out_of_corpus` is True, return the
   abstain message directly in state and append `abstain_out_of_corpus`
   to trace.
5. Add conditional edge in the graph: `analyze_query →
   retrieve` (default) OR `analyze_query → abstain` (if
   out_of_corpus).
6. **Full test suite**: 203 → likely 204-205 tests, all green.
7. **Re-run pipelines**: `python scripts/run_eval.py` (~13 min, ~$2).
8. **Re-run RAGAS**: `python scripts/run_ragas.py --pipeline enhanced
   --no-resume` (only enhanced; baseline unchanged) (~7 min, ~$0.30).
9. Verify `crag_metrics.json → abstain_correctness` moved from 0.0 →
   something positive.
10. If a previously-answering query now abstains incorrectly, debug the
    prompt; don't ship a regression.

## Test plan
- Unit: scope prompt returns `out_of_corpus=True` on Fed question.
- Unit: scope prompt returns `out_of_corpus=False` on q01 (vote split).
- Integration: full enhanced pipeline on q21 → abstain, trace contains
  `abstain_out_of_corpus`.
- Regression: all 24 non-q21 queries produce same `is_abstain` as
  before. The only acceptable change: q21 becomes an abstain. Anything
  else flagged in the diff.

## Rollback
`git revert` the feat/abstain-scope-check commit. All changes are in
3-4 files, no data migrations.

## Effort: 2-3 hours coding + testing, ~$2.30 in re-eval spend

## Branch: `feat/abstain-scope-check`
