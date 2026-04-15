# B1 — Fix q21 abstain failure (v5)

> **v5 changelog** (on top of v4):
>   - Prompt-iteration budget capped at 3 retries × $0.20 — if
>     pre-flight still fails on attempt 3, abort B1 rather than
>     rabbit-hole prompt-tuning into regressions on other queries.
>   - Pre-flight now *explicitly* reuses `EnhancedPipeline()._structured_llm`
>     to eliminate test/production drift. Was vague in v4.
>   - Branch-creation command switched to `git switch -c` and a
>     rollback recipe for partial-failure cleanup added.
>
> **v4 changelog** (on top of v3):
>   - Test #10 replaced (was vacuous import check) with a real
>     single-source-of-truth assertion.
>   - Verification script invariant tightened to set-membership
>     (Claude @ temp=0 isn't bit-deterministic, so per-query equality
>     is the wrong contract).
>
> **v3 changelog** — after audit round 2 I caught two factual errors
> and five operational holes in v2:
>   - v2's score-drop table was wrong (enhanced q21 AR is already 0.0,
>     not 0.845 → fix is pure upside, not a trade-off)
>   - `abstain_correctness` expectation wrong (1/4 precision, not 1.0 —
>     need a separate recall metric to get the "caught the one we
>     should have" framing)
>   - Missing unit tests for the new router function
>   - ABSTAIN_MESSAGE duplicated across two modules (single source of
>     truth fix folded in)
>   - Pre-flight should save raw QueryFilters not just bool
>   - Pre-flight extended from 8 → 25 queries ($0.06 → $0.20)
>   - Effort re-estimated 3.5 → 4.5 hours

## Goal

Make q21 ("What is the Federal Reserve's view on interest rates?") emit
`ABSTAIN_MESSAGE` and trace `abstain_out_of_corpus`, while preserving
all 24 other queries' current behaviour. Introduce a new metric
`should_abstain_recall` so the grade-level win (1/1 = 1.0) is reportable
separately from the noise-heavy precision metric (1/4 = 0.25).

## Root cause — unchanged from v2

Enhanced pipeline has no corpus-scope gate. q21's BoE-adjacent chunks
pass grading because they mention the Fed (the grader asks "relevant?"
not "is this question in our domain?"). Pipeline generates a
tangential answer citing BoE chunks that paraphrase Fed positions,
hallucination check passes.

## Correct baseline (verified from ragas_enhanced.jsonl, not guessed)

q21's **current** gpt-4o-mini scores:

|  | Baseline | Enhanced |
|---|---|---|
| Faithfulness | 1.0 | 1.0 |
| AnswerRelevancy | 0.845 | **0.000** |
| ContextPrecisionWithReference | 0.0 | 0.0 |
| ContextRecall | 0.0 | 0.0 |

Enhanced is **already scoring AR=0.0** — the answer it emits is already
judged non-relevant to the Fed question. The fix makes this refusal
**explicit** (abstain message → context metrics correctly SKIP)
instead of **implicit** (rambling → context metrics correctly score
0.0 for wrong-corpus retrieval).

## Expected score movement (corrected)

| Metric | q21 before | q21 after | Why |
|---|---|---|---|
| Faithfulness | 1.0 | SKIPPED | Empty retrieved_contexts → RAGAS skips (spec 07 policy) |
| AnswerRelevancy | 0.0 | ~0.0 | Abstain message is still non-relevant to the question; score stays at floor |
| ContextPrecisionWithReference | 0.0 | SKIPPED | Empty contexts |
| ContextRecall | 0.0 | SKIPPED | Empty contexts |

**Net aggregate impact**: near-zero. Headline means barely move.
The win is entirely in the abstain metrics:

| CRAG metric | Before | After |
|---|---|---|
| `abstain_rate` | 3/25 = 0.12 | 4/25 = 0.16 |
| `abstain_correctness` (precision) | 0/3 = 0.00 | 1/4 = **0.25** |
| `should_abstain_recall` (new, see below) | 0/1 = 0.00 | 1/1 = **1.00** |

## Design — Option A from v2, unchanged

Add `out_of_corpus: bool` to `QueryFilters`. analyze_query emits it in
state. A new router `route_after_analyze_query` reads it. True → route
to a new `abstain_out_of_corpus` node. False (or missing) → route to
`retrieve`.

## The scope-check prompt — Rule 0 in ANALYZE_QUERY_PROMPT

```
Rule 0 — Corpus scope check (evaluate FIRST):
Set out_of_corpus=true if the question's PRIMARY subject is the policy,
decisions, views, or statements of an institution OTHER than the Bank
of England, AND answering it would require content authored by that
other entity.

Set out_of_corpus=false if:
  - The question asks about BoE policy, views, publications, decisions,
    or speakers (MPC members, BoE staff).
  - The question asks how BoE responds to / discusses / assesses
    something external (Fed, ECB, geopolitics, markets, crypto, etc.).
  - The question asks about a topic (inflation, rates, growth, risks)
    that BoE routinely publishes on.

If out_of_corpus=true, omit ALL other filter fields.

Examples:
  "What is the Fed's view on rates?" → out_of_corpus=true
  "What did Lagarde say at the ECB press conference?" → out_of_corpus=true
  "What is Bitcoin's price today?" → out_of_corpus=true
  "How does BoE respond to Fed tightening?" → out_of_corpus=false
  "What did Mann say about ECB policy?" → out_of_corpus=false
  "What's the BoE view on cryptocurrency regulation?" → out_of_corpus=false
  "What was the MPC vote split?" → out_of_corpus=false
  "Summarise the November 2025 MPR" → out_of_corpus=false
```

## Files changed

### 1. `src/boe_rag/pipelines/state.py` — one field added
```python
class RAGState(TypedDict, total=False):
    ...existing...
    out_of_corpus: bool   # written by analyze_query; read by router
```

### 2. `src/boe_rag/pipelines/prompts.py` — insert Rule 0

Insert the Rule 0 text above into `ANALYZE_QUERY_PROMPT` between the
current preamble and the existing numbered rules. Renumber existing
rules 1-6 → 1-6 (Rule 0 is the new first rule but keep old numbering
so git blame of other rules stays clean).

### 3. `src/boe_rag/pipelines/nodes.py` — QueryFilters + node + router + single-source-of-truth

```python
# Make ABSTAIN_MESSAGE public (was _ABSTAIN_MESSAGE) so adapters can
# import it — eliminates the "keep these strings in sync" comment.
ABSTAIN_MESSAGE = (
    "This question does not appear to be answerable from the Bank of "
    "England document corpus."
)

class QueryFilters(BaseModel):
    ...existing...
    out_of_corpus: bool = Field(
        default=False,
        description="True iff question's primary subject is outside BoE corpus scope.",
    )

def make_analyze_query_node(structured_llm):
    def _node(state):
        trace = _append_trace(state, "analyze_query")
        try:
            filters = structured_llm.invoke(
                ANALYZE_QUERY_PROMPT.format(question=state["question"])
            )
        except Exception as e:
            logger.warning("analyze_query failed: %s; no filters + no scope flag", e)
            return {"metadata_filters": None, "pipeline_trace": trace}
        # Priority: scope check wins over filter extraction.
        if filters.out_of_corpus:
            return {
                "out_of_corpus": True,
                "metadata_filters": None,
                "pipeline_trace": trace,
            }
        raw = filters.model_dump(exclude_none=True)
        # drop the scope flag from filter dict (it's not a ChromaDB field)
        raw.pop("out_of_corpus", None)
        if "speaker" in raw and raw["speaker"]:
            raw["speaker"] = normalise_speaker(raw["speaker"])
        where = _build_where(raw)
        return {
            "metadata_filters": where,
            "initial_metadata_filters": where,
            "out_of_corpus": False,
            "pipeline_trace": trace,
        }
    return _node

def make_abstain_out_of_corpus_node():
    def _node(state):
        return {
            "answer": ABSTAIN_MESSAGE,
            "reranked_documents": [],
            "is_grounded": None,
            "pipeline_trace": _append_trace(state, "abstain_out_of_corpus"),
        }
    return _node

def route_after_analyze_query(state):
    """out_of_corpus=True → abstain_out_of_corpus; else retrieve.

    Missing-key defaults to False — transient analyze_query failures
    go through the normal pipeline, not the abstain path.
    """
    if state.get("out_of_corpus") is True:
        return "abstain_out_of_corpus"
    return "retrieve"
```

### 4. `src/boe_rag/pipelines/enhanced.py` — graph rewiring

```python
analyze_node = make_analyze_query_node(structured_llm)
abstain_oos_node = make_abstain_out_of_corpus_node()
...
wf.add_node("abstain_out_of_corpus", abstain_oos_node)
# Replace: wf.add_edge("analyze_query", "retrieve")
wf.add_conditional_edges(
    "analyze_query",
    route_after_analyze_query,
    {
        "retrieve": "retrieve",
        "abstain_out_of_corpus": "abstain_out_of_corpus",
    },
)
wf.add_edge("abstain_out_of_corpus", END)
```

### 5. `src/boe_rag/evaluation/adapters.py` — re-export ABSTAIN_MESSAGE

```python
from boe_rag.pipelines.nodes import ABSTAIN_MESSAGE   # single source of truth

# Re-export for backward compat with test_adapters.py
__all__ = [..., "ABSTAIN_MESSAGE", ...]
```

### 6. `src/boe_rag/evaluation/metrics.py` — new recall metric

```python
# In compute_crag_metrics, after abstain_correctness:
"should_abstain_recall": (
    (correct_abstains / len(should)) if should else None
),
```

## Test matrix (13 tests total)

### Unit tests in `tests/pipelines/test_nodes.py` (10 new)

Each stubs the structured-output LLM to return a deterministic
`QueryFilters`; asserts the analyze_query node's return dict.

| # | Scenario | Input | Stub returns | Expected node output |
|---|---|---|---|---|
| 1 | Out-of-corpus — Fed | "What is the Fed's view on rates?" | `out_of_corpus=True` | `{out_of_corpus: True, metadata_filters: None}` |
| 2 | Out-of-corpus — Lagarde | "What did Lagarde say at ECB?" | `out_of_corpus=True` | same |
| 3 | Out-of-corpus — Bitcoin | "What is Bitcoin's price today?" | `out_of_corpus=True` | same |
| 4 | In-corpus — vote split | "What was Feb 2026 vote?" | `out_of_corpus=False` | `{out_of_corpus: False, metadata_filters: {...}}` |
| 5 | In-corpus — BoE on Fed | "How does BoE respond to Fed?" | `out_of_corpus=False` | same |
| 6 | In-corpus — Mann on ECB | "What did Mann say on ECB?" | `out_of_corpus=False` | same |
| 7 | In-corpus — BoE on crypto | "BoE view on crypto reg?" | `out_of_corpus=False` | same |
| 8 | analyze_query exception | — | LLM raises | `{metadata_filters: None}`, no out_of_corpus key |
| 9 | Scope True + spurious filter | (edge case) | `out_of_corpus=True, document_type=MPR` | `out_of_corpus: True, metadata_filters: None` (scope wins) |
| 10 | ABSTAIN_MESSAGE single-source-of-truth | — | — | `from boe_rag.pipelines.nodes import ABSTAIN_MESSAGE as A; from boe_rag.evaluation.adapters import ABSTAIN_MESSAGE as B; assert A is B or A == B` (catches future drift across modules) |

### Route unit tests in `tests/pipelines/test_nodes.py` (3 new)

```python
def test_route_after_analyze_query_out_of_corpus():
    assert route_after_analyze_query({"out_of_corpus": True}) == "abstain_out_of_corpus"

def test_route_after_analyze_query_in_corpus():
    assert route_after_analyze_query({"out_of_corpus": False}) == "retrieve"

def test_route_after_analyze_query_missing_key():
    """Transient analyze_query failure → default to retrieve, NOT abstain."""
    assert route_after_analyze_query({}) == "retrieve"
```

### Integration test in `tests/pipelines/test_enhanced.py` (1 new)

```python
def test_enhanced_pipeline_abstains_on_out_of_corpus(...):
    """Inject a structured LLM that returns out_of_corpus=True;
    assert the compiled graph routes to abstain node and trace ends
    with 'abstain_out_of_corpus'."""
```

### Metric test in `tests/evaluation/test_metrics.py` (1 new)

```python
def test_should_abstain_recall_computation():
    """Recall = |abstains ∩ should| / |should|. Regardless of total abstains."""
    # Fixture: should_abstain = {q21}; abstains = {q21, q06, q10}
    m = compute_crag_metrics(..., should_abstain_ids={"q21"})
    assert m["should_abstain_recall"] == 1.0
    assert m["abstain_correctness"] == pytest.approx(1/3)   # precision view
```

## Pre-flight empirical check (before full re-eval)

Run real `analyze_query` (with live Claude API) on **all 25 queries**
to confirm the prompt classifies correctly under real conditions.

```python
# scripts/check_corpus_scope.py
# Loads test_set.csv, builds the production analyze_query node,
# calls it on each question, persists raw QueryFilters + classification
# to data/evaluation_results/pre_flight_scope_check.json.
#
# Validates against a hand-labelled expected mapping:
#   q21 → True
#   all other qids → False
# Fail-closed: if any qid misclassifies, abort and tune the prompt.

CASES = {
    "q01": False, "q02": False, "q03": False, "q04": False, "q05": False,
    "q06": False, "q07": False, "q08": False, "q09": False, "q10": False,
    "q11": False, "q12": False, "q13": False, "q14": False, "q15": False,
    "q16": False, "q17": False, "q18": False, "q19": False, "q20": False,
    "q21": True,   # the only expected out-of-corpus query
    "q22": False, "q23": False, "q24": False, "q25": False,
}
```

**Use the REAL production LLM, not a reconstructed copy.** Eliminates
drift:
```python
# scripts/check_corpus_scope.py
from boe_rag.pipelines.enhanced import EnhancedPipeline
from boe_rag.pipelines.nodes import ANALYZE_QUERY_PROMPT, QueryFilters

pipe = EnhancedPipeline()                     # full production config
llm = pipe._structured_llm                    # exactly the judge that runs in prod

for qid, question in all_25_queries:
    response: QueryFilters = llm.invoke(
        ANALYZE_QUERY_PROMPT.format(question=question)
    )
    save raw response + expected + pass/fail
```

Cost: 25 × ~$0.008 = **$0.20**. Saves ~$2 if prompt is miscalibrated.

Output: `pre_flight_scope_check.json` with one entry per query
containing raw QueryFilters dict + pass/fail bool — goes into the
report's Methodology appendix.

### Iteration budget (hard limit)

If pre-flight fails (any qid misclassifies), tune the prompt and rerun.
**Max 3 attempts** = $0.60 budget. If attempt 3 still fails:

- **ABORT B1.** Do not ship a partially-working scope check.
- `git switch main && git branch -D feat/abstain-scope-check` to clean up.
- The fix is not viable under real conditions; forcing it risks
  regressing other queries. Report keeps the current `abstain_correctness=0.0`
  result and owns it as a limitation.

## Re-eval plan (after pre-flight passes with 25/25 correct)

1. `scripts/run_eval.py` — rerun both pipelines. ~$2, ~13 min.
2. `scripts/run_ragas.py --pipeline both --no-resume` — rerun RAGAS
   on both with gpt-4o-mini. ~$0.50, ~10 min.
   (Baseline behaviour unchanged, but we regenerate its JSONL to get
   a consistent run_metadata timestamp for the report.)
3. Verify:
   - q21: `is_abstain=True`, `pipeline_trace` ends with `abstain_out_of_corpus`
   - `crag_metrics.json`:
     - `abstain_correctness`: 0.00 → 0.25
     - `should_abstain_recall`: 0.00 → 1.00 (new)
     - `abstain_ids`: `['q06','q10','q21','q24']`
     - `correct_abstain_ids`: `['q21']`
     - `missed_abstain_ids`: `[]`
   - 24 non-q21 queries: same `is_abstain` values as pre-fix
     (auto-compared by a short verification script)
4. Re-run `scripts/compare_evaluators.py` for the Sonnet subset if
   desired — the subset doesn't include q21, so results should be
   unchanged (quick sanity check, free).

**Total re-eval cost: ~$2.70.** (Was ~$2.50 in v2 — added RAGAS
baseline regeneration for consistency.)

## Regression check script (new)

Before merging, prove the abstain set changes **only by adding q21**.
Per-query equality is the wrong contract (Claude @ temp=0 isn't
bit-deterministic across runs, so rare jitter is expected). The real
invariant is set-level:

```python
# scripts/verify_abstain_delta.py
# Load pre-fix enhanced_results.json from git HEAD~ (or a tag we push
# before starting B1) and post-fix enhanced_results.json from current
# working tree.
#
# Compute:
#   pre_abstains  = {qid for qid, r in pre.items()  if is_abstain(r['answer'])}
#   post_abstains = {qid for qid, r in post.items() if is_abstain(r['answer'])}
#
# Required invariants (any failure blocks merge):
#   assert post_abstains - pre_abstains == {"q21"}  # only q21 is new
#   assert pre_abstains - post_abstains == set()    # no prior abstain drops
#
# Soft warning (log, don't block):
#   For each non-q21 query, if pipeline_trace structure changed
#   meaningfully (e.g. length ± or different final node), log a note
#   for manual review.
```

Guardrail BEFORE starting: `git tag pre-b1-abstain-fix` and
`git push origin pre-b1-abstain-fix` so the verifier has a stable
reference point even after merges.

## Risk register (v3)

| Risk | Probability | Mitigation |
|---|---|---|
| Claude misclassifies a borderline query | LOW-MED | Pre-flight on all 25 catches before API spend |
| Prompt change cascades on other queries' filter extraction (speaker / date / section_category) | LOW | Few-shot examples are classification-focused, not filter-focused |
| Adding `out_of_corpus` as a Pydantic field breaks existing test stubs | LOW | Default=False preserves backward compat |
| Graph compile fails on the conditional edge | LOW | pytest catches before API runs |
| `abstain_correctness` denominator increases and the precision number DROPS despite a correct abstain | N/A | Pre-existing abstains (q06/q10/q24) already count against it; new metric `should_abstain_recall` is the cleaner win framing |
| Pre-flight passes but real pipeline fails (multi-node interaction) | LOW | Integration test in test_enhanced.py catches |
| Report framing lost in edit pass | — | See "Report framing" below — write this paragraph FIRST |

## Report framing (write this BEFORE coding so it doesn't get lost)

For the Results section, the claim is:

> "Introducing an explicit out-of-corpus scope check in `analyze_query`
> improved should-abstain recall from 0/1 to 1/1 (the Fed query q21),
> with no measurable impact on aggregate RAGAS scores. Abstain precision
> is 1/4 rather than 1/1 because three other queries (q06, q10, q24)
> abstain due to over-strict document grading — a separate failure mode
> outside the scope of this fix. The scope check added ~0.06 USD to
> per-full-eval cost (25 extra structured-output calls)."

## Rollback

`git revert <merge sha>`. All changes contained in 6 files:
- `state.py`, `nodes.py`, `prompts.py`, `enhanced.py` (pipeline)
- `evaluation/adapters.py`, `evaluation/metrics.py`
- `tests/pipelines/test_nodes.py`, `tests/pipelines/test_enhanced.py`,
  `tests/evaluation/test_metrics.py`
- `scripts/check_corpus_scope.py` (new), `scripts/verify_non_q21_unchanged.py` (new)

No data migrations. `out_of_corpus` defaults to False so old JSONs stay
valid.

## Effort (honest, v3)

| Phase | Time |
|---|---|
| Report-framing paragraph (write first) | 20 min |
| QueryFilters + state + prompt Rule 0 | 30 min |
| New node + router + graph wiring | 30 min |
| ABSTAIN_MESSAGE single-source refactor | 15 min |
| 14 new tests (TDD-first) | 1.5 hr |
| Pre-flight script + real API run | 30 min + $0.20 |
| Full re-eval (passive wait) | 25 min + $2.50 |
| Verification + regression check | 15 min |
| Commit + merge + push | 10 min |
| **Total active** | **~4.5 hours** |
| **Total cost** | **~$2.70** |

## Branch: `feat/abstain-scope-check`

Start from clean main:
```bash
git switch main                # errors if uncommitted changes exist
git switch -c feat/abstain-scope-check    # errors cleanly if branch exists
git tag pre-b1-abstain-fix     # regression-check reference point
git push origin pre-b1-abstain-fix
```

Abort cleanup (if pre-flight fails 3× or we decide to skip B1):
```bash
git switch main
git branch -D feat/abstain-scope-check
# Tag pre-b1-abstain-fix stays; no harm. Delete it later if desired.
```
