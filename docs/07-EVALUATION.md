# 07 — Evaluation (v2)

> **v2 changelog** — full rewrite after P0/P1 audit. Aligned to the test
> set we actually have (25 queries, real schema), RAGAS 0.4.3 canonical
> API (`ragas.metrics.collections`, `EvaluationDataset`/`SingleTurnSample`,
> `RunConfig`), proper concurrency + checkpoint/resume, abstain policy,
> paired statistical tests, repro metadata. No more hardcoded "expected
> deltas" — numbers come from the run, not from hope.

## Objective
Score both pipelines on the same 25-query test set with four RAGAS
metrics, compute CRAG-specific metrics from the enhanced pipeline's
trace, run paired Wilcoxon tests to separate signal from noise, and save
everything for the report (spec 08) and demo log (spec 09).

## Depends on
- 05-BASELINE-PIPELINE + 06-ENHANCED-PIPELINE — both `.run(q)` callable.
- `scripts/run_eval.py` — already exists and has produced
  `data/evaluation_results/{baseline,enhanced}_results.json`. This spec
  consumes those JSONs; it does NOT re-run the pipelines unless `--rerun`
  is passed.

## Deliverables
- [ ] `src/boe_rag/evaluation/ragas_eval.py` — RAGAS runner with
      checkpoint, resume, dry-run, and per-sample output.
- [ ] `src/boe_rag/evaluation/metrics.py` — CRAG-specific metrics +
      paired Wilcoxon + category breakdown.
- [ ] `src/boe_rag/evaluation/adapters.py` — pipeline-result → RAGAS
      `SingleTurnSample` conversion (keeps the RAGAS-specific shape out
      of domain code).
- [ ] `scripts/run_ragas.py` — CLI entry: `--sample N`, `--resume`,
      `--pipeline {baseline,enhanced,both}`, `--rerun-pipelines`.
- [ ] `data/evaluation_results/ragas_{baseline,enhanced}.jsonl` —
      per-sample scores, one row per (query, metric). Resumable.
- [ ] `data/evaluation_results/ragas_aggregate.json` — headline means
      (with AND without abstain), paired p-values, CIs, repro metadata.
- [ ] `data/evaluation_results/crag_metrics.json` — rewrite rate,
      hallucination rate, filter usage, rerank impact, abstain stats.
- [ ] `data/evaluation_results/comparison_table.csv` — Table 2 of the
      report, one row per metric.
- [ ] `data/evaluation_results/per_category.csv` — Faithfulness +
      Context Recall by category, used for the Deep Context story.
- [ ] Tests: the RAGAS runner + CRAG metrics have unit tests with stub
      LLMs / fixture JSON (TDD discipline as in specs 03/06).

---

## Test set — ground truth

**Path**: `data/test_set.csv` (per `boe_rag.config.Paths.TEST_SET`).

**Actual schema** (already in the file — no rename):
```
question, category, expected_answer, source_document, source_paragraph, source_quote
```

**Size**: 25 rows (0 TODOs as of 2026-04-14). Category distribution
(verified from the CSV, not guessed):

| Category                    | N  | Purpose                                               |
|-----------------------------|----|-------------------------------------------------------|
| `simple_factual`            | 5  | Both should find; delta via citations                 |
| `comparative`               | 5  | Enhanced should win via metadata filtering            |
| `deep_context`              | 11 | Enhanced should win decisively (includes speech qs)   |
| `edge_case_out_of_scope`    | 1  | q21 — Fed question. Enhanced **should abstain**       |
| `edge_case_ambiguous`       | 1  | q22 — crypto regulation. Either pipeline acceptable   |
| `edge_case_too_broad`       | 1  | q23 — "summarise entire MPR". Tests vague behaviour   |
| `edge_case_numerical`       | 1  | q24 — exact GDP figure. Tests precision               |

The `deep_context` bucket absorbed what an earlier draft called a
separate `speech` category — speech questions (Taylor, Bailey, Mann,
Dhingra, Ramsden, Breeden) live here. For per-category reporting
we can optionally split `deep_context` into `deep_context_speech`
(speaker in question) and `deep_context_other` post-hoc if the story
needs it.

**Rules for ground truth** (already enforced in the CSV):
- `expected_answer`: Complete factual answer in BoE language.
- `source_quote`: Verbatim quote used by `LLMContextRecall` — this is
  the string RAGAS checks sub-claims against.
- For edge cases, `expected_answer` describes the **correct behaviour**
  (e.g. "Pipeline should abstain — query is out of corpus scope").
- Every row manually verified against the scraped document in
  `data/raw/`.

---

## RAGAS wiring — v0.4.3 canonical (verified against installed library)

### Gotcha that bit the first draft

Collections metrics (the non-deprecated ones) **do not accept
`LangchainLLMWrapper`**. They raise:
```
ValueError: Collections metrics only support modern InstructorLLM.
Found: LangchainLLMWrapper. Use: llm_factory(...)
```
So the "obvious" wiring from every blog post is wrong for 0.4.3
collections metrics. Use `ragas.llms.llm_factory` with a raw provider
SDK client instead.

### Canonical class names (verified)

The names differ from the deprecated `ragas.metrics` path — do not
guess them:

| This is what's in `ragas.metrics.collections`     | Not this                          |
|---------------------------------------------------|-----------------------------------|
| `Faithfulness`                                    | `Faithfulness` ✓ (same)           |
| `AnswerRelevancy`                                 | ~~`ResponseRelevancy`~~           |
| `ContextPrecisionWithReference`                   | ~~`LLMContextPrecisionWithReference`~~ |
| `ContextRecall`                                   | ~~`LLMContextRecall`~~            |

### Imports
```python
from anthropic import Anthropic
from openai import OpenAI

from ragas import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,   # we have references — use this
    ContextRecall,
)
```

### Why `ContextPrecisionWithReference` not `ContextPrecisionWithoutReference`
We have `expected_answer` and `source_quote` for every query. The
with-reference variant compares each retrieved chunk against the
reference answer directly — lower variance, better signal than the
reference-free variant which bootstraps relevance from the generated
response (noisy).

### Evaluator LLM + embeddings
```python
# Raw Anthropic client with SDK-level retry. The Anthropic SDK handles
# 429s and 5xx with exponential backoff internally — no need for a
# LangChain retry wrapper (which isn't compatible with InstructorLLM
# anyway). max_retries=5 gives ~2 min of total backoff headroom.
anthropic_client = Anthropic(max_retries=5)

evaluator_llm = llm_factory(
    model=config.GENERATION_MODEL,       # sonnet-4, matches pipeline
    provider="anthropic",
    client=anthropic_client,
    temperature=0.0,                     # REQUIRED for reproducibility
    max_tokens=2048,                     # default 1024 truncates
                                         # Faithfulness claim lists on
                                         # long answers. Verified.
)

evaluator_embeddings = RagasOpenAIEmbeddings(
    client=OpenAI(),                     # picks up OPENAI_API_KEY
    model=config.EMBEDDING_MODEL,        # text-embedding-3-small
)
```

Notes:
- `llm_factory` returns an `InstructorLLM` whose `model_args` end up
  `{'temperature': 0.0, 'top_p': 0.1, 'max_tokens': 2048}`. The
  `top_p=0.1` default is intentional on RAGAS's end (nudge toward
  deterministic token picks).
- Using the **same model** (Sonnet 4) for generation and evaluation is a
  known weakness (self-assessment bias). Called out explicitly in the
  report as a limitation (spec 08). Mitigating with a second-opinion
  grader on a small subsample is out of scope / budget.
- `temperature=0` is non-negotiable. RAGAS prompts are not
  deterministic enough at t>0 for a rerun to reproduce numbers within
  ±0.05.

### Concurrency + timeout

Collections-metric per-sample scoring doesn't use RAGAS `RunConfig`
(that's the batched-`evaluate()` path). For per-sample checkpointed
loops we control concurrency directly:

```python
import asyncio
import inspect

SEM = asyncio.Semaphore(4)   # tier-1 Sonnet-friendly burst budget

# Each collections metric has a DIFFERENT ascore signature — you can't pass
# the full kwargs bag uniformly or you'll get TypeError. Filter per metric.
#
# Verified 0.4.3 signatures:
#   Faithfulness.ascore(user_input, response, retrieved_contexts)
#   AnswerRelevancy.ascore(user_input, response)
#   ContextPrecisionWithReference.ascore(user_input, reference, retrieved_contexts)
#   ContextRecall.ascore(user_input, retrieved_contexts, reference)

def _dispatch_kwargs(metric, sample):
    full = dict(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=list(sample.retrieved_contexts),
        reference=sample.reference,
    )
    sig = inspect.signature(metric.ascore)
    return {k: v for k, v in full.items() if k in sig.parameters}

async def _bounded_score(metric, sample):
    async with SEM:
        return await metric.ascore(**_dispatch_kwargs(metric, sample))
```

**Do not use the sync `metric.score(**kwargs)`** — it just wraps
`asyncio.run(ascore(**kwargs))` and will still TypeError on extra
kwargs, plus it can't run inside an existing event loop. Always go
async with the filter-dispatch above.

Rationale for 4-way concurrency: tier-1 Anthropic is 30k TPM on Sonnet.
Each RAGAS metric invocation ~2-3k input tokens + ~1k output. Four
parallel calls ≈ 12-16k tokens/sec burst — fits the window. Eight
workers caused 429 cascades on the spec 06 v5 run. The Anthropic SDK's
own `max_retries=5` absorbs the rare 429 we still get.

If we ever switch to the batched path (`evaluate()`), `RunConfig` is
available there:
```python
from ragas.run_config import RunConfig
rc = RunConfig(max_workers=4, timeout=180, max_retries=3, max_wait=60)
evaluate(dataset=ds, metrics=METRICS, llm=evaluator_llm,
         embeddings=evaluator_embeddings, run_config=rc,
         raise_exceptions=False)
```
But `evaluate()` loses per-metric checkpointing, so we only fall back to
it if the per-sample loop proves unworkable.

### Dataset construction
```python
samples = [
    SingleTurnSample(
        user_input=row["question"],
        retrieved_contexts=[d.text for d in result.sources],
        response=result.answer,
        reference=row["expected_answer"],
    )
    for row, result in zip(test_set, pipeline_outputs)
]
dataset = EvaluationDataset(samples=samples)
```

Column renames from v0.1 → v0.4:
| v0.1              | v0.4+               |
|-------------------|---------------------|
| `question`        | `user_input`        |
| `answer`          | `response`          |
| `contexts`        | `retrieved_contexts`|
| `ground_truth`    | `reference`         |

Do NOT feed a `datasets.Dataset` with the legacy names. RAGAS 0.4 has
rename shims but silent casts have caused real bugs — go native.

---

## The metrics, in order of report importance

| # | Metric                              | Needs reference? | Cost per query | Failure mode                    |
|---|-------------------------------------|------------------|----------------|---------------------------------|
| 1 | `Faithfulness`                      | no               | ~3 LLM calls   | Can't extract sub-claims → 0    |
| 2 | `AnswerRelevancy`                   | no               | 3 embeds + 1 LLM| Generic answers score high     |
| 3 | `ContextPrecisionWithReference`     | **yes**          | ~k LLM calls   | Reference is too narrow         |
| 4 | `ContextRecall`                     | **yes**          | ~k LLM calls   | `source_quote` too short        |

**Cost per metric per query** multiplied out: ~4 + 3 + k + k LLM calls.
At k=5 retrieved chunks that's ~17 calls per (query, pipeline). Total:
17 × 25 × 2 = **~850 LLM calls** for a full run, plus ~150 embedding
calls. At ~3s/call with 4-way concurrency: **~10-15 min wall clock**,
~$5-8 in API cost. Confirmed via `--sample 2` dry-run before any full
run.

---

## Abstain policy — explicit (refined after reading RAGAS source)

The enhanced pipeline abstains with a fixed string
(`_ABSTAIN_MESSAGE`) when two grading passes find no relevant docs.
Baseline has no abstain path and always produces a synthetic answer.

### RAGAS input-validation behaviour (verified from source, 0.4.3)

| Metric                            | Empty `retrieved_contexts`? | Empty `reference`? |
|-----------------------------------|-----------------------------|--------------------|
| `Faithfulness`                    | **ValueError**              | n/a                |
| `AnswerRelevancy`                 | ignored (not a field)       | n/a                |
| `ContextPrecisionWithReference`   | **ValueError**              | **ValueError**     |
| `ContextRecall`                   | **ValueError**              | **ValueError**     |

Abstain rows in `enhanced_results.json` have
`reranked_documents=[]` → `retrieved_contexts=[]`. Running Faithfulness,
ContextPrecisionWithReference, or ContextRecall on those rows raises.
AnswerRelevancy runs but scores the boilerplate abstain message (~0)
against the original question.

### Policy

1. Every result row keeps a boolean `is_abstain` flag
   (`answer == _ABSTAIN_MESSAGE`).
2. **Abstain rows are SKIPPED upfront** for Faithfulness,
   ContextPrecisionWithReference, and ContextRecall. A `"skipped":
   "abstain"` JSONL record is written for those tuples so the
   checkpoint is deterministic and auditable.
3. Abstain rows ARE scored by AnswerRelevancy — the ~0 score is
   meaningful ("the abstain does not address the question").
4. Headline aggregates are reported **twice** only where both make
   sense:
   - AnswerRelevancy: report `mean_all` and `mean_answered`.
   - Faithfulness / ContextPrecision / ContextRecall: only
     `mean_answered` is defined (abstain rows have no score). Report it
     alongside the abstain count so readers can do their own
     adjustment.
5. Separate `abstain_correctness` metric computed against a hand-coded
   set of "should-abstain" queries:
   - **Correct abstain**: q21 (Federal Reserve — out of corpus).
   - **Questionable abstain**: any other enhanced abstain — investigate
     and note in the report (filters too narrow? chunk missing?).
   - Baseline never abstains, so its abstain-correctness on q21 is
     `0/1` by definition — this is a **signal**, not a bug; baseline
     answering q21 is exactly the failure mode we expect.

```python
SHOULD_ABSTAIN_IDS = {"q21"}   # extend if more out-of-corpus queries added
```
Equivalent to `{qid for qid, row in test_set if row.category == "edge_case_out_of_scope"}`
— the constant exists to keep the policy auditable rather than
coupling abstain-correctness to a category-string match that could
silently change.

---

## CRAG-specific metrics (implementable versions)

Per the P1 audit, the v1 spec had "rewrite success rate" defined as
something we can't measure. Corrected definitions:

| Metric                      | Definition                                                                 |
|-----------------------------|----------------------------------------------------------------------------|
| `rewrite_trigger_rate`      | `crag_rewrites > 0` count / N                                             |
| `rewrite_recovery_rate`     | Of rewrite-triggered queries, fraction that ended grounded AND non-abstain |
| `hallucination_flag_rate`   | `is_grounded == False` count / N                                          |
| `hallucination_recovery_rate`| Of retries, fraction that ended grounded (`is_grounded == True` after bump)|
| `metadata_filter_rate`      | `metadata_filters_used is not None` count / N                             |
| `mean_chunks_retrieved`     | mean(`chunks_retrieved`)                                                  |
| `mean_chunks_used`          | mean(`chunks_used`)                                                       |
| `rerank_top1_change_rate`   | Requires `pre_rerank_ids` / `post_rerank_ids` in PipelineResult — see below |
| `abstain_rate`              | Abstains / N                                                              |
| `abstain_correctness`       | Correct abstains / total abstains; also correct-abstain-on-should-abstain |

### Implementation gap: rerank top-1 change

The state carries `pre_rerank_ids` / `post_rerank_ids` but
`_state_to_pipeline_result` in `enhanced.py` drops them. **Required
change to spec 06 artifact**:
- Add two fields to `PipelineResult`:
  ```python
  pre_rerank_ids: Sequence[str] = ()
  post_rerank_ids: Sequence[str] = ()
  ```
  (default `()` keeps baseline compatibility — baseline sets both empty).
- Carry them through `_state_to_pipeline_result`.
- Serialize them in `scripts/run_eval.py`.

Without this change, rerank impact reporting in spec 08 is impossible.

---

## Paired Wilcoxon + bootstrap CI — the statistical honesty

N=25 is tiny. A 0.05 difference in means is well within noise. Mean
comparisons without a paired test are anecdote.

### The test
```python
from scipy.stats import wilcoxon

# per-query baseline score vs enhanced score for the same query
# (paired: same 25 questions, two systems)
# Drop rows where EITHER side is NaN/None (e.g. a metric failed on
# baseline but not enhanced — unpaired data can't be used).
statistic, p_value = wilcoxon(
    baseline_scores,
    enhanced_scores,
    alternative="less",    # H0: baseline >= enhanced; H1: baseline < enhanced
    zero_method="zsplit",  # split tied pairs half-and-half rather than
                           # dropping them entirely. On bounded [0,1]
                           # scores ties are common (both score 0 or both
                           # score 1). "wilcox" drops ties and
                           # under-powers the test.
)
```

Why non-parametric:
- Small N, bounded [0,1] scores, skew toward 0 or 1 — normality
  assumption fails.
- Paired: same 25 questions on both systems — paired is more powerful
  than unpaired at small N.

### Multiple-testing correction

We run the Wilcoxon on 4 metrics. At per-test α=0.05 the
family-wise error rate is ≈1 − 0.95⁴ = **0.19**, i.e. ~19% chance of
at least one false positive even if both pipelines are equivalent.

**Correction applied**: Holm-Bonferroni (stepdown). Slightly more powerful
than vanilla Bonferroni at the same FWER. Both raw and corrected
p-values appear in the output; the report quotes corrected.

Hand-coded (10 lines, no `statsmodels` dependency):
```python
def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the same order as input.
    Each adjusted p = min(1, max_so_far) where max_so_far is the
    cumulative max of (n - rank) * p over the sorted sequence.
    """
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (n - rank) * pvalues[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted
```

### 95% CI on the delta (paired BCa bootstrap)

`scipy.stats.bootstrap` with `paired=True` and `method="BCa"`
(bias-corrected and accelerated — better than the percentile method
for small, skewed samples):

```python
import numpy as np
from scipy.stats import bootstrap

def _delta_mean(b, e):
    return np.mean(np.asarray(e) - np.asarray(b))

ci = bootstrap(
    data=(baseline_scores, enhanced_scores),
    statistic=_delta_mean,
    paired=True,           # resample PAIRS, not independent columns
    vectorized=False,
    n_resamples=10_000,    # BCa needs more than the percentile method;
                           # 10k is standard and fast in-memory
    confidence_level=0.95,
    method="BCa",
    rng=np.random.default_rng(42),   # seed for reproducibility
).confidence_interval
```

### Statistical power — honest caveat

With N=22 effective (after 3 abstains dropped for context metrics) and
α=0.05, a paired Wilcoxon detects an effect size of approximately 0.6
standard deviations at 80% power. In bounded-[0,1] metric terms with
observed SDs ≈0.2-0.3, that translates to **minimum detectable delta
≈0.12-0.18**. If the true delta is smaller, we are underpowered and
will fail to reject H0 even if enhanced is truly better. The report
states this explicitly — "absence of evidence is not evidence of
absence" must be on the page.

### Reported shape
```json
{
  "faithfulness": {
    "n_scored_baseline": 22, "n_scored_enhanced": 22,
    "baseline_mean_answered": 0.xx, "enhanced_mean_answered": 0.xx,
    "delta": 0.xx,
    "wilcoxon_statistic": x.x, "p_raw": 0.0x, "p_holm": 0.0x,
    "ci95_delta_low": 0.xx, "ci95_delta_high": 0.xx,
    "ci_method": "BCa_paired_10k"
  },
  "answer_relevancy": {
    "n_scored_baseline": 25, "n_scored_enhanced": 25,
    "baseline_mean_all": 0.xx, "enhanced_mean_all": 0.xx,
    "baseline_mean_answered": 0.xx, "enhanced_mean_answered": 0.xx,
    "delta_answered": 0.xx,
    ...
  },
  ...
}
```

---

## Checkpoint / resume

Full eval can take 45-60 min at tier-1 (see the TPM analysis in the
cost section). A single 429 or timeout must not force a restart.

**Design**: write per-sample scores as JSONL (one line per
`(pipeline, metric, query_id)` tuple). On resume, skip tuples already
present.

```
ragas_enhanced.jsonl
{"pipeline":"enhanced","query_id":"q01","metric":"faithfulness","score":0.83,"ts":"..."}
{"pipeline":"enhanced","query_id":"q01","metric":"answer_relevancy","score":0.78,...}
{"pipeline":"enhanced","query_id":"q01","metric":"context_precision_with_reference","score":0.71,...}
{"pipeline":"enhanced","query_id":"q01","metric":"context_recall","score":0.92,...}
...
```

Metric name strings are exactly `metric.name` on the instantiated
collections class: `faithfulness`, `answer_relevancy`,
`context_precision_with_reference`, `context_recall`.

`ragas_eval.py` (sketch — real implementation async with the semaphore
from the concurrency section):
```python
async def run_ragas(
    samples: list[SingleTurnSample],
    query_ids: list[str],
    pipeline_name: str,
    metrics: list,                # list of instantiated collections metrics
    *,
    out_path: Path,
    resume: bool = True,
) -> None:
    done = _load_done_keys(out_path) if resume else set()
    tasks = []
    for metric in metrics:
        for sample, qid in zip(samples, query_ids):
            key = (pipeline_name, qid, metric.name)
            if key in done:
                continue
            tasks.append(_score_and_persist(metric, sample, key, out_path))
    await asyncio.gather(*tasks)

# Metrics that raise on empty retrieved_contexts (verified from source)
_CONTEXT_REQUIRED = {
    "faithfulness",
    "context_precision_with_reference",
    "context_recall",
}

async def _score_and_persist(metric, sample, key, out_path):
    # Abstain-skip: upfront, before we burn a semaphore slot or an API call.
    if metric.name in _CONTEXT_REQUIRED and not sample.retrieved_contexts:
        _append_jsonl(out_path, key, score=None, skipped="abstain", err=None)
        return
    async with SEM:
        score, err = None, None
        try:
            result = await metric.ascore(**_dispatch_kwargs(metric, sample))
            score = float(result.value)      # MetricResult.value
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:500]}"
        _append_jsonl(out_path, key, score=score, skipped=None, err=err)
```

`evaluate()` on the whole dataset is faster (batched, auto-parallelised
via `RunConfig`) but gives us no per-metric checkpoint granularity.
Trade-off: per-sample is ~20% slower but resumable. We go per-sample.
Errors are persisted as null scores with an `err` field so a retry pass
can target only the failed tuples.

---

## Dry-run + CLI

Before any full run:
```bash
python scripts/run_ragas.py --sample 2 --pipeline both
```

Hits 2 queries per pipeline × 4 metrics = 16 LLM-call chains. Confirms:
- API keys valid
- RAGAS can parse our samples
- Score distribution is sane (not all 0 or all 1)
- No import / config errors

Full run:
```bash
python scripts/run_ragas.py --pipeline both           # defaults to resume
python scripts/run_ragas.py --pipeline both --no-resume   # start fresh
```

Optional re-run the pipelines first (if prompts or chunker changed):
```bash
python scripts/run_ragas.py --rerun-pipelines --pipeline both
```

This calls `scripts/run_eval.py` first, then runs RAGAS on its output.

---

## Repro metadata

Every output JSON/CSV gets a header block:

```json
{
  "run_metadata": {
    "timestamp": "2026-04-14T21:03:00+00:00",
    "git_sha": "024f73f...",
    "git_dirty": false,
    "ragas_version": "0.4.3",
    "generation_model": "claude-sonnet-4-20250514",
    "grading_model": "claude-sonnet-4-20250514",
    "embedding_model": "text-embedding-3-small",
    "rerank_model": "rerank-v3.5",
    "evaluator_model": "claude-sonnet-4-20250514",
    "evaluator_temperature": 0.0,
    "n_queries": 25,
    "test_set_hash": "sha256:..."
  },
  "results": {...}
}
```

`test_set_hash` detects silent test-set edits between runs — if the
hash changes, headline numbers are not comparable across runs.

---

## File layout

```
src/boe_rag/evaluation/
  __init__.py
  adapters.py          # PipelineResult + CSV row -> SingleTurnSample;
                       #   SHOULD_ABSTAIN_IDS; _load_test_set()
  ragas_eval.py        # run_ragas(), checkpointed per-sample loop,
                       #   dataset construction, metric configuration
  metrics.py           # CRAG-specific metrics, paired Wilcoxon,
                       #   bootstrap CI, per-category aggregation
  repro.py             # collect_run_metadata(): git sha, versions, hash

scripts/
  run_ragas.py         # CLI

tests/evaluation/
  test_adapters.py     # ground-truth CSV parsing; abstain flag
  test_metrics.py      # CRAG metrics from fixture results JSON;
                       #   Wilcoxon with known inputs
  test_ragas_eval.py   # checkpoint/resume; dry-run skips metrics on
                       #   already-scored keys; stub metric scorer
```

---

## Output files (exact paths, exact shapes)

```
data/evaluation_results/
├── baseline_results.json         # already exists (spec 06 artefact)
├── enhanced_results.json         # already exists
├── ragas_baseline.jsonl          # per-sample per-metric, resumable
├── ragas_enhanced.jsonl
├── ragas_aggregate.json          # headline table + p-values + CIs
├── crag_metrics.json             # enhanced-only CRAG stats
├── comparison_table.csv          # Table 2 of the report
├── per_category.csv              # Faithfulness + Recall by category
└── demo_log_worst.json           # bottom-3 queries per metric (feeds spec 09)
```

### `comparison_table.csv` shape
```
metric,baseline_mean_all,enhanced_mean_all,baseline_mean_answered,enhanced_mean_answered,delta_answered,wilcoxon_p,ci95_low,ci95_high
faithfulness,0.xx,0.xx,0.xx,0.xx,+0.xx,0.xx,-0.xx,+0.xx
answer_relevancy,...
context_precision_with_reference,...
context_recall,...
```

### `per_category.csv` shape
```
category,n,metric,baseline_mean,enhanced_mean,delta
simple_factual,5,faithfulness,0.xx,0.xx,+0.xx
simple_factual,5,context_recall,0.xx,0.xx,+0.xx
comparative,5,faithfulness,...
deep_context,11,faithfulness,...
deep_context,11,context_recall,...
edge_case_out_of_scope,1,faithfulness,...
...
```

Only the two metrics most-cited in the report (Faithfulness as the
headline grounding metric; Context Recall as the headline retrieval
metric) appear in `per_category.csv` to keep it readable. Full
4-metric-per-category table lives in the notebook.

**N=1 categories are plotted but not discussed statistically** in the
report: `edge_case_out_of_scope`, `_ambiguous`, `_too_broad`,
`_numerical` each have one row. A single observation is an anecdote;
include it in the CSV for transparency but the report narrative focuses
on `simple_factual` (5), `comparative` (5), and `deep_context` (11)
where N is large enough for a mean to mean something.

---

## Cost + time budget (honest, post-TPM-audit)

### Cost
Token accounting per RAGAS metric per query (averaged, measured from
sample runs — not wishcast):

| Metric                          | Input tok/q | Output tok/q |
|---------------------------------|-------------|--------------|
| Faithfulness (3 LLM calls)      | ~6,000      | ~1,200       |
| AnswerRelevancy (1 LLM + 3 emb) | ~500        | ~200         |
| ContextPrecisionWithReference (k=5) | ~7,500  | ~500         |
| ContextRecall (k=5)             | ~7,500      | ~500         |

Per pipeline (25q × 4 metrics): ~540k input, ~60k output. Both
pipelines: ~1.08M input, ~120k output. At Sonnet 4 pricing
($3/MTok in, $15/MTok out): **~$5 for RAGAS + ~$2 pipeline rerun + ~$3
contingency ≈ $10-12 total.** Previously estimated $6 (RAGAS only) —
still in ballpark, but documented now.

### Wall clock — the binding constraint is TPM, not concurrency

At tier-1 Anthropic (30k input tokens/min on Sonnet 4), steady-state
throughput caps at `30k tok/min × 1 min = 30k/min`. Total input
tokens needed: 1.08M. Minimum wall clock under a perfectly-filled TPM
window: **1.08M / 30k = ~36 min**. With SDK backoff on 429 bursts and
imperfect packing, expect **40-60 min wall clock** for a full run on
tier 1.

| Phase                         | ~Input tok | Min wall @ tier-1 30k | Min wall @ tier-2 80k |
|-------------------------------|------------|-----------------------|-----------------------|
| RAGAS full (both, 25q)        | 1.08M      | 36 min → 45-60 real   | 14 min → 20-30 real   |
| Pipeline rerun (both, 25q)    | ~0.15M     | 5 min                 | 2 min                 |
| Dry run (--sample 2)          | ~0.09M     | 3 min                 | 1 min                 |

`max_workers=4` from the concurrency section is only the *shape* of
the burst — the TPM window is the hard ceiling. With 4 workers each
firing a ~3k-token call every ~3s, we'd burst 24k tok/s = 1.44M tok/min,
which is well over the 30k/min limit. The SDK's 429-retry absorbs this
by spacing calls out; net throughput settles near 30k/min.

**Practical implication**: plan for 1 hour wall clock, not 15 min.
Schedule full runs when you can leave them running. Dry-runs stay
sub-5-min.

### Alternative: GPT-4o-mini evaluator (escape hatch)

If Anthropic tier capacity is a real problem, swap the evaluator LLM
to OpenAI GPT-4o-mini:
```python
from openai import OpenAI
evaluator_llm = llm_factory(
    model="gpt-4o-mini", provider="openai",
    client=OpenAI(), temperature=0.0,
)
```
- **Pros**: OpenAI tier limits are much more generous (tier 1 = 200k
  TPM on 4o-mini), so wall clock drops to ~5-10 min. Cost is ~10x
  cheaper (gpt-4o-mini is $0.15/MTok in, $0.60/MTok out).
- **Cons**: different model than the pipeline generator, different
  judging style, less comparable to the few published BoE-RAG results
  that also used Claude.
- **Methodological upside**: using a *different* model as judge
  partially mitigates the self-grading bias discussed in the report.

**Default: Sonnet 4 evaluator** (matches pipeline, keeps the
self-grading bias we flag as a limitation). **Fallback: gpt-4o-mini**
if the Sonnet run can't complete in an acceptable window.

### Stop-loss

If first full RAGAS run fails >30% of samples (null scores / errors
in the JSONL), stop and debug before retrying. Common causes
previously seen: 429 cascades from concurrency too high, truncated
Faithfulness output on `max_tokens` too low, evaluator LLM refusing
due to prompt format.

---

## What "done" looks like

### Acceptance criteria
1. `python scripts/run_ragas.py --sample 2 --pipeline both` exits 0 with
   plausible scores (none 0.0, none 1.0 across the board).
2. Full run writes all 7 output files, all with valid JSON / CSV.
3. `ragas_aggregate.json` contains non-null `wilcoxon_p` and `ci95`
   fields for all 4 metrics.
4. `comparison_table.csv` opens cleanly in a spreadsheet with 4 rows.
5. `per_category.csv` has rows for every category × both metrics.
6. Abstain rows appear in per-sample JSONL but are flagged and the
   `_answered` means exclude them.
7. `crag_metrics.json` has all 10 metrics listed above.
8. All `tests/evaluation/` tests pass.
9. Repro metadata block present in every top-level output.
10. The comparison table is realistic (no 0.0 or 1.0, no delta > 0.6 —
    those indicate a bug, not a miracle).

### What is NOT in scope for this spec
- RAGAS custom metrics (faithfulness-with-citations, answer-correctness
  with embedding similarity). Out of scope; use stock metrics.
- Second-opinion evaluator (different LLM as judge). Acknowledged as
  limitation in spec 08, not built here. GPT-4o-mini escape hatch
  (above) is a budget/time measure, NOT a bias-mitigation study.
- Formal statistical power analysis. The honest-caveat paragraph under
  "Paired Wilcoxon" gives the minimum detectable delta; that's the
  only power claim we make.

---

## Record formats (exact schemas — no guessing later)

### Per-sample JSONL line (`ragas_{pipeline}.jsonl`)

One JSON object per line. One line per `(pipeline, query_id, metric)`
tuple. On resume, existence of a line with identical
`(pipeline, query_id, metric)` means that tuple is done — skip.

```json
{
  "pipeline": "enhanced",
  "query_id": "q07",
  "metric": "faithfulness",
  "score": 0.83,
  "skipped": null,
  "err": null,
  "ts": "2026-04-14T22:15:03+00:00"
}
```
- `score`: float in [0,1], or `null` on error/skip.
- `skipped`: null, or `"abstain"` (context-requiring metric on abstain
  row), or `"missing_reference"` (edge-case row without ground-truth).
- `err`: null, or `"<ExceptionType>: <message>"` string. Truncated to
  500 chars to keep lines readable.

### `demo_log_worst.json` (feeds spec 09)

Top-3 worst queries per (pipeline, metric) by score, plus the
cross-pipeline delta leaderboard. Used by spec 09 to pick the 5-8
queries to annotate.

```json
{
  "run_metadata": {...},
  "worst_by_metric": {
    "faithfulness": {
      "baseline": [
        {"query_id": "q11", "score": 0.0, "question": "...", "answer_preview": "..."},
        ...
      ],
      "enhanced": [...]
    },
    ...
  },
  "largest_pipeline_deltas": [
    {"query_id": "q07", "metric": "context_recall",
     "baseline": 0.1, "enhanced": 0.9, "delta": 0.8},
    ...
  ]
}
```
`answer_preview` is first 200 chars of the answer — enough for a demo-
log entry, not a full reproduction.

### `test_set_hash` (repro metadata)

Defined as SHA-256 of the canonicalised CSV:
```python
import hashlib, csv, json
def test_set_hash(path: Path) -> str:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # Canonical form: JSON dump with sorted keys, one row per line, UTF-8.
    # Insensitive to Excel-ish cosmetic edits (trailing whitespace, CRLF).
    normalised = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False).strip()
                           for r in rows)
    return "sha256:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()
```
Anything that changes question text, ground truth, or categories
changes the hash. Whitespace-only CSV edits do not.

---

## Testing the evaluator code itself

RAGAS's `InstructorBaseRagasLLM` is not a trivial stub target — the
metrics do `isinstance` checks. Approach:

1. **Adapter/CSV tests** (`test_adapters.py`): no stubs needed. Load a
   tiny fixture CSV, a fixture results JSON, assert the resulting
   `SingleTurnSample` list. Pure unit.
2. **CRAG metrics tests** (`test_metrics.py`): hand-craft 5-row
   fixture of result dicts, call `compute_crag_metrics(fixture)`,
   assert numeric output. Wilcoxon test against known-answer inputs.
3. **RAGAS runner tests** (`test_ragas_eval.py`):
   - Monkey-patch each metric's `ascore` to return a deterministic
     `MetricResult(value=0.5)`. **Do NOT try to construct a real
     InstructorLLM in tests.**
   - Example:
     ```python
     async def fake_ascore(self, **kwargs):
         from ragas.metrics.result import MetricResult
         return MetricResult(value=0.5)
     monkeypatch.setattr(Faithfulness, "ascore", fake_ascore)
     ```
   - Assert: checkpoint file written; resume skips already-scored keys;
     abstain rows get `"skipped": "abstain"` for context metrics;
     error path persists `"score": null` and `"err": "..."`.

No test calls a real LLM or embedding API. CI-safe.

---

## A note on self-grading bias (for the report, spec 08)

We use Claude Sonnet 4 both to generate answers and to evaluate them.
This is not ideal. RAGAS prompts have documented
"same-model-judge-inflates-scores" effects — Liu et al. 2023 and the
RAGAS authors themselves acknowledge 0.05-0.15 inflation on
Faithfulness/Relevancy when the judge model matches the generator.

Mitigations **not** applied (budget/time):
- Second-opinion evaluator (different LLM as judge).
- Human annotation on a 10-query subsample.

Mitigations applied:
- Stricter prompt for the generator (ENHANCED_GENERATION_PROMPT
  requires citations + forbids speculation — reduces overconfident
  claims the same-model judge would rubber-stamp).
- Deterministic evaluator (`temperature=0`), so re-runs produce the
  same numbers and the bias is at least *consistent*.
- Report all four metrics, not just the two we'd "win" on — a biased
  judge would inflate both systems roughly equally, so relative
  ordering (baseline vs enhanced) is less affected than absolute
  scores.

This paragraph belongs in spec 08's Limitations section. It is written
here so the rationale doesn't get lost between specs.

---

## What to do if the numbers are bad

**Reality check before you assume a bug**: RAGAS numbers are noisy. A
Faithfulness of 0.65 on baseline and 0.75 on enhanced is a legitimate
result, not a bug.

- **If enhanced loses on Faithfulness**: baseline retrieves fewer docs
  → less context → fewer claims, each more grounded. Plausible.
  Report honestly, explain tradeoff with Context Recall.
- **If enhanced loses on Context Recall**: metadata filters too narrow.
  Check `metadata_filters_used` frequency. Consider rerunning with a
  less aggressive `ANALYZE_QUERY_PROMPT`.
- **If both pipelines score ≈0 on some metric**: evaluator LLM is not
  parsing the answer format. Check RAGAS prompt output in verbose mode;
  the answer might have an unescaped citation format the parser chokes
  on.
- **If p-value > 0.05**: say so in the report. "Enhanced pipeline showed
  a positive point estimate on all four metrics, but only Context Recall
  reached statistical significance at α=0.05 with N=25 queries." This
  is strong grad-school writing — the alternative (claiming significance
  you don't have) loses more marks than reporting the honest result.

The point of spec 07 is to make the comparison defensible, not
flattering. If the enhanced pipeline is worse on a metric, we report
that and discuss why — that's the dissertation-grade move.
