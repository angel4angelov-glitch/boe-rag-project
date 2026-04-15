# A1 — LangSmith tracing (v2)

> **v2 changelog**: v1 was technically correct but operationally naive.
> Fixed the overclaim on coverage, pinned env-var naming to the
> installed SDK version, isolated tests, explicitly disabled tracing
> during RAGAS, specified tagging strategy, and made the @traceable
> decorator required (not optional).

## Goal
Every **pipeline invocation** captures a parent trace; every
**ChatAnthropic / LangGraph-node** call captures as a nested child
trace. Viewable in a `boe-rag` project on smith.langchain.com, tagged
by pipeline + query id so the dashboard is filterable.

## Honest coverage

| Captured | Mechanism |
|---|---|
| `pipeline.run(q)` top-level span | `@traceable(name="...")` decorator |
| Every ChatAnthropic `.invoke()` / `.batch()` | LangChain auto-trace |
| `with_structured_output(QueryFilters)` | Auto-traced as child chain |
| `with_retry(...)` attempts | Each retry appears as its own child |
| LangGraph node execution + edge routing | langgraph native tracing |
| Claude token usage per call | auto |

| **NOT captured** | Why |
|---|---|
| `chroma.collection.query(...)` | Raw SDK, not a LangChain object — surfaces only as a "retrieve" node box with its state I/O |
| `cohere_client.rerank(...)` | Same — opaque inside the "rerank" node |
| OpenAI embeddings inside ChromaDB | Called via the collection, not directly — no trace |
| RAGAS evaluator calls | Explicitly **disabled** during RAGAS runs (see below) |

For the demo log, this is fine: the LLM calls are where the interesting
behaviour lives (grading verdicts, structured-output filters,
hallucination checks). Chroma/Cohere are deterministic given their
inputs — not interesting to trace.

## Risk: LOW (not zero — demoted from v1)
Network calls are async-fire-and-forget — LangSmith outage shouldn't
block the pipeline, but this must be verified once before the demo.
Tests must be isolated (see below) or they'll spam the project and
blow the free-tier quota.

## Free-tier quota
5k traces/month on the free plan. One pipeline run ≈ 12 traces (nodes +
LLM calls). One full 25-query eval ≈ 300 traces. Two or three full
evals + ad-hoc demo queries easily stays under quota — IF tests and
RAGAS runs are disabled from tracing. Without that guard we'd blow it
in an afternoon.

## Scope
**New files**:
- `docs/screenshots/.gitkeep` (dir for demo-log trace screenshots)

**Modified files**:
- `.env.example` — add 3 keys (commented)
- `src/boe_rag/pipelines/baseline.py` — `@traceable` on `run()`
- `src/boe_rag/pipelines/enhanced.py` — `@traceable` on `run()`
- `tests/conftest.py` (create) — force `LANGSMITH_TRACING=false` in tests
- `scripts/run_ragas.py` — explicitly set
  `os.environ["LANGSMITH_TRACING"]="false"` at entry, with a one-line
  comment about why
- `README.md` — 1-paragraph section on how to view traces

## Env-var naming (pinned)
LangChain ≥0.2 honours **both** `LANGCHAIN_TRACING_V2=true` AND
`LANGSMITH_TRACING=true`. The newer `LANGSMITH_*` is canonical; the
SDK deprecation notice points that way. Use `LANGSMITH_*`.

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...           # from smith.langchain.com/settings
LANGSMITH_PROJECT=boe-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com   # default, explicit for portability
```

## Tagging strategy
Every top-level trace gets tags + metadata so the dashboard can
filter. Done via the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(
    name="EnhancedPipeline.run",
    tags=["pipeline:enhanced"],
    metadata={"model": GENERATION_MODEL},
)
def run(self, query: str) -> PipelineResult:
    ...
```

For per-call tagging (e.g. query_id during eval), `run_eval.py` can
wrap each call with a `with tracing_context(tags=["q01"]):` block.

## Steps
1. Sign up free at smith.langchain.com. Generate API key in Settings →
   API Keys. Note the project is created automatically on first trace.
2. Add the 3 env vars to local `.env` (real values) and `.env.example`
   (keys only, no values).
3. Create `tests/conftest.py`:
   ```python
   import os, pytest
   @pytest.fixture(autouse=True)
   def _disable_langsmith_in_tests(monkeypatch):
       monkeypatch.setenv("LANGSMITH_TRACING", "false")
   ```
   This fires on every test, overriding any ambient env var.
4. Add `@traceable` to `BaselinePipeline.run` and `EnhancedPipeline.run`
   with tags + metadata as above.
5. In `scripts/run_ragas.py` top: `os.environ["LANGSMITH_TRACING"] =
   "false"` before any imports that might initialize tracing.
   Comment: "RAGAS evaluator calls burn free-tier quota; trace pipeline
   runs only."
6. Run one query end-to-end via a notebook or repl:
   ```python
   from boe_rag.pipelines import EnhancedPipeline
   r = EnhancedPipeline().run("What was the vote split in February 2026?")
   ```
7. Verify on smith.langchain.com:
   - Project `boe-rag` exists
   - One trace appears with name `EnhancedPipeline.run`
   - Expanding shows: analyze_query (structured LLM call, QueryFilters
     visible in output), retrieve (opaque box), grade_documents (10
     LLM children), rerank, generate, check_hallucination
   - Retry attempts on with_retry nested under their parent call
8. Screenshot a good-looking trace → `docs/screenshots/langsmith-trace-enhanced.png`.
9. Screenshot a baseline trace for contrast → `langsmith-trace-baseline.png`.

## Test plan
- All 203 existing tests pass with the conftest.py fixture (verify no
  LangSmith network calls during `pytest`).
- Manual: one query produces a trace matching the expected tree above.
- Manual: running RAGAS with `LANGSMITH_TRACING=true` in shell → still
  no traces added (script override wins).
- Failure-mode check: set `LANGSMITH_API_KEY=invalid`, run one pipeline
  query, confirm the pipeline still completes (trace send fails
  silently, pipeline answer unchanged).

## Rollback
Unset `LANGSMITH_TRACING` in `.env`. The `@traceable` decorator
becomes a no-op when tracing is off. No code revert needed. For full
rollback, delete the decorator imports and `conftest.py` fixture.

## Effort (honest)
60-90 minutes:
- 15 min: signup + env var plumbing
- 15 min: conftest + RAGAS isolation
- 15 min: decorator + one end-to-end test
- 15-30 min: iterate on trace quality (tags, metadata, nested structure)
- 15 min: screenshots + README paragraph

## Branch: `feat/langsmith`

## Failure-mode verification (mandatory before merging)
Before `git merge`, run this checklist:
1. `LANGSMITH_API_KEY=invalid python -c "from boe_rag.pipelines import EnhancedPipeline; EnhancedPipeline().run('test')"` → pipeline succeeds, trace send silently errors.
2. `LANGSMITH_TRACING=false python scripts/run_eval.py` → no traces appear in dashboard.
3. `pytest` → no traces appear in dashboard (conftest working).
4. 1 notebook query with tracing on → trace visible, properly nested,
   tags set correctly.

If any of these fail, do NOT merge to main.
