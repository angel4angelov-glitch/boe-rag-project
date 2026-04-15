# A3 — Response cache

## Goal
Identical queries return cached `PipelineResult` instead of re-running
the whole graph. Keyed on `(pipeline_name, question, prompt_hash)` so
prompt changes invalidate.

## Why
- **Demo safety**: the demo script won't 429 on repeat questions during
  a live walkthrough.
- **Cost**: saves API spend during iterative report-writing.
- **Not a grade-lifter**: report mark-wise this is minor. Demo
  reliability is the real win.

## Risk: ZERO
Cache wraps `.run()` externally. If the cache is cold or corrupted, the
pipeline runs normally. Disabling it is one env var.

## Scope
**New files**:
- `src/boe_rag/cache/__init__.py`
- `src/boe_rag/cache/disk_cache.py` — diskcache wrapper
- `tests/cache/test_disk_cache.py`

**Modified files**:
- `pyproject.toml` — add `diskcache` to optional-deps (`cache = [...]`)
- `scripts/run_eval.py` — optional `--use-cache` flag

## Steps
1. `pip install diskcache`
2. Write `disk_cache.py`:
   ```python
   class CachedPipeline:
       def __init__(self, pipeline, cache_dir="./.cache/pipeline"):
           self._inner = pipeline
           self._cache = diskcache.Cache(cache_dir)

       def run(self, query: str) -> PipelineResult:
           key = self._cache_key(query)
           hit = self._cache.get(key)
           if hit is not None:
               return _deserialize(hit)
           result = self._inner.run(query)
           self._cache[key] = _serialize(result)
           return result
   ```
3. Key = `sha256(pipeline_name + GENERATION_MODEL + prompt_hash + question)`.
   Prompt hash invalidates cache when we change prompts.
4. Serialize `PipelineResult` to dict (already done in `run_eval.py`).
5. Tests: cache miss → inner called. Cache hit → inner NOT called
   (assert call count unchanged). Different question → cache miss.

## Test plan
- 5 cache tests: miss / hit / invalidation / serialization round-trip
- All existing tests pass unchanged
- Manual: run same query twice → second call <100ms

## Rollback
Drop `--use-cache` flag in scripts. Delete `src/boe_rag/cache/`. No
dependency on cache anywhere outside the wrapper.

## Effort: 30-45 minutes

## Branch: `feat/response-cache`
