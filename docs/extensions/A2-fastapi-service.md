# A2 — FastAPI service wrapper (v2)

> **v2 changelog**: v1 was 60% honest. Fixed the startup-event
> deprecation, explicit sync-vs-async concurrency model, API-key
> placeholder, real /ready health check, pipeline init strategy
> (enhanced only by default, baseline on demand), OpenAPI docs
> mention, error-surface decision.

## Goal
`POST /query` returns a JSON `PipelineResult` for the given question.
Callable via `curl` for the demo log. `/docs` auto-generates the
OpenAPI schema view. Signals "deployed as HTTP service" on the CV
without overstating.

## Why
- **Demo log (spec 09)**: `curl` screenshot + `/docs` screenshot beats
  notebook cells for impact.
- **Tier-A for the report**: "exposed as an HTTP service" adds one
  concrete achievement without touching the pipeline.
- **Foundation for A4 (Docker)**: Docker needs a process to run;
  FastAPI is that process.

## Risk: ZERO
Lives in a new top-level `service/` directory. The pipeline package
isn't modified. If FastAPI import fails, core tests + notebooks still
work.

## Scope
**New files**:
- `service/__init__.py`
- `service/main.py` — FastAPI app, `lifespan` manager, `/query`, `/health`, `/ready`
- `service/schemas.py` — pydantic `QueryRequest`, `QueryResponse`, `SourceItem`
- `service/auth.py` — stub API-key dependency (optional via env var)
- `tests/service/__init__.py`
- `tests/service/test_api.py` — TestClient tests with stubbed pipeline

**Modified files**:
- `pyproject.toml` — add `fastapi`, `uvicorn[standard]`, `httpx` to a
  new `[project.optional-dependencies.service]` group so core install
  stays lean

## Concurrency model
The pipeline's `run()` is **synchronous** (LangGraph `invoke`, sync
`ChatAnthropic.invoke`). FastAPI handles sync path functions by
dispatching them to a threadpool — fine for demo traffic. For true
async we'd need `ainvoke` and `await` which is Tier C.

Explicit in the code: `def query(req: QueryRequest)` (sync) so the
threadpool dispatch is automatic.

## Pipeline init strategy
- **Enhanced pipeline** is initialised in `lifespan` startup — cold
  start cost (~2-3s) hits once, not per request.
- **Baseline** is init'd lazily on first `?pipeline=baseline` request.
  Saves ~1s on normal startup; negligible cost when a demo toggles it.

## API-key gate
A single header `X-API-Key` is checked **if** `SERVICE_API_KEY` env
var is set. When unset (default): no auth. This is a placeholder, not
production auth — documented as such in `service/auth.py`.

## Error surface
The pipeline currently swallows exceptions inside `run()` and returns
a `PipelineResult` with `answer="[Pipeline error: ...]"`. The FastAPI
layer **re-raises as HTTP 500** when the answer starts with
`[Pipeline error`. This is the contract:
- 200 OK: real answer or intentional abstain (`is_abstain=true`).
- 500: pipeline crashed. Body includes the error string for debugging.

## Schema shape
```python
class SourceItem(BaseModel):
    chunk_id: str
    score: float
    text: str                          # full chunk (baseline + enhanced)
    text_preview: str                  # first 200 chars for UI truncation
    metadata: dict | None = None       # enhanced only

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    pipeline_name: Literal["baseline", "enhanced"]
    chunks_retrieved: int
    chunks_used: int
    is_grounded: bool | None
    is_abstain: bool
    crag_rewrites: int
    hallucination_retries: int
    metadata_filters_used: dict | None
    pipeline_trace: list[str]
```

## Endpoints
- `POST /query` — run a query
- `GET /health` — liveness (always 200; just says "ok")
- `GET /ready` — readiness (checks the ChromaDB collection is
  reachable via `.count()`; 200 if OK, 503 otherwise)
- `GET /docs` — auto-generated OpenAPI Swagger UI
- `GET /redoc` — auto-generated ReDoc UI

## Steps
1. Add optional-deps:
   ```toml
   [project.optional-dependencies]
   service = ["fastapi", "uvicorn[standard]", "httpx"]
   ```
2. `pip install -e ".[service]"`
3. Write `service/schemas.py` — Pydantic models.
4. Write `service/auth.py` — `require_api_key()` dependency.
5. Write `service/main.py` — `lifespan`, routes, error mapping.
6. Write `tests/service/test_api.py`:
   - TestClient with the real FastAPI app
   - Stub `EnhancedPipeline.run` with monkeypatch to return a canned
     `PipelineResult` (no LLM calls).
   - Assertions on: happy path 200, error path 500, auth 401/200,
     `/health` and `/ready`.
7. Smoke test: `uvicorn service.main:app --reload`. Hit:
   ```bash
   curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"question":"What was the Feb 2026 MPC vote split?"}'
   ```
8. Screenshot `/docs` Swagger UI → `docs/screenshots/fastapi-swagger.png`.
9. Screenshot curl call → `docs/screenshots/fastapi-curl.png`.

## Test plan
- New: `test_api.py` with 5-7 tests (happy, error, auth on/off, health, ready)
- All 210 existing tests still pass unchanged.
- Manual: uvicorn up → curl returns valid JSON matching `QueryResponse`.
- `/docs` loads + shows `/query` endpoint with schema.

## Rollback
Delete `service/` + `tests/service/`, remove `[service]` optional-deps
entry. No changes to `src/boe_rag/`.

## Effort (honest): 1.5-2 hours
- 15 min: deps install + schemas
- 30 min: main.py with lifespan + routes + error mapping
- 30 min: tests with stubbed pipeline
- 15 min: manual smoke test + two screenshots
- 15 min: README update pointing to `/docs`

## Branch: `feat/fastapi-service`
