# A2 — FastAPI service wrapper

## Goal
Expose the enhanced pipeline as `POST /query` so it can be demoed via
curl/a browser instead of only via Python REPL. "Deployed as HTTP
service" on the CV is a legitimate claim.

## Why
- **Demo log (spec 09)**: a terminal screenshot showing
  `curl -X POST .../query -d '{"question":"..."}' → JSON` is a stronger
  demo than notebook cells.
- **Report — "what's next" section**: shortens the production-gap list.
- **CV**: "Built CRAG pipeline and deployed as FastAPI service with
  streaming JSON responses" vs "Built CRAG pipeline". Both honest; one
  gets more interviews.

## Risk: ZERO
New directory (`service/`), new dependencies added. Existing package
imports the pipeline — we don't modify it. If FastAPI breaks, core
tests still pass, notebook still works.

## Scope
**New files**:
- `service/__init__.py`
- `service/main.py` — FastAPI app, `/query` route, `/health` route
- `service/schemas.py` — pydantic `QueryRequest`, `QueryResponse`
- `tests/service/test_api.py` — TestClient tests (mock pipeline)

**Modified files**:
- `pyproject.toml` — add `fastapi`, `uvicorn` to optional-dependencies
  (`service = [...]`) so core install stays lean

## Steps
1. Add optional-deps section:
   ```toml
   [project.optional-dependencies]
   service = ["fastapi", "uvicorn[standard]", "pydantic"]
   ```
2. `pip install -e ".[service]"`
3. Write `service/schemas.py`:
   ```python
   class QueryRequest(BaseModel):
       question: str
       pipeline: Literal["baseline", "enhanced"] = "enhanced"

   class QueryResponse(BaseModel):
       answer: str
       sources: list[SourceItem]
       is_grounded: bool | None
       chunks_retrieved: int
       chunks_used: int
       pipeline_trace: list[str]
   ```
4. Write `service/main.py`:
   - Lazy-init pipelines on startup (avoid cold-starts on every request)
   - `POST /query` → `EnhancedPipeline().run(req.question)` → map to
     `QueryResponse`
   - `GET /health` → `{"status": "ok"}`
   - Add middleware for request logging
5. Write `tests/service/test_api.py`:
   - Stub the pipeline, assert request/response schemas round-trip
   - 400 on empty question, 422 on invalid payload
6. Smoke test: `uvicorn service.main:app --reload`, curl `/query`.

## Test plan
- All 203 existing tests pass unchanged.
- New: 3-5 API tests with stubbed pipeline (no real LLM calls).
- Manual: `uvicorn service.main:app`, one curl → answer JSON.
- Screenshot of curl call for demo log.

## Rollback
`service/` directory is isolated. Delete it, remove the optional-deps
entry. No changes to `src/boe_rag/`.

## Effort: 1-2 hours

## Branch: `feat/fastapi-service`
