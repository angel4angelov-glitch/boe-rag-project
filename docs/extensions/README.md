# Extension plans — post-spec-07

These are **optional** additions on top of the core spec (01-10). Each
plan is scoped to a git branch (`feat/...`) so if any extension breaks,
we revert one branch; `main` keeps shipping.

## Tier A — zero-risk (new files, existing code untouched)

| Plan | Goal | Effort |
|------|------|--------|
| [A1 — LangSmith tracing](A1-langsmith-tracing.md) | Auto-capture LLM/retrieval traces for the demo | 30 min |
| [A2 — FastAPI service](A2-fastapi-service.md) | Wrap the pipeline as an HTTP endpoint | 1-2 hr |
| [A3 — Response cache](A3-response-cache.md) | Skip API calls on repeat queries | 30 min |
| [A4 — Docker + compose](A4-docker.md) | Containerise the FastAPI service | 1 hr |
| [A5 — Sonnet consistency check](A5-sonnet-consistency-check.md) | Cross-evaluator sanity check on worst-delta queries | 30 min |

## Tier B — low-medium risk (touches prompts/config, needs re-eval)

| Plan | Goal | Effort | Risk |
|------|------|--------|------|
| [B1 — q21 abstain fix](B1-fix-q21-abstain.md) | Flip abstain_correctness 0/3 → 1/3 on Fed question | 2-3 hr + $5 re-eval | Medium |
| [B2 — Expand test set](B2-expand-test-set.md) | N=25 → N=40 for better statistical power | 3-4 hr + $10 re-eval | Low code / high time |

## Tier C — deliberately skipped before submission

Listed here so "why didn't you do X" has an answer:
- **Comparative-Recall fix** — modifies `ANALYZE_QUERY_PROMPT` to stop
  over-narrowing. High risk: one-shot change cascades across all 25
  queries, no time to debug regressions before 2026-04-16.
- **Streaming responses** — structural conflict with the hallucination
  check node which needs the full answer. Architecture change.
- **Multi-turn conversation** — requires state beyond RAGState.

## Rule of thumb before the deadline

Every extension lives on its own branch (`git checkout -b feat/X`). If
the branch's tests pass AND the demo still works on `main` after merge,
merge. Otherwise, keep the branch unmerged — it's documented history
either way.
