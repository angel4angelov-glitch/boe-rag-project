"""Async RAGAS runner with checkpoint, resume, and abstain-skip.

One line of JSONL per ``(pipeline, query_id, metric)`` tuple. On resume,
existence of a matching line is "already done — skip". Errors and
abstain-skips are persisted as null scores with structured fields so a
retry pass can target only the failures.

Abstain-skip is upfront: if the metric requires retrieved_contexts
(Faithfulness / ContextPrecisionWithReference / ContextRecall) and the
sample has none, we log ``skipped: "abstain"`` without invoking the
metric. This avoids burning an API call to get a deterministic
ValueError — and keeps the checkpoint deterministic.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from ragas import SingleTurnSample

logger = logging.getLogger(__name__)


# RAGAS 0.4.3 collections metrics that raise on empty retrieved_contexts.
# Verified by reading the metric source — each ascore begins with a
# ``if not retrieved_contexts: raise ValueError`` block. AnswerRelevancy
# does not take retrieved_contexts at all, so it's absent here.
CONTEXT_REQUIRED_METRICS: frozenset[str] = frozenset({
    "faithfulness",
    "context_precision_with_reference",
    "context_recall",
})


def _dispatch_kwargs(metric: Any, sample: SingleTurnSample) -> dict:
    """Filter ``sample`` fields to only the ones ``metric.ascore`` declares.

    Each collections metric has a different ``ascore`` signature — passing
    the full kwargs bag uniformly would raise TypeError on any
    unexpected kwarg. Introspect the signature each call.
    """
    full = {
        "user_input": sample.user_input,
        "response": sample.response,
        "retrieved_contexts": list(sample.retrieved_contexts or []),
        "reference": sample.reference,
    }
    params = inspect.signature(metric.ascore).parameters
    return {k: v for k, v in full.items() if k in params}


def load_done_keys(path: Path) -> set[tuple[str, str, str]]:
    """Parse existing JSONL and return the ``(pipeline, query_id, metric)`` set."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["pipeline"], r["query_id"], r["metric"]))
            except (json.JSONDecodeError, KeyError):
                # Skip corrupt lines rather than failing the resume.
                logger.warning("skipping corrupt JSONL line in %s", path)
    return done


def _append_jsonl(
    path: Path,
    *,
    pipeline: str,
    query_id: str,
    metric: str,
    score: float | None,
    skipped: str | None,
    err: str | None,
) -> None:
    record = {
        "pipeline": pipeline,
        "query_id": query_id,
        "metric": metric,
        "score": score,
        "skipped": skipped,
        "err": err,
        "ts": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _score_one(
    metric: Any,
    sample: SingleTurnSample,
    *,
    pipeline: str,
    query_id: str,
    out_path: Path,
    sem: asyncio.Semaphore,
) -> None:
    # Abstain-skip: upfront, before we burn a semaphore slot.
    if metric.name in CONTEXT_REQUIRED_METRICS and not sample.retrieved_contexts:
        _append_jsonl(
            out_path, pipeline=pipeline, query_id=query_id, metric=metric.name,
            score=None, skipped="abstain", err=None,
        )
        return

    async with sem:
        score: float | None = None
        err: str | None = None
        try:
            result = await metric.ascore(**_dispatch_kwargs(metric, sample))
            score = float(result.value)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:500]}"
            logger.warning(
                "metric %s failed on %s/%s: %s", metric.name, pipeline, query_id, err,
            )

    _append_jsonl(
        out_path, pipeline=pipeline, query_id=query_id, metric=metric.name,
        score=score, skipped=None, err=err,
    )


async def run_ragas(
    *,
    samples: Iterable[SingleTurnSample],
    query_ids: Iterable[str],
    pipeline_name: str,
    metrics: Iterable[Any],
    out_path: Path,
    resume: bool = True,
    concurrency: int = 4,
) -> None:
    """Score every (sample, metric) pair; persist to JSONL.

    Args:
        samples: SingleTurnSample per query.
        query_ids: matching ``qNN`` identifiers — zip'd positionally with
            samples.
        pipeline_name: "baseline" or "enhanced" — stamped on every record.
        metrics: instantiated RAGAS collections metrics (any order).
        out_path: JSONL output. Parent dir must exist.
        resume: when True, skip tuples already present in ``out_path``.
            When False, truncate ``out_path`` and score everything fresh.
        concurrency: max in-flight metric calls. Default 4 tuned for
            tier-1 Anthropic; drop to 2 at worst, raise to 8 at tier-2.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
        out_path.unlink()

    done = load_done_keys(out_path) if resume else set()

    samples = list(samples)
    query_ids = list(query_ids)
    metrics = list(metrics)

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for metric in metrics:
        for sample, qid in zip(samples, query_ids, strict=True):
            key = (pipeline_name, qid, metric.name)
            if key in done:
                continue
            tasks.append(_score_one(
                metric, sample,
                pipeline=pipeline_name, query_id=qid,
                out_path=out_path, sem=sem,
            ))

    if not tasks:
        logger.info("run_ragas: nothing to do for %s (all %d keys present)",
                    pipeline_name, len(samples) * len(metrics))
        return

    logger.info("run_ragas: scoring %d (sample, metric) pairs for %s",
                len(tasks), pipeline_name)
    await asyncio.gather(*tasks)


# ── Factory for the four metrics we use ─────────────────────


def build_metrics(
    *, llm: Any, embeddings: Any,
) -> list[Any]:
    """Instantiate the 4 RAGAS metrics with a shared evaluator LLM + embeddings.

    Kept here rather than in the CLI so tests can stub them out; real
    callers hit this from ``scripts/run_ragas.py`` which supplies the
    llm_factory + OpenAIEmbeddings objects.
    """
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecisionWithReference(llm=llm),
        ContextRecall(llm=llm),
    ]
