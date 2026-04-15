"""Adapter layer: CSV test set + pipeline results → RAGAS ``SingleTurnSample``.

The enhanced pipeline uses a fixed abstain message (``ABSTAIN_MESSAGE``
below) that must stay identical to the constant defined in
``boe_rag.pipelines.nodes`` — detection of abstain rows downstream
depends on an exact-match (mod whitespace). If either string changes,
update both.

``SHOULD_ABSTAIN_IDS`` is the hand-curated set of query IDs for which
an abstain is the **correct** behaviour (out-of-corpus questions).
Extend when new out-of-scope test-set rows are added.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ragas import SingleTurnSample

# Single source of truth — imported verbatim from the pipeline so the
# abstain-detection string can never drift between producer and consumer.
# A unit test (test_abstain_message_single_source_of_truth) pins the
# identity.
from boe_rag.pipelines.nodes import ABSTAIN_MESSAGE

# Query IDs where an abstain is the correct answer. Currently only q21
# (the Federal Reserve out-of-corpus question). Keep this explicit rather
# than derived from category string so the policy is auditable.
SHOULD_ABSTAIN_IDS: frozenset[str] = frozenset({"q21"})


def is_abstain(answer: str) -> bool:
    return answer.strip() == ABSTAIN_MESSAGE


def load_test_set(path: Path) -> dict[str, dict]:
    """Parse ``data/test_set.csv`` into a qid-keyed dict.

    IDs are generated as ``q01``, ``q02``, ... in CSV row order, matching
    the IDs used by ``scripts/run_eval.py`` so results JSON keys line up.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {f"q{i + 1:02d}": row for i, row in enumerate(rows)}


def load_pipeline_results(path: Path) -> dict[str, dict]:
    """Load a ``{baseline,enhanced}_results.json`` into a qid-keyed dict."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _chunk_text(source: dict) -> str:
    """Prefer full ``text`` (new format); fall back to ``text_preview``."""
    return source.get("text") or source.get("text_preview") or ""


def results_to_samples(
    results: dict[str, dict],
    test_set: dict[str, dict],
) -> list[tuple[str, SingleTurnSample]]:
    """Build (qid, SingleTurnSample) pairs for RAGAS scoring.

    Only IDs present in BOTH results and test_set are emitted — mismatched
    IDs (stale results or an expanded test set) are silently skipped
    rather than raising, so a partial re-run doesn't block scoring the
    good rows.
    """
    pairs: list[tuple[str, SingleTurnSample]] = []
    for qid, row in test_set.items():
        res = results.get(qid)
        if res is None:
            continue
        contexts = [
            _chunk_text(s) for s in res.get("sources", [])
            if _chunk_text(s)
        ]
        pairs.append((qid, SingleTurnSample(
            user_input=row["question"],
            response=res.get("answer", ""),
            retrieved_contexts=contexts,
            reference=row["expected_answer"],
        )))
    return pairs
