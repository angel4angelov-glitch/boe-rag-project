"""Run-metadata collection.

Every top-level evaluation output gets a ``run_metadata`` block so
re-runs are distinguishable and comparable. Includes git SHA (if any),
library versions, model names, and a content-hash of the test set so
silent edits between runs are detected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from boe_rag import config


def compute_test_set_hash(path: Path) -> str:
    """SHA-256 of the canonicalised test-set CSV.

    Canonical form: each row parsed by ``csv.DictReader``, trailing
    whitespace stripped from every value, then dumped as JSON with
    sorted keys. One row per line, UTF-8. Insensitive to Excel-ish
    cosmetic edits (trailing spaces, CRLF). Changes when question text,
    categories, or ground truth change.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = [
        json.dumps(
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()},
            sort_keys=True,
            ensure_ascii=False,
        )
        for row in rows
    ]
    normalised = "\n".join(lines)
    return "sha256:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect_run_metadata(
    *,
    test_set_path: Path,
    evaluator_model: str | None = None,
    evaluator_temperature: float = 0.0,
) -> dict:
    """Build the run_metadata block for output files.

    Counts test-set rows by parsing the CSV; captures git state, key
    library versions, all four model names from config, and the
    evaluator settings so a reviewer can tell exactly which run
    produced which numbers.
    """
    with test_set_path.open(newline="", encoding="utf-8") as f:
        n_queries = sum(1 for _ in csv.DictReader(f))

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "ragas_version": _pkg_version("ragas"),
        "scipy_version": _pkg_version("scipy"),
        "generation_model": config.GENERATION_MODEL,
        "grading_model": config.GRADING_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "evaluator_model": evaluator_model or config.GENERATION_MODEL,
        "evaluator_temperature": evaluator_temperature,
        "n_queries": n_queries,
        "test_set_hash": compute_test_set_hash(test_set_path),
    }
