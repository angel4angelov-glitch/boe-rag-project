"""Evaluation package: RAGAS runner + CRAG metrics + statistical tests.

Imports here are convenience re-exports so callers can write:

    from boe_rag.evaluation import load_test_set, run_ragas

rather than drilling into submodules. Keep this module side-effect-free
(no API calls, no filesystem I/O on import).
"""

from boe_rag.evaluation.adapters import (
    ABSTAIN_MESSAGE,
    SHOULD_ABSTAIN_IDS,
    is_abstain,
    load_pipeline_results,
    load_test_set,
    results_to_samples,
)
from boe_rag.evaluation.metrics import (
    bootstrap_paired_delta_ci,
    compute_crag_metrics,
    holm_bonferroni,
    paired_wilcoxon,
    per_category_means,
)
from boe_rag.evaluation.ragas_eval import (
    CONTEXT_REQUIRED_METRICS,
    build_metrics,
    load_done_keys,
    run_ragas,
)
from boe_rag.evaluation.repro import collect_run_metadata, compute_test_set_hash

__all__ = [
    "ABSTAIN_MESSAGE",
    "CONTEXT_REQUIRED_METRICS",
    "SHOULD_ABSTAIN_IDS",
    "bootstrap_paired_delta_ci",
    "build_metrics",
    "collect_run_metadata",
    "compute_crag_metrics",
    "compute_test_set_hash",
    "holm_bonferroni",
    "is_abstain",
    "load_done_keys",
    "load_pipeline_results",
    "load_test_set",
    "paired_wilcoxon",
    "per_category_means",
    "results_to_samples",
    "run_ragas",
]
