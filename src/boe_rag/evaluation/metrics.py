"""Statistical + CRAG-specific metrics on per-sample scores.

Separated from ``ragas_eval.py`` so the aggregation layer can be
unit-tested on fixture inputs with no LLM/embedding clients in sight.

Contents:
  - ``holm_bonferroni`` — multiple-testing correction over the 4 RAGAS
    p-values. Family-wise error rate ~0.19 at α=0.05 uncorrected.
  - ``paired_wilcoxon`` — one-sided (H1: enhanced > baseline) paired test
    with ``zero_method="zsplit"`` so tied-pair mass isn't discarded.
  - ``bootstrap_paired_delta_ci`` — BCa 95% CI on the paired mean delta
    via ``scipy.stats.bootstrap(paired=True, method='BCa')``.
  - ``compute_crag_metrics`` — rewrite/hallucination/filter/rerank rates
    from the enhanced pipeline's result JSON.
  - ``per_category_means`` — group per-query scores by category column.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import bootstrap, wilcoxon


# ── Multiple-testing correction ────────────────────────────


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the same order as input.

    Sort p-values ascending; the i-th smallest (0-indexed rank i) gets
    multiplier (n - i). Running max enforces monotonicity. Clip to [0, 1].
    Slightly more powerful than vanilla Bonferroni at the same FWER.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (n - rank) * pvalues[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted


# ── Paired Wilcoxon ───────────────────────────────────────


def paired_wilcoxon(
    baseline_scores: Iterable[float],
    enhanced_scores: Iterable[float],
) -> dict:
    """One-sided paired Wilcoxon (H1: enhanced > baseline).

    Drops pairs where either side is None or NaN. Uses
    ``zero_method="zsplit"`` so ties (both systems score the same) are
    split 50/50 rather than discarded — important on bounded [0,1]
    metrics where ties are frequent.

    Returns ``statistic``, ``p_value``, ``n_pairs``. p_value is None
    when N < 2 or all pairs are identical (Wilcoxon undefined).
    """
    pairs = [
        (b, e)
        for b, e in zip(baseline_scores, enhanced_scores)
        if b is not None and e is not None and not _isnan(b) and not _isnan(e)
    ]
    if len(pairs) < 2:
        return {"statistic": None, "p_value": None, "n_pairs": len(pairs)}

    baseline, enhanced = zip(*pairs, strict=True)
    try:
        result = wilcoxon(
            x=list(baseline),
            y=list(enhanced),
            alternative="less",
            zero_method="zsplit",
        )
        stat = float(result.statistic)
        p = float(result.pvalue)
    except ValueError:
        # Raised when all differences are zero — no signal to test.
        return {"statistic": None, "p_value": None, "n_pairs": len(pairs)}
    return {"statistic": stat, "p_value": p, "n_pairs": len(pairs)}


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x


# ── Bootstrap CI ───────────────────────────────────────────


def bootstrap_paired_delta_ci(
    baseline_scores: Iterable[float],
    enhanced_scores: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """95% BCa CI on the paired mean delta (enhanced - baseline).

    BCa (bias-corrected and accelerated) handles small, skewed samples
    better than the percentile method. ``paired=True`` ensures pairs
    are resampled together. Returns (None, None) when N < 3 — BCa
    can't compute the acceleration with fewer pairs.
    """
    pairs = [
        (b, e)
        for b, e in zip(baseline_scores, enhanced_scores)
        if b is not None and e is not None and not _isnan(b) and not _isnan(e)
    ]
    if len(pairs) < 3:
        return None, None

    baseline, enhanced = zip(*pairs, strict=True)

    def _delta_mean(b, e):
        return float(np.mean(np.asarray(e) - np.asarray(b)))

    try:
        ci = bootstrap(
            data=(list(baseline), list(enhanced)),
            statistic=_delta_mean,
            paired=True,
            vectorized=False,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="BCa",
            rng=np.random.default_rng(seed),
        ).confidence_interval
        return float(ci.low), float(ci.high)
    except Exception:
        # scipy can raise DegenerateDataWarning → Error in edge cases
        return None, None


# ── CRAG-specific metrics ──────────────────────────────────


def compute_crag_metrics(
    enhanced_results: dict[str, dict],
    *,
    should_abstain_ids: Iterable[str],
) -> dict:
    """Rates + means computed from the enhanced pipeline's per-query state.

    See spec 07 for definitions. The numerator / denominator conventions:
      - rates are over ALL N queries (including abstains), unless the
        metric is conditioned on a state (e.g. rerank_top1_change_rate
        is over queries with rerank data).
      - means are simple arithmetic means over N.
    """
    should = set(should_abstain_ids)
    rows = list(enhanced_results.values())
    n = len(rows)
    if n == 0:
        return {}

    from boe_rag.evaluation.adapters import is_abstain

    rewrites = sum(1 for r in rows if r.get("crag_rewrites", 0) > 0)
    hallucinations = sum(1 for r in rows if r.get("is_grounded") is False)
    with_filters = sum(1 for r in rows if r.get("metadata_filters_used") is not None)
    abstains = [qid for qid, r in enhanced_results.items() if is_abstain(r.get("answer", ""))]
    correct_abstains = sum(1 for qid in abstains if qid in should)

    # rewrite_recovery: of triggered rewrites, how many ended grounded + non-abstain
    rewrite_recovered = sum(
        1 for r in rows
        if r.get("crag_rewrites", 0) > 0
        and r.get("is_grounded") is True
        and not is_abstain(r.get("answer", ""))
    )
    # hallucination_recovery: of retries, how many final is_grounded True
    retry_rows = [r for r in rows if r.get("hallucination_retries", 0) > 0]
    hallucination_recovered = sum(1 for r in retry_rows if r.get("is_grounded") is True)

    # Rerank top-1 change (over rows with rerank data)
    rerank_rows = [
        r for r in rows
        if r.get("pre_rerank_ids") and r.get("post_rerank_ids")
    ]
    top1_changed = sum(
        1 for r in rerank_rows
        if r["pre_rerank_ids"][0] != r["post_rerank_ids"][0]
    )

    return {
        "n": n,
        "rewrite_trigger_rate": rewrites / n,
        "rewrite_recovery_rate": (rewrite_recovered / rewrites) if rewrites else None,
        "hallucination_flag_rate": hallucinations / n,
        "hallucination_recovery_rate": (
            hallucination_recovered / len(retry_rows) if retry_rows else None
        ),
        "metadata_filter_rate": with_filters / n,
        "mean_chunks_retrieved": sum(r.get("chunks_retrieved", 0) for r in rows) / n,
        "mean_chunks_used": sum(r.get("chunks_used", 0) for r in rows) / n,
        "rerank_top1_change_rate": (top1_changed / len(rerank_rows)) if rerank_rows else None,
        "abstain_rate": len(abstains) / n,
        "abstain_correctness": (correct_abstains / len(abstains)) if abstains else None,
        # B1: recall over the should-abstain set. Precision (above) punishes
        # false-positive abstains; recall measures whether we caught the
        # true-positives. For the report's scope-check narrative, recall is
        # the cleaner single-metric story (1/1 = 1.0 when q21 is caught).
        "should_abstain_recall": (correct_abstains / len(should)) if should else None,
        "abstain_ids": abstains,
        "correct_abstain_ids": [qid for qid in abstains if qid in should],
        "missed_abstain_ids": [qid for qid in should if qid not in abstains],
    }


# ── Per-category aggregation ──────────────────────────────


def per_category_means(
    scores: dict[str, dict],
) -> dict[str, dict]:
    """Group ``{qid: {category, score}}`` → ``{category: {mean, n}}``.

    Ignores rows with ``score is None``. Empty categories are omitted
    (no entry at all) rather than reported with NaN — cleaner downstream
    CSV output.
    """
    buckets: dict[str, list[float]] = {}
    for row in scores.values():
        s = row.get("score")
        cat = row.get("category")
        if s is None or cat is None:
            continue
        buckets.setdefault(cat, []).append(float(s))

    return {
        cat: {"mean": sum(xs) / len(xs), "n": len(xs)}
        for cat, xs in buckets.items()
        if xs
    }
