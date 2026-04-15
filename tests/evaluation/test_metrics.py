"""Statistics + CRAG metrics."""

from __future__ import annotations

import pytest

from boe_rag.evaluation.metrics import (
    bootstrap_paired_delta_ci,
    compute_crag_metrics,
    holm_bonferroni,
    paired_wilcoxon,
    per_category_means,
)


# ── Holm-Bonferroni ─────────────────────────────────────────


class TestHolmBonferroni:
    def test_sorted_input_known_values(self) -> None:
        # From spec hand-calculation
        adj = holm_bonferroni([0.012, 0.02, 0.04, 0.05])
        assert adj[0] == pytest.approx(0.048)
        assert adj[1] == pytest.approx(0.06)
        assert adj[2] == pytest.approx(0.08)
        assert adj[3] == pytest.approx(0.08)  # monotonic non-decreasing

    def test_all_ones_clip(self) -> None:
        adj = holm_bonferroni([0.5, 0.5, 0.5, 0.5])
        assert all(a == 1.0 for a in adj)

    def test_preserves_input_order(self) -> None:
        raw = [0.5, 0.01, 0.5, 0.5]
        adj = holm_bonferroni(raw)
        assert adj[1] == pytest.approx(0.04)  # 4 * 0.01 = 0.04
        assert adj[0] == 1.0 and adj[2] == 1.0 and adj[3] == 1.0

    def test_empty_list(self) -> None:
        assert holm_bonferroni([]) == []

    def test_single_pvalue_unchanged(self) -> None:
        assert holm_bonferroni([0.03]) == [pytest.approx(0.03)]


# ── Paired Wilcoxon ─────────────────────────────────────────


class TestPairedWilcoxon:
    def test_detects_one_sided_superiority(self) -> None:
        baseline = [0.1] * 10
        enhanced = [0.5] * 10
        result = paired_wilcoxon(baseline, enhanced)
        assert result["p_value"] < 0.01

    def test_no_signal_gives_high_p(self) -> None:
        baseline = [0.5, 0.5, 0.5, 0.5, 0.5]
        enhanced = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = paired_wilcoxon(baseline, enhanced)
        # With all ties, p_value may be NaN; accept that or p >= 0.5
        p = result["p_value"]
        assert p != p or p >= 0.1  # NaN check via self-comparison

    def test_drops_nan_pairs(self) -> None:
        baseline = [0.1, 0.2, float("nan"), 0.4, 0.5]
        enhanced = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = paired_wilcoxon(baseline, enhanced)
        assert result["n_pairs"] == 4

    def test_returns_none_for_insufficient_data(self) -> None:
        result = paired_wilcoxon([0.5], [0.6])
        assert result["p_value"] is None or result["p_value"] != result["p_value"]

    def test_shape_contains_all_fields(self) -> None:
        result = paired_wilcoxon([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])
        for key in ("statistic", "p_value", "n_pairs"):
            assert key in result


# ── Bootstrap CI ────────────────────────────────────────────


class TestBootstrapCI:
    def test_ci_brackets_point_estimate(self) -> None:
        import random
        random.seed(0)
        baseline = [random.uniform(0.2, 0.4) for _ in range(30)]
        # Add independent jitter so the per-pair delta has variance;
        # without it BCa collapses the interval onto the point estimate.
        enhanced = [b + 0.2 + random.gauss(0, 0.05) for b in baseline]
        low, high = bootstrap_paired_delta_ci(baseline, enhanced, seed=42)
        assert low < 0.2 < high

    def test_ci_ordering(self) -> None:
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5] * 3
        enhanced = [0.2, 0.3, 0.4, 0.5, 0.6] * 3
        low, high = bootstrap_paired_delta_ci(baseline, enhanced, seed=1)
        assert low <= high

    def test_returns_none_for_too_few_pairs(self) -> None:
        low, high = bootstrap_paired_delta_ci([0.5], [0.6], seed=1)
        assert low is None and high is None


# ── CRAG metrics ────────────────────────────────────────────


def _fixture_enhanced_results() -> dict:
    return {
        "q01": {  # ordinary success
            "answer": "Real answer", "pipeline_trace": ["retrieve", "generate"],
            "crag_rewrites": 0, "hallucination_retries": 0,
            "is_grounded": True, "metadata_filters_used": {"document_type": "MPR"},
            "chunks_retrieved": 10, "chunks_used": 5,
            "pre_rerank_ids": ["a", "b", "c"], "post_rerank_ids": ["c", "a", "b"],
        },
        "q02": {  # rewrite triggered + recovered
            "answer": "Answer after rewrite", "pipeline_trace": ["rewrite_query", "generate"],
            "crag_rewrites": 1, "hallucination_retries": 0,
            "is_grounded": True, "metadata_filters_used": None,
            "chunks_retrieved": 10, "chunks_used": 3,
            "pre_rerank_ids": ["a", "b"], "post_rerank_ids": ["a", "b"],  # top-1 unchanged
        },
        "q03": {  # hallucination retry
            "answer": "Re-generated", "pipeline_trace": ["generate", "generate_retry"],
            "crag_rewrites": 0, "hallucination_retries": 1,
            "is_grounded": True, "metadata_filters_used": None,
            "chunks_retrieved": 10, "chunks_used": 4,
            "pre_rerank_ids": [], "post_rerank_ids": [],
        },
        "q04": {  # abstain
            "answer": "This question does not appear to be answerable from the Bank of England document corpus.",
            "pipeline_trace": ["abstain"],
            "crag_rewrites": 1, "hallucination_retries": 0,
            "is_grounded": None, "metadata_filters_used": None,
            "chunks_retrieved": 0, "chunks_used": 0,
            "pre_rerank_ids": [], "post_rerank_ids": [],
        },
    }


class TestCRAGMetrics:
    def test_rewrite_trigger_rate(self) -> None:
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["rewrite_trigger_rate"] == pytest.approx(2 / 4)

    def test_hallucination_flag_rate_counts_final_ungrounded(self) -> None:
        """Flag rate = rows where final is_grounded is False (retry exhausted)."""
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        # Fixture has no final-ungrounded rows; q03 retried and recovered.
        assert m["hallucination_flag_rate"] == pytest.approx(0.0)

    def test_hallucination_recovery_rate(self) -> None:
        """Of queries that triggered a retry, fraction that ended grounded."""
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["hallucination_recovery_rate"] == pytest.approx(1.0)

    def test_hallucination_flag_rate_with_final_ungrounded(self) -> None:
        results = _fixture_enhanced_results()
        results["q05"] = {
            "answer": "Ungrounded answer after retry budget used",
            "pipeline_trace": ["generate", "generate_retry"],
            "crag_rewrites": 0, "hallucination_retries": 1,
            "is_grounded": False, "metadata_filters_used": None,
            "chunks_retrieved": 10, "chunks_used": 5,
            "pre_rerank_ids": [], "post_rerank_ids": [],
        }
        m = compute_crag_metrics(results, should_abstain_ids={"q04"})
        assert m["hallucination_flag_rate"] == pytest.approx(1 / 5)

    def test_metadata_filter_rate(self) -> None:
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["metadata_filter_rate"] == pytest.approx(1 / 4)

    def test_abstain_metrics(self) -> None:
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["abstain_rate"] == pytest.approx(1 / 4)
        # q04 is both "abstained" and "should have" → correct (precision-style)
        assert m["abstain_correctness"] == pytest.approx(1.0)

    def test_should_abstain_recall_all_captured(self) -> None:
        """Recall = |abstains ∩ should_abstain| / |should_abstain|.

        Independent of how many false-positive abstains also happen —
        that's the precision story (`abstain_correctness`).
        """
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["should_abstain_recall"] == pytest.approx(1.0)

    def test_should_abstain_recall_with_false_positives(self) -> None:
        """Precision drops when other queries abstain spuriously; recall doesn't."""
        results = _fixture_enhanced_results()
        # q01 also abstains (wrongly); q04 abstains (correctly).
        results["q01"]["answer"] = (
            "This question does not appear to be answerable from the "
            "Bank of England document corpus."
        )
        m = compute_crag_metrics(results, should_abstain_ids={"q04"})
        # 2 abstains, 1 correct → precision = 0.5
        assert m["abstain_correctness"] == pytest.approx(0.5)
        # recall still 1.0 (q04 was caught)
        assert m["should_abstain_recall"] == pytest.approx(1.0)

    def test_should_abstain_recall_missed(self) -> None:
        """q04 should abstain but doesn't → recall = 0."""
        results = _fixture_enhanced_results()
        results["q04"]["answer"] = "Real answer that shouldn't have been given."
        m = compute_crag_metrics(results, should_abstain_ids={"q04"})
        assert m["should_abstain_recall"] == pytest.approx(0.0)

    def test_should_abstain_recall_none_when_no_should_abstain(self) -> None:
        """Empty should_abstain set → recall undefined, returned as None."""
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids=set())
        assert m["should_abstain_recall"] is None

    def test_rerank_top1_change_rate(self) -> None:
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        # q01: c≠a (changed), q02: a=a (unchanged), q03/q04: empty
        # Count on rows that HAVE rerank data (q01, q02) → 1 of 2
        assert m["rerank_top1_change_rate"] == pytest.approx(0.5)

    def test_mean_chunks(self) -> None:
        m = compute_crag_metrics(_fixture_enhanced_results(),
                                 should_abstain_ids={"q04"})
        assert m["mean_chunks_retrieved"] == pytest.approx((10 + 10 + 10 + 0) / 4)
        assert m["mean_chunks_used"] == pytest.approx((5 + 3 + 4 + 0) / 4)


# ── Per-category means ─────────────────────────────────────


class TestPerCategoryMeans:
    def test_groups_scores_by_category(self) -> None:
        scores = {
            "q01": {"category": "A", "score": 0.5},
            "q02": {"category": "A", "score": 0.7},
            "q03": {"category": "B", "score": 0.9},
        }
        result = per_category_means(scores)
        assert result["A"]["mean"] == pytest.approx(0.6)
        assert result["A"]["n"] == 2
        assert result["B"]["mean"] == pytest.approx(0.9)
        assert result["B"]["n"] == 1

    def test_ignores_none_scores(self) -> None:
        scores = {
            "q01": {"category": "A", "score": 0.5},
            "q02": {"category": "A", "score": None},
        }
        result = per_category_means(scores)
        assert result["A"]["mean"] == pytest.approx(0.5)
        assert result["A"]["n"] == 1
