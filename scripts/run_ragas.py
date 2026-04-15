"""CLI: score pipeline outputs with RAGAS and aggregate into report tables.

Typical invocations:
    # Dry-run, 2 queries per pipeline — verify wiring before spending $20
    python scripts/run_ragas.py --sample 2 --pipeline both

    # Full run, resume from any existing JSONL checkpoint
    python scripts/run_ragas.py --pipeline both

    # Start fresh (wipes ragas_{baseline,enhanced}.jsonl)
    python scripts/run_ragas.py --pipeline both --no-resume

    # Use GPT-4o-mini as evaluator (escape hatch for tier-1 Anthropic TPM)
    python scripts/run_ragas.py --pipeline both --evaluator-provider openai \\
            --evaluator-model gpt-4o-mini

Outputs written to ``data/evaluation_results/``:
    ragas_{baseline,enhanced}.jsonl    per-sample scores (resumable)
    ragas_aggregate.json               headline means + Wilcoxon + CIs
    crag_metrics.json                  CRAG-specific metrics (enhanced only)
    comparison_table.csv               Table 2 of the report
    per_category.csv                   Per-category breakdown
"""

from __future__ import annotations

# Disable LangSmith tracing for RAGAS runs BEFORE any langchain/langsmith
# import fires. RAGAS fires ~800 evaluator LLM calls per full run; tracing
# those burns the free-tier 5k/month quota in a single run without any
# useful signal (the interesting behaviour is in the pipeline, not the
# judge). langsmith.utils.get_env_var is lru_cached, so it must be off
# BEFORE the first import resolves it.
import os
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse
import asyncio
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from boe_rag.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    Paths,
    setup_logging,
)
from boe_rag.evaluation import (
    SHOULD_ABSTAIN_IDS,
    bootstrap_paired_delta_ci,
    build_metrics,
    collect_run_metadata,
    compute_crag_metrics,
    holm_bonferroni,
    is_abstain,
    load_pipeline_results,
    load_test_set,
    paired_wilcoxon,
    per_category_means,
    results_to_samples,
    run_ragas,
)

logger = logging.getLogger(__name__)


METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision_with_reference",
    "context_recall",
]


def _build_evaluator(args) -> tuple[object, object, str]:
    """Wire the evaluator LLM + embeddings for collections metrics.

    Anthropic uses llm_factory + raw Anthropic client. OpenAI uses
    llm_factory + raw OpenAI client. Embeddings always use OpenAI
    (text-embedding-3-small) to match the indexing pipeline.
    """
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory

    # ascore() is async — pass ASYNC provider clients, not sync.
    # Using a sync client raises: "Cannot use agenerate() with a
    # synchronous client. Use generate() instead."
    if args.evaluator_provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(max_retries=5)
        llm = llm_factory(
            model=args.evaluator_model, provider="anthropic",
            client=client, temperature=0.0, max_tokens=2048,
        )
    elif args.evaluator_provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(max_retries=5)
        llm = llm_factory(
            model=args.evaluator_model, provider="openai",
            client=client, temperature=0.0, max_tokens=2048,
        )
    else:
        raise SystemExit(f"Unknown evaluator provider: {args.evaluator_provider}")

    # Embeddings use RAGAS's own wrapper; pass the async OpenAI client so
    # embedding calls don't block the event loop during concurrent scoring.
    from openai import AsyncOpenAI as _AsyncOpenAI
    embeddings = RagasOpenAIEmbeddings(
        client=_AsyncOpenAI(max_retries=5),
        model=EMBEDDING_MODEL,
    )
    return llm, embeddings, args.evaluator_model


def _load_jsonl_scores(path: Path) -> dict[tuple[str, str], float | None]:
    """Parse JSONL output → {(query_id, metric): score}. None for skipped/err."""
    if not path.exists():
        return {}
    scores: dict[tuple[str, str], float | None] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scores[(r["query_id"], r["metric"])] = r["score"]
    return scores


def _aggregate(
    baseline_scores: dict[tuple[str, str], float | None],
    enhanced_scores: dict[tuple[str, str], float | None],
    test_set: dict[str, dict],
    baseline_results: dict[str, dict],
    enhanced_results: dict[str, dict],
) -> dict:
    """Compute headline table: means (all + answered), Wilcoxon, Holm, BCa CI."""
    baseline_abstains = {qid for qid, r in baseline_results.items()
                         if is_abstain(r.get("answer", ""))}
    enhanced_abstains = {qid for qid, r in enhanced_results.items()
                         if is_abstain(r.get("answer", ""))}
    abstain_qids = baseline_abstains | enhanced_abstains
    qids = list(test_set.keys())

    per_metric: dict[str, dict] = {}
    raw_pvalues: list[float | None] = []
    metric_keys_with_p: list[str] = []

    for metric in METRIC_NAMES:
        b_all = [baseline_scores.get((qid, metric)) for qid in qids]
        e_all = [enhanced_scores.get((qid, metric)) for qid in qids]
        b_answered = [baseline_scores.get((qid, metric)) for qid in qids
                      if qid not in abstain_qids]
        e_answered = [enhanced_scores.get((qid, metric)) for qid in qids
                      if qid not in abstain_qids]

        wx = paired_wilcoxon(b_answered, e_answered)
        ci_low, ci_high = bootstrap_paired_delta_ci(b_answered, e_answered)

        entry = {
            "metric": metric,
            "n_scored_baseline": sum(1 for x in b_all if x is not None),
            "n_scored_enhanced": sum(1 for x in e_all if x is not None),
            "baseline_mean_all": _mean(b_all),
            "enhanced_mean_all": _mean(e_all),
            "baseline_mean_answered": _mean(b_answered),
            "enhanced_mean_answered": _mean(e_answered),
            "delta_answered": _delta(b_answered, e_answered),
            "wilcoxon_statistic": wx["statistic"],
            "p_raw": wx["p_value"],
            "n_pairs": wx["n_pairs"],
            "ci95_delta_low": ci_low,
            "ci95_delta_high": ci_high,
            "ci_method": "BCa_paired_10k",
        }
        per_metric[metric] = entry
        if wx["p_value"] is not None:
            raw_pvalues.append(wx["p_value"])
            metric_keys_with_p.append(metric)

    # Holm-Bonferroni across metrics with defined p-values.
    if raw_pvalues:
        adjusted = holm_bonferroni(raw_pvalues)
        for metric_key, p_adj in zip(metric_keys_with_p, adjusted, strict=True):
            per_metric[metric_key]["p_holm"] = p_adj

    return per_metric


def _mean(values):
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _delta(baseline, enhanced):
    pairs = [
        (b, e) for b, e in zip(baseline, enhanced)
        if b is not None and e is not None
    ]
    if not pairs:
        return None
    return sum(e - b for b, e in pairs) / len(pairs)


def _write_comparison_csv(per_metric: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "metric", "baseline_mean_all", "enhanced_mean_all",
            "baseline_mean_answered", "enhanced_mean_answered",
            "delta_answered", "p_raw", "p_holm",
            "ci95_low", "ci95_high", "n_pairs",
        ])
        for m in METRIC_NAMES:
            row = per_metric[m]
            w.writerow([
                m,
                _fmt(row["baseline_mean_all"]),
                _fmt(row["enhanced_mean_all"]),
                _fmt(row["baseline_mean_answered"]),
                _fmt(row["enhanced_mean_answered"]),
                _fmt(row["delta_answered"]),
                _fmt(row.get("p_raw")),
                _fmt(row.get("p_holm")),
                _fmt(row.get("ci95_delta_low")),
                _fmt(row.get("ci95_delta_high")),
                row.get("n_pairs"),
            ])


def _write_per_category_csv(
    path: Path,
    baseline_scores: dict[tuple[str, str], float | None],
    enhanced_scores: dict[tuple[str, str], float | None],
    test_set: dict[str, dict],
) -> None:
    """Per-category means for the two headline metrics.

    Narrative metrics: Faithfulness (headline grounding) + context_recall
    (headline retrieval). Other metrics are in the notebook.
    """
    metrics = ["faithfulness", "context_recall"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "n", "metric", "baseline_mean", "enhanced_mean", "delta"])
        for metric in metrics:
            b_rows = {
                qid: {"category": test_set[qid]["category"],
                      "score": baseline_scores.get((qid, metric))}
                for qid in test_set
            }
            e_rows = {
                qid: {"category": test_set[qid]["category"],
                      "score": enhanced_scores.get((qid, metric))}
                for qid in test_set
            }
            b_cat = per_category_means(b_rows)
            e_cat = per_category_means(e_rows)
            cats = sorted(set(b_cat) | set(e_cat))
            for cat in cats:
                b_mean = b_cat.get(cat, {}).get("mean")
                e_mean = e_cat.get(cat, {}).get("mean")
                n = max(b_cat.get(cat, {}).get("n", 0),
                        e_cat.get(cat, {}).get("n", 0))
                delta = (e_mean - b_mean) if b_mean is not None and e_mean is not None else None
                w.writerow([cat, n, metric, _fmt(b_mean), _fmt(e_mean), _fmt(delta)])


def _fmt(x):
    if x is None:
        return ""
    return f"{x:.4f}" if isinstance(x, float) else x


def main() -> int:
    load_dotenv()
    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["baseline", "enhanced", "both"],
                        default="both")
    parser.add_argument("--sample", type=int, default=None,
                        help="Dry-run with first N queries only")
    parser.add_argument(
        "--subset-ids",
        default=None,
        help="Comma-separated query IDs (e.g. q04,q07,q21) — scores only these. "
             "Takes precedence over --sample.",
    )
    parser.add_argument(
        "--out-suffix",
        default=None,
        help="Append to output filenames (e.g. 'sonnet_check' → "
             "ragas_baseline_sonnet_check.jsonl). Keeps headline results "
             "untouched when running a side experiment.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip writing ragas_aggregate.json / comparison_table.csv — "
             "useful with --subset-ids when you only want the per-sample JSONL.",
    )
    parser.add_argument("--no-resume", action="store_true",
                        help="Truncate existing JSONL and score fresh")
    parser.add_argument("--concurrency", type=int, default=4)
    # Default evaluator = GPT-4o-mini. Rationale:
    #   1) ~10x cheaper than Sonnet 4 ($0.15/$0.60 vs $3/$15 per MTok)
    #   2) 6x higher OpenAI tier-1 TPM (200k vs Anthropic 30k)
    #   3) Different model from the generator (Sonnet 4) — avoids
    #      same-model self-grading bias (RAGAS literature reports
    #      5-15% inflation when judge = generator).
    # Override with --evaluator-provider anthropic for a Sonnet rerun.
    parser.add_argument("--evaluator-provider", choices=["anthropic", "openai"],
                        default="openai")
    parser.add_argument("--evaluator-model", default="gpt-4o-mini")
    args = parser.parse_args()

    out_dir = Paths.DATA_EVAL
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    test_set = load_test_set(Paths.TEST_SET)
    if args.subset_ids:
        wanted = {qid.strip() for qid in args.subset_ids.split(",") if qid.strip()}
        missing = wanted - test_set.keys()
        if missing:
            raise SystemExit(f"Unknown query IDs in --subset-ids: {sorted(missing)}")
        test_set = {qid: row for qid, row in test_set.items() if qid in wanted}
        logger.info("Subset mode: scoring %d queries: %s",
                    len(test_set), sorted(test_set.keys()))
    elif args.sample is not None:
        limited = dict(list(test_set.items())[: args.sample])
        logger.info("Sample mode: scoring first %d queries only", len(limited))
        test_set = limited

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""

    baseline_results = load_pipeline_results(out_dir / "baseline_results.json")
    enhanced_results = load_pipeline_results(out_dir / "enhanced_results.json")

    # Evaluator wiring
    llm, embeddings, evaluator_model = _build_evaluator(args)
    metrics = build_metrics(llm=llm, embeddings=embeddings)

    # Score each requested pipeline
    to_run: list[tuple[str, dict, Path]] = []
    if args.pipeline in ("baseline", "both"):
        to_run.append(("baseline", baseline_results,
                       out_dir / f"ragas_baseline{suffix}.jsonl"))
    if args.pipeline in ("enhanced", "both"):
        to_run.append(("enhanced", enhanced_results,
                       out_dir / f"ragas_enhanced{suffix}.jsonl"))

    for name, results, jsonl_path in to_run:
        pairs = results_to_samples(results, test_set)
        qids = [qid for qid, _ in pairs]
        samples = [s for _, s in pairs]
        logger.info("Running RAGAS for %s: %d queries × %d metrics",
                    name, len(samples), len(metrics))
        asyncio.run(run_ragas(
            samples=samples, query_ids=qids,
            pipeline_name=name, metrics=metrics,
            out_path=jsonl_path,
            resume=not args.no_resume,
            concurrency=args.concurrency,
        ))

    if args.skip_aggregate or args.subset_ids:
        # Subset / side-experiment runs skip the aggregate writes so the
        # main ragas_aggregate.json / comparison_table.csv stay in sync
        # with the headline 25-query run.
        print("\nWrote per-sample JSONL only (aggregate skipped for subset/side run):")
        for _, _, p in to_run:
            if p.exists():
                print(f"  {p} ({p.stat().st_size} bytes)")
        return 0

    # Aggregate
    baseline_scores = _load_jsonl_scores(out_dir / "ragas_baseline.jsonl")
    enhanced_scores = _load_jsonl_scores(out_dir / "ragas_enhanced.jsonl")

    run_metadata = collect_run_metadata(
        test_set_path=Paths.TEST_SET,
        evaluator_model=evaluator_model,
    )
    aggregate = {
        "run_metadata": run_metadata,
        "per_metric": _aggregate(
            baseline_scores, enhanced_scores, test_set,
            baseline_results, enhanced_results,
        ),
        "abstains": {
            "baseline": [qid for qid, r in baseline_results.items() if is_abstain(r.get("answer", ""))],
            "enhanced": [qid for qid, r in enhanced_results.items() if is_abstain(r.get("answer", ""))],
            "should_abstain_ids": sorted(SHOULD_ABSTAIN_IDS),
        },
    }
    (out_dir / "ragas_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False)
    )

    crag = compute_crag_metrics(enhanced_results, should_abstain_ids=SHOULD_ABSTAIN_IDS)
    crag_out = {"run_metadata": run_metadata, "metrics": crag}
    (out_dir / "crag_metrics.json").write_text(
        json.dumps(crag_out, indent=2, ensure_ascii=False)
    )

    _write_comparison_csv(aggregate["per_metric"], out_dir / "comparison_table.csv")
    _write_per_category_csv(
        out_dir / "per_category.csv",
        baseline_scores, enhanced_scores, test_set,
    )

    print("\nWrote:")
    for p in (
        "ragas_baseline.jsonl", "ragas_enhanced.jsonl",
        "ragas_aggregate.json", "crag_metrics.json",
        "comparison_table.csv", "per_category.csv",
    ):
        path = out_dir / p
        print(f"  {path} ({path.stat().st_size} bytes)" if path.exists()
              else f"  {path} (MISSING)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
