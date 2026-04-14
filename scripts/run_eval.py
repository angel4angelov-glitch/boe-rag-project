"""Run baseline + enhanced pipelines on every question in test_set.csv.

Writes per-query JSON results to data/evaluation_results/. Used by
Notebook 03 and spec 07's RAGAS scoring. Safe to re-run — overwrites the
two result files atomically.
"""

from __future__ import annotations

import json
import logging
import sys
import time

import pandas as pd

from boe_rag.config import Paths, setup_logging
from boe_rag.pipelines import BaselinePipeline, EnhancedPipeline


def _result_to_dict(result) -> dict:
    return {
        "answer": result.answer,
        "pipeline_name": result.pipeline_name,
        "chunks_retrieved": result.chunks_retrieved,
        "chunks_used": result.chunks_used,
        "crag_rewrites": result.crag_rewrites,
        "hallucination_retries": result.hallucination_retries,
        "is_grounded": result.is_grounded,
        "metadata_filters_used": result.metadata_filters_used,
        "pipeline_trace": list(result.pipeline_trace),
        "sources": [
            {"chunk_id": d.chunk_id, "score": d.score, "text_preview": d.text[:200]}
            for d in result.sources
        ],
    }


def main() -> int:
    setup_logging(logging.WARNING)
    df = pd.read_csv(Paths.TEST_SET)
    print(f"Running {len(df)} queries through both pipelines", flush=True)

    baseline = BaselinePipeline()
    enhanced = EnhancedPipeline()

    baseline_results: dict = {}
    enhanced_results: dict = {}
    t_start = time.time()

    for i, row in df.iterrows():
        q = row["question"]
        cat = row["category"]
        qid = f"q{i + 1:02d}"
        print(f"{qid} [{cat[:20]:<20s}] {q[:60]}", flush=True)

        t0 = time.time()
        br = baseline.run(q)
        tb = time.time() - t0
        t0 = time.time()
        er = enhanced.run(q)
        te = time.time() - t0
        errored = er.answer.startswith("[Pipeline error")
        print(
            f"   baseline={tb:.1f}s  enhanced={te:.1f}s  "
            f"trace={len(er.pipeline_trace)}  grounded={er.is_grounded}  errored={errored}",
            flush=True,
        )
        baseline_results[qid] = {"question": q, "category": cat, **_result_to_dict(br)}
        enhanced_results[qid] = {"question": q, "category": cat, **_result_to_dict(er)}

    out_dir = Paths.DATA_EVAL
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_results.json").write_text(
        json.dumps(baseline_results, indent=2)
    )
    (out_dir / "enhanced_results.json").write_text(
        json.dumps(enhanced_results, indent=2)
    )

    elapsed = time.time() - t_start
    n_errors = sum(
        1 for r in enhanced_results.values() if r["answer"].startswith("[Pipeline error")
    )
    n_abstain = sum(
        1 for r in enhanced_results.values() if "abstain" in r["pipeline_trace"]
    )
    n_rewrites = sum(r["crag_rewrites"] for r in enhanced_results.values())
    n_retries = sum(r["hallucination_retries"] for r in enhanced_results.values())

    print(f"\n=== Done in {elapsed:.0f}s ===", flush=True)
    print(f"  errors:                {n_errors}/{len(df)}", flush=True)
    print(f"  abstains:              {n_abstain}/{len(df)}", flush=True)
    print(f"  CRAG rewrites:         {n_rewrites}", flush=True)
    print(f"  hallucination retries: {n_retries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
