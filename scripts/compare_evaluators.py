"""Compare two evaluator runs on the same (query, metric) tuples.

Typical use: headline run scored by gpt-4o-mini; side run scored by
Sonnet 4 on a 5-query subset. This script tells you whether the two
judges agree.

Usage:
    python scripts/compare_evaluators.py \\
        --a data/evaluation_results/ragas_baseline.jsonl \\
        --b data/evaluation_results/ragas_baseline_sonnet_check.jsonl \\
        --out data/evaluation_results/evaluator_divergence_baseline.csv

Output CSV columns: query_id, metric, score_a, score_b, abs_diff.
Prints max/mean abs diff so the caller can decide "consistent enough
to quote gpt-4o-mini numbers in the report" vs "material disagreement,
needs a paragraph of explanation".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_scores(path: Path) -> dict[tuple[str, str], float | None]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    out: dict[tuple[str, str], float | None] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["query_id"], r["metric"])] = r["score"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True,
                    help="First JSONL (headline, e.g. gpt-4o-mini)")
    ap.add_argument("--b", type=Path, required=True,
                    help="Second JSONL (side run, e.g. Sonnet)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()

    a = _load_scores(args.a)
    b = _load_scores(args.b)

    # Only keys present in BOTH can be compared.
    common = sorted(a.keys() & b.keys())
    if not common:
        print("No overlapping (query, metric) keys between inputs.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    diffs: list[float] = []
    for qid, metric in common:
        sa = a[(qid, metric)]
        sb = b[(qid, metric)]
        if sa is None or sb is None:
            # skip rows either side skipped (e.g. abstains)
            continue
        diff = abs(sa - sb)
        diffs.append(diff)
        rows.append({
            "query_id": qid,
            "metric": metric,
            f"score_{args.label_a}": round(sa, 4),
            f"score_{args.label_b}": round(sb, 4),
            "abs_diff": round(diff, 4),
        })

    if not rows:
        print("No comparable rows after dropping null scores.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["abs_diff"], reverse=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    median_diff = sorted(diffs)[len(diffs) // 2]

    print(f"Compared N={len(rows)} (query, metric) pairs between {args.label_a} and {args.label_b}")
    print(f"  max abs diff:    {max_diff:.3f}")
    print(f"  mean abs diff:   {mean_diff:.3f}")
    print(f"  median abs diff: {median_diff:.3f}")
    print()
    top = min(5, len(rows))
    print(f"Top {top} disagreements:")
    for r in rows[:top]:
        print(f"  {r['query_id']:<5} {r['metric']:<34} "
              f"{args.label_a}={r[f'score_{args.label_a}']:.3f}  "
              f"{args.label_b}={r[f'score_{args.label_b}']:.3f}  "
              f"|diff|={r['abs_diff']:.3f}")

    # Thresholds — report-relevant guidance
    if max_diff < 0.15:
        verdict = "CONSISTENT — evaluators agree within noise. Quote headline numbers."
    elif max_diff < 0.30:
        verdict = "MIXED — some disagreement. Worth a sentence in Methodology."
    else:
        verdict = "MATERIAL — divergence is substantive. Report both, explain."
    print()
    print(f"Verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
