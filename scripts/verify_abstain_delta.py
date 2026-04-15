"""B1 regression gate: verify post-fix abstain set equals pre-fix ∪ {q21}.

Loads two enhanced_results.json files (pre-fix from a git revision, post-fix
from working tree) and asserts the only delta in the abstain set is the
addition of q21. Per-query equality isn't the right contract because
Claude @ temp=0 isn't bit-deterministic — we care about the ABSTAIN SET,
not every field.

Usage:
    # Compare working tree against pre-b1 tag:
    git show pre-b1-abstain-fix:data/evaluation_results/enhanced_results.json > /tmp/pre.json
    python scripts/verify_abstain_delta.py \\
        --pre /tmp/pre.json \\
        --post data/evaluation_results/enhanced_results.json

Exits 0 on invariant holds, 1 otherwise. Logs soft warnings for
non-q21 trace-structure changes but doesn't block on them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boe_rag.evaluation.adapters import is_abstain


def _abstains(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    return {qid for qid, r in data.items() if is_abstain(r.get("answer", ""))}


def _trace_length(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    return {qid: len(r.get("pipeline_trace", [])) for qid, r in data.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", type=Path, required=True,
                    help="Pre-fix enhanced_results.json (from git)")
    ap.add_argument("--post", type=Path, required=True,
                    help="Post-fix enhanced_results.json (working tree)")
    args = ap.parse_args()

    pre = _abstains(args.pre)
    post = _abstains(args.post)

    added = post - pre
    removed = pre - post

    print(f"Pre-fix abstains   ({len(pre)}): {sorted(pre)}")
    print(f"Post-fix abstains  ({len(post)}): {sorted(post)}")
    print(f"Added:   {sorted(added)}")
    print(f"Removed: {sorted(removed)}")
    print()

    invariant_pass = True
    if added != {"q21"}:
        print(f"FAIL: added abstains must be exactly {{'q21'}}, got {sorted(added)}")
        invariant_pass = False
    if removed:
        print(f"FAIL: no prior abstains should disappear. Lost: {sorted(removed)}")
        invariant_pass = False

    # Soft warning on trace-length changes outside q21
    pre_lens = _trace_length(args.pre)
    post_lens = _trace_length(args.post)
    shifts = []
    for qid in sorted(set(pre_lens) & set(post_lens)):
        if qid == "q21":
            continue
        if pre_lens[qid] != post_lens[qid]:
            shifts.append((qid, pre_lens[qid], post_lens[qid]))
    if shifts:
        print(f"Soft warning: trace-length changed on {len(shifts)} non-q21 queries:")
        for qid, a, b in shifts:
            print(f"  {qid}: {a} -> {b}")
        print("(Claude non-determinism at temp=0 can cause this; review manually)")
    else:
        print("All non-q21 trace lengths identical.")

    if invariant_pass:
        print()
        print("PASS — abstain set delta is exactly {q21}.")
        return 0
    print()
    print("FAIL — regression check failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
