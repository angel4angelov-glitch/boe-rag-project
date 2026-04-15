"""B1 pre-flight: run analyze_query's scope check on every test-set query.

Validates that Claude correctly classifies every question as in-corpus or
out-of-corpus BEFORE the full re-eval spend. Uses the same
``_structured_llm`` that EnhancedPipeline constructs in production —
NOT a re-built copy. Any drift between test and production would invalidate
the pre-flight.

Output: data/evaluation_results/pre_flight_scope_check.json with raw
QueryFilters + pass/fail per query. Exits non-zero on any misclassification
so a CI / shell script can gate the next step (full re-eval).

Cost: 25 × ~$0.008 = ~$0.20 on Sonnet 4 via the production analyze_query
wiring.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from boe_rag.config import Paths, setup_logging
from boe_rag.evaluation import load_test_set
from boe_rag.pipelines.nodes import QueryFilters
from boe_rag.pipelines.prompts import ANALYZE_QUERY_PROMPT

logger = logging.getLogger(__name__)


# Ground-truth mapping. Only q21 is expected to be out-of-corpus (the
# Federal Reserve question). Every other category-in-corpus query must
# classify as False. Update if the test set grows new out-of-scope rows.
EXPECTED: dict[str, bool] = {
    f"q{i:02d}": (i == 21) for i in range(1, 26)
}


def main() -> int:
    load_dotenv()
    setup_logging(logging.WARNING)

    test_set = load_test_set(Paths.TEST_SET)
    if len(test_set) != 25:
        logger.error("Expected 25 queries; got %d. Update EXPECTED mapping.", len(test_set))
        return 2

    # Use the SAME structured LLM construction as EnhancedPipeline's
    # __init__ — identical model, temperature, retry wrapper,
    # .with_structured_output(QueryFilters) ordering. Any divergence
    # invalidates the pre-flight.
    from boe_rag.pipelines.enhanced import _with_retries
    from langchain_anthropic import ChatAnthropic
    from boe_rag.config import GRADING_MODEL, LLM_TEMPERATURE
    structured_llm = _with_retries(
        ChatAnthropic(
            model=GRADING_MODEL, temperature=LLM_TEMPERATURE
        ).with_structured_output(QueryFilters)
    )

    results: list[dict] = []
    mismatches: list[dict] = []

    print(f"Pre-flight: {len(test_set)} queries through analyze_query's "
          f"scope check ({GRADING_MODEL}, temperature={LLM_TEMPERATURE})")
    print("-" * 80)

    for qid, row in test_set.items():
        question = row["question"]
        expected = EXPECTED.get(qid, False)
        try:
            filters: QueryFilters = structured_llm.invoke(
                ANALYZE_QUERY_PROMPT.format(question=question)
            )
        except Exception as e:
            logger.error("LLM raised on %s: %s", qid, e)
            results.append({
                "query_id": qid, "question": question,
                "expected_out_of_corpus": expected,
                "actual_out_of_corpus": None,
                "raw_filters": None,
                "error": f"{type(e).__name__}: {e}",
                "match": False,
            })
            mismatches.append({"qid": qid, "expected": expected, "actual": None})
            continue

        actual = bool(filters.out_of_corpus)
        match = actual == expected
        marker = "✓" if match else "✗"
        print(f"  {marker} {qid} expected={expected!s:<5} actual={actual!s:<5}  {question[:60]}")
        results.append({
            "query_id": qid, "question": question,
            "expected_out_of_corpus": expected,
            "actual_out_of_corpus": actual,
            "raw_filters": filters.model_dump(),
            "error": None,
            "match": match,
        })
        if not match:
            mismatches.append({"qid": qid, "expected": expected, "actual": actual})

    out_path = Paths.DATA_EVAL / "pre_flight_scope_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": GRADING_MODEL,
            "temperature": LLM_TEMPERATURE,
            "n_queries": len(test_set),
            "n_mismatches": len(mismatches),
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    print("-" * 80)
    print(f"Wrote: {out_path}")
    print(f"Mismatches: {len(mismatches)} / {len(test_set)}")

    if mismatches:
        print()
        print("FAIL — pre-flight did not pass. Tune the prompt and retry.")
        for m in mismatches:
            print(f"  {m['qid']}: expected={m['expected']} actual={m['actual']}")
        return 1
    print()
    print("PASS — all 25 queries classified correctly. Safe to re-eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
