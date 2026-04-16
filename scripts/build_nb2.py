"""Build notebooks/02_baseline_and_enhanced.ipynb (also exported as demo_log.pdf)."""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "02_baseline_and_enhanced.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ── Title ───────────────────────────────────────────────────
md("""# Notebook 2: Baseline vs Enhanced Pipeline (Demo Log)

**Objective**: comparison of the baseline RAG and the enhanced corrective RAG (CRAG) pipeline on six queries covering the major CRAG behaviours.

This notebook is the source for `demo_log.pdf` (spec 09).

**Format**: one smoke-test cell live-runs the enhanced pipeline (proves the code path is intact); the six examples below load locked answers from `data/evaluation_results/`, so the answers shown match exactly what RAGAS scored in Notebook 3.

**Behaviours covered**:

| # | qid | Behaviour |
|---|-----|-----------|
| 1 | q11 | Section-aware chunking + box-analysis metadata filter |
| 2 | q01 | Both pipelines answer correctly; rerank changes the top-1 chunk |
| 3 | q15 | CRAG corrective loop: rewrite fires on retrieval failure and recovers |
| 4 | q24 | Limitation: both pipelines struggle with a page-number citation |
| 5 | q21 | Out-of-corpus scope gate: enhanced abstains instead of hallucinating |
| 6 | q13 | Hallucination check fires; rerank changes the top-1 |""")

# ── Setup ──────────────────────────────────────────────────
md("""## Setup""")

code("""import json, os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RESULTS = ROOT / "data" / "evaluation_results"

baseline_results = json.loads((RESULTS / "baseline_results.json").read_text())
enhanced_results = json.loads((RESULTS / "enhanced_results.json").read_text())

print(f"Loaded results for {len(baseline_results)} baseline queries and "
      f"{len(enhanced_results)} enhanced queries.")

def _trim(s: str, n: int = 350) -> str:
    s = (s or "").strip().replace("\\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + " […]"

def show_query(qid: str, behaviour: str):
    b = baseline_results[qid]
    e = enhanced_results[qid]
    print("=" * 88)
    print(f"  {qid}  |  {behaviour}")
    print("=" * 88)
    print(f"\\nQuestion:  {b['question']}")
    print(f"Category:  {b['category']}\\n")

    print("--- BASELINE -------------------------------------------------")
    print(f"  trace:           {b['pipeline_trace']}")
    print(f"  chunks used:     {b['chunks_used']}")
    print(f"  is_grounded:     {b['is_grounded']}")
    print(f"  answer:\\n      {_trim(b['answer'], 600)}\\n")

    print("--- ENHANCED -------------------------------------------------")
    print(f"  trace:           {e['pipeline_trace']}")
    print(f"  chunks used:     {e['chunks_used']}")
    print(f"  rewrites:        {e['crag_rewrites']}")
    print(f"  halluc retries:  {e['hallucination_retries']}")
    print(f"  is_grounded:     {e['is_grounded']}")
    print(f"  filters used:    {e['metadata_filters_used']}")
    pre, post = e.get('pre_rerank_ids') or [], e.get('post_rerank_ids') or []
    if pre and post:
        flipped = "yes" if pre[0] != post[0] else "no"
        print(f"  rerank flipped top-1: {flipped}")
        print(f"    pre-rerank ids:  {pre[:5]}")
        print(f"    post-rerank ids: {post[:5]}")
    print(f"  answer:\\n      {_trim(e['answer'], 600)}")
""")

# ── Smoke test ─────────────────────────────────────────────
md("""## Smoke test (live execution)

One live call to prove the code path is intact. Uses the simplest factual query in the test set (q02). All examples below load locked outputs.""")

code("""# q02: simple factual, ~3 API calls (Anthropic + OpenAI + Cohere), <$0.05.
# Wrapped in try/except so transient API issues don't break the notebook.
try:
    from boe_rag.pipelines.enhanced import EnhancedPipeline
    pipe = EnhancedPipeline()
    smoke_question = enhanced_results["q02"]["question"]
    print(f"Question: {smoke_question}\\n")
    result = pipe.run(smoke_question)
    print(f"pipeline_trace: {result.pipeline_trace}")
    print(f"chunks_used:    {result.chunks_used}")
    print(f"is_grounded:    {result.is_grounded}\\n")
    print(f"Answer:\\n  {result.answer[:500]}")
except Exception as exc:
    print(f"[smoke test skipped] live API call failed: {type(exc).__name__}: {exc}")
    print("All showcase cells below use locked results from data/evaluation_results/")
""")

# ── 6 showcases ────────────────────────────────────────────
md("""## Example 1: Box D consumption weakness (q11)

The metadata filter (`section_category=box_analysis`, `date=2025-11`, `document_type=MPR`) retrieves the Box D chunk directly: one targeted chunk instead of five. The hallucination check fires and the retry tightens the answer to what that chunk strictly supports, flagging that Section 3.3 (referenced by Box D) falls outside the retrieval window. The baseline, with five unfiltered chunks, composes a broader multi-chunk answer. The two outputs illustrate different grounding philosophies (strict single-chunk grounding vs permissive multi-chunk synthesis), not a direct win-loss on this query.""")
code("""show_query("q11", "Section-aware chunking + box-analysis metadata filter")""")

md("""## Example 2: MPC vote split February 2026 (q01)

Both pipelines answer correctly (vote splits are simple factual content appearing in many chunks). For the enhanced pipeline, the relevant detail is the rerank: pre-rerank top-1 is one chunk, post-rerank top-1 is a different one. Cohere's relevance model surfaces a more relevant chunk than vector cosine similarity alone.""")
code("""show_query("q01", "Both pipelines succeed; rerank flips top-1")""")

md("""## Example 3: Lombardelli asymmetric policy risk (q15)

The enhanced pipeline's first retrieval pass returned nothing the grader judged relevant. The CRAG corrective loop fired: `analyze_query → retrieve → grade_documents → rewrite_query → retrieve → grade_documents → rerank → generate`. The rewritten query recovered relevant chunks and produced a grounded answer.""")
code("""show_query("q15", "CRAG corrective loop: rewrite fires and recovers")""")

md("""## Example 4: GDP Q3 2025, page 23 (q24)

The question asks for an exact figure from a specific page of a specific document. Both pipelines struggle because (a) embeddings don't preserve numerical precision and (b) chunks don't preserve page numbers. The enhanced pipeline ultimately abstains (`abstain` in the trace) rather than fabricate a number, which is the correct behaviour even though the assignment's ground truth says this question is in-corpus. This is one of three false-positive abstains contributing to the 0.25 abstain-precision reported in Notebook 3.""")
code("""show_query("q24", "Both pipelines struggle; enhanced abstains rather than fabricate")""")

md("""## Example 5: Federal Reserve view on rates (q21)

The question is about the Fed, not the Bank of England. The scope-detection layer in `analyze_query` flags `out_of_corpus=True`, and the router short-circuits to `abstain_out_of_corpus`: no retrieval, no generation, no API spend on a question the corpus cannot answer. Trace: `analyze_query → abstain_out_of_corpus`. This is the should-abstain-recall = 1.00 behaviour.""")
code("""show_query("q21", "Out-of-corpus scope gate: enhanced abstains")""")

md("""## Example 6: US corporate default risks (q13)

This query triggers two safety mechanisms in sequence: Cohere reranking changes the top-1 chunk, and the hallucination check fires after the first generation (trace: `generate → check_hallucination → generate → check_hallucination`). The retry produced a grounded final answer.""")
code("""show_query("q13", "Hallucination check fires + rerank flips top-1")""")

# ── Summary table ─────────────────────────────────────────
md("""## Cross-query summary""")

code("""rows = []
for qid in ["q11", "q01", "q15", "q24", "q21", "q13"]:
    e = enhanced_results[qid]
    pre, post = e.get('pre_rerank_ids') or [], e.get('post_rerank_ids') or []
    rerank_flipped = (bool(pre) and bool(post) and pre[0] != post[0])
    rows.append({
        "qid": qid,
        "category": e["category"],
        "trace_steps": len(e["pipeline_trace"]),
        "rewrite_fired": "rewrite_query" in e["pipeline_trace"],
        "rerank_flipped": rerank_flipped,
        "halluc_retry": e["hallucination_retries"] > 0,
        "abstained": "abstain" in e["pipeline_trace"] or "abstain_out_of_corpus" in e["pipeline_trace"],
        "is_grounded": e["is_grounded"],
        "chunks_used": e["chunks_used"],
    })
display(pd.DataFrame(rows).set_index("qid"))
""")

# ── Synthesis ─────────────────────────────────────────────
md("""## What these six queries collectively demonstrate

Across the six representative queries, every advanced technique from Section 2 of the report fires at least once:

- **Metadata filter** applied on 5 of 6 (every query except q21, which short-circuits before retrieval)
- **Rerank flipped top-1** on 3 of 6 (q01, q15, q13)
- **CRAG rewrite** triggered on 2 of 6 (q15 recovered, q24 did not)
- **Hallucination retry** fired on 2 of 6 (q11, q13 both produced grounded final answers)
- **Out-of-corpus scope gate** fired on 1 of 6 (q21) with zero API spend beyond `analyze_query`
- **Post-retrieval abstain** fired on 1 of 6 (q24, a false-positive on in-corpus content)

The abstain-related behaviours (q21 correct, q24 false-positive) are the decisive architectural difference from the baseline: baseline attempts every query and therefore cannot demonstrate scope awareness. Notebook 3 quantifies whether this behavioural delta translates into RAGAS metric gains at the 25-query scale.""")

# ── Closing ───────────────────────────────────────────────
md("""## What's next

Notebook 3 quantifies these behaviours across the full 25-query test set with RAGAS metrics, paired Wilcoxon and Holm-Bonferroni statistical tests, and per-category analysis.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}

OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")
