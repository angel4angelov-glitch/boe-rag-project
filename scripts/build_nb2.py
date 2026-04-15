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
md("""# Notebook 2 — Baseline vs Enhanced Pipeline (Demo Log)

**Objective**: side-by-side comparison of the naive baseline RAG and the enhanced corrective RAG (CRAG) pipeline on six representative queries that together exercise every behaviour the enhanced pipeline implements.

This notebook is the source for `demo_log.pdf` (spec 09 deliverable).

**Format**: one **smoke test** cell that live-runs the enhanced pipeline (proves the code works end-to-end), then six **showcase** queries that load locked answers from `data/evaluation_results/` so the answers shown here are exactly the answers RAGAS scored in NB3. This guarantees the demo log and the report's metric tables refer to the same answers.

**Behaviours showcased**:

| # | qid | Behaviour |
|---|-----|-----------|
| 1 | q11 | Section-aware chunking + box-analysis metadata filter (enhanced wins) |
| 2 | q01 | Both pipelines succeed; rerank changes top-1 chunk ordering |
| 3 | q15 | CRAG corrective loop: rewrite fires on retrieval failure and recovers |
| 4 | q24 | Honest limitation: both pipelines struggle with a page-number citation |
| 5 | q21 | Out-of-corpus scope gate: enhanced abstains rather than hallucinate |
| 6 | q13 | Hallucination check fires + rerank changes top-1 |""")

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
md("""## Smoke test — live execution of the enhanced pipeline

One live call to prove the code path is intact end-to-end. Uses the simplest factual query in the test set (q02). All other showcase cells below load locked outputs.""")

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
md("""## Showcase 1 — Box D consumption weakness (q11)

**What to notice**: the enhanced pipeline filters retrieval to `section_category=box_analysis`, pulling Box D as a coherent unit. The baseline scatters across unrelated MPR paragraphs and produces a longer but less precise answer. The enhanced pipeline's hallucination check also fires here (trace shows `generate → check_hallucination → generate`) — the retry tightened the answer.""")
code("""show_query("q11", "Section-aware chunking + box-analysis metadata filter")""")

md("""## Showcase 2 — MPC vote split February 2026 (q01)

**What to notice**: both pipelines succeed (vote splits are simple factual content that lives in many chunks). The interesting bit for the enhanced pipeline is the rerank: pre-rerank top-1 is one chunk, post-rerank top-1 is a different one. Cohere's relevance model surfaces a more directly-on-question chunk than vector cosine similarity alone.""")
code("""show_query("q01", "Both pipelines succeed; rerank flips top-1")""")

md("""## Showcase 3 — Lombardelli asymmetric policy risk (q15)

**What to notice**: the enhanced pipeline's first retrieval pass found nothing the grader judged relevant. The CRAG corrective loop fired: `analyze_query → retrieve → grade_documents → rewrite_query → retrieve → grade_documents → rerank → generate`. The rewritten query recovered relevant chunks and produced a grounded answer.""")
code("""show_query("q15", "CRAG corrective loop: rewrite fires and recovers")""")

md("""## Showcase 4 — GDP Q3 2025, page 23 (q24) — honest limitation

**What to notice**: the question asks for an exact figure from a specific page of a specific document. Both pipelines struggle because (a) embeddings don't preserve numerical precision and (b) chunks don't preserve page numbers. The enhanced pipeline ultimately abstains (`abstain` in the trace) rather than fabricate a number — which is the right behaviour, even though the assignment ground truth says this question is in-corpus. This is one of three false-positive abstains contributing to the 0.25 abstain-precision noted in NB3.""")
code("""show_query("q24", "Both pipelines struggle; enhanced abstains rather than fabricate")""")

md("""## Showcase 5 — Federal Reserve view on rates (q21) — out-of-corpus

**What to notice**: the question is about the Fed, not the Bank of England. The B1 scope-detection extension flags `out_of_corpus=True` in `analyze_query`, and the router short-circuits straight to `abstain_out_of_corpus` — no retrieval, no generation, no API spend on a question the corpus cannot answer. Trace: `analyze_query → abstain_out_of_corpus`. This is the should-abstain-recall=1.00 behaviour.""")
code("""show_query("q21", "Out-of-corpus scope gate: enhanced abstains")""")

md("""## Showcase 6 — US corporate default risks (q13)

**What to notice**: this query exercises two safety mechanisms in sequence. Cohere reranking changes the top-1 chunk, and the hallucination check fires after the first generation (trace shows `generate → check_hallucination → generate → check_hallucination`). The retry produced a grounded final answer.""")
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

# ── Closing ───────────────────────────────────────────────
md("""## What's next

Notebook 3 quantifies these behaviours across the full 25-query test set with RAGAS metrics, paired Wilcoxon + Holm-Bonferroni statistical tests, and per-category analysis.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}

OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")
