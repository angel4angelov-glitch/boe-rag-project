# Notebook Population Plan — Submission Tier 1, Step 1 (v2 — final)

**Status**: v2 draft, awaiting approval before any cell runs
**Deadline**: 2026-04-16 12:00 UK (~24h out at time of writing)
**v1 → v2 changes** (read this first):
1. **No live 5-query pipeline run in NB2.** Instead: load the locked results from `enhanced_results.json` / `baseline_results.json` for the showcase + one cheap live "smoke test" on q02 to prove the code works. Reason: non-determinism. The answer from a live re-run will not exactly match the answer evaluated in the locked RAGAS run that the report cites. Marker comparing NB2 to the report would see the discrepancy. **Integrity > optics.**
2. **No ChromaDB delete + rebuild in NB1.** Reason: embeddings are version-sensitive (OpenAI model drift, tokeniser changes). Rebuilding could subtly shift retrieval. Current DB state produced the locked eval; don't clobber it. Instead: NB1 reads counts + samples + runs a sanity query, with a clearly-marked "how-to-rebuild-from-scratch" block that is **not executed** by default.
3. **NB2 == demo_log.pdf.** Spec 09 explicitly allows the demo log to live inside a notebook (line 10). Exporting NB2 to PDF produces `demo_log.pdf`. We are NOT building two separate artefacts. This collapses one todo.
4. **Concrete demo query IDs committed** (not "to be chosen at execution time").
5. **matplotlib added to dev deps** — needed for one NB3 chart.

---

## Ram's framing (locked)

1. Markers open `.ipynb` statically and scroll. Outputs must be saved; render speed doesn't matter.
2. Anything you re-run on deadline day is a new liability against a locked, reproducible artefact. Default to read-only unless there's a reason to re-execute.
3. Option A was partially right: live re-exec for deterministic/cheap stages, load-from-JSON for expensive/non-deterministic ones.

---

## Scope decisions (locked)

| Decision | Chosen | Why |
|---|---|---|
| Re-scrape from live URLs | **No** | `data/html_cache/` has every HTML. Live fetch = flaky. |
| Re-chunk | **No (read from `data/chunks/*.json`)** | Already on disk, deterministic. A re-chunk cell commented out, available for demonstration. |
| Re-embed + re-index ChromaDB | **No** | Preserves integrity with locked eval. Demonstrates-only cell at end of NB1, not executed. |
| Live pipeline runs in NB2 | **One smoke test only (q02)** | Proves code works end-to-end. All 6 showcases load from JSON for answer-integrity with the report. |
| Re-run RAGAS in NB3 | **No — load existing JSON** | $10 + 30 min + flakiness on deadline day; also would invalidate the `test_set_hash` cited in reports. |
| Collapse NB2 + demo_log.pdf | **Yes** | Spec 09 allows notebook format. Export NB2 to PDF = demo log. |

---

## Demo query slate (6 queries, concrete IDs, mapped to spec 09)

| Slot | QID | Question (truncated) | CRAG behaviour shown | Spec 09 Ex # |
|---|---|---|---|---|
| 1 | **q11** | Box D consumption weakness scenario (MPR Nov 2025) | Section-aware chunking + `box_analysis` metadata filter | Ex 1 |
| 2 | **q01** | MPC vote split Feb 2026 | Both pipelines succeed; **rerank flipped top-1** | Ex 2 + Ex 6 |
| 3 | **q15** | Lombardelli asymmetric policy risk (Nov 2025) | **Rewrite fired → recovered grounded answer** | Ex 3 |
| 4 | **q24** | GDP Q3 2025 page-23 citation | Both pipelines struggle; honest limitation | Ex 4 |
| 5 | **q21** | "What is the Fed's view on interest rates?" | **Out-of-corpus abstain** (scope gate) | Ex 5 |
| 6 | **q13** | US corporate default risks (Dec 2025 FSR) | **Hallucination retry + rerank flip** (bonus) | Ex 7 extra |

These IDs are confirmed by inspecting `enhanced_results.json`:
- `rewrite_query` in trace for q03, q06, q10, q15, q24 — **q15 is the clean rewrite-recovery case** (grounded=True, not abstained)
- Top-1 swapped after rerank for q01, q04, q05, q08, q13, q15, q18, q19, q20, q22, q23, q25
- Abstained: q06, q10, q21, q24 — **q21 is the correct one** (spec 09 Ex 5)
- Hallucination retry trace: q09, q11, q13, q20, q25 — **q13 also had rerank flip**

Swap rule if execution reveals one of these isn't as expected: use `q03` for rewrite instead of q15, and `q10` for false-positive abstain bonus.

---

## Notebook 1 — Data Ingestion & Indexing

**Target**: ~10 cells, ~1 min execution, ~$0 cost.

### Cell-by-cell spec

1. **[md]** Title + objective: "End-to-end data pipeline: scrape → chunk → embed → index. This notebook inspects and validates the committed state; it does NOT rebuild embeddings (see final cell for how-to)."

2. **[md]** Reproducibility banner placeholder — git SHA, Python version, key package versions. Populated by the next code cell.

3. **[code]** Imports + reproducibility banner: use `importlib.metadata.version("anthropic")`, etc., and `subprocess.check_output(["git","rev-parse","--short","HEAD"])`. Print as a nice block.

4. **[code]** `load_dotenv()`. Assert keys exist (OPENAI for sanity query; don't require ANTHROPIC here). Use masked print — do not echo key contents.

5. **[md]** "Scraped corpus"

6. **[code]** Load `data/raw/manifest.csv` → pandas, `display(df.head())`, `df.groupby('document_type').size()`, total word count. Expected: 23 rows, 4 doc types.

7. **[md]** "Chunking"

8. **[code]** Load one baseline chunk file + one enhanced chunk file (e.g. `mpc_2025_06.json` from both). Display one sample `Chunk` from each side-by-side. Print total chunk counts from both collections.

9. **[md]** "Indexing"

10. **[code]** `chromadb.PersistentClient("chroma_db")` → get both collections by name → print `.count()` for each. Run ONE sanity query: `boe_enhanced.query(query_texts=["What did MPC decide in November 2025?"], n_results=3)`. Display returned chunk ids + metadata + (1 - distance) scores.

11. **[md]** "How to rebuild from scratch (demonstration only, not executed)"

12. **[code]** `if False:` guarded block containing the full scrape → chunk → index code. Comment at top: "Set to True to rebuild. Will overwrite committed state — don't do this unless you know why."

13. **[md]** "Validation checklist" — explicit ticked boxes that the prior cells have demonstrated (counts match, metadata populated, sanity query returned relevant chunks).

### Expected marker-visible outputs
- Reproducibility banner (git SHA, versions, timestamp)
- Manifest head + grouping showing 23 docs across 4 types
- Side-by-side baseline vs enhanced chunk sample (the metadata difference is visible)
- Collection counts + sanity query results with scores

---

## Notebook 2 — Pipeline Demo (=== `demo_log.pdf` source)

**Target**: ~20 cells, ~30 s execution, ~$0.05 cost (one smoke test call).

### Cell-by-cell spec

1. **[md]** Title + objective: "Side-by-side demonstration of BaselinePipeline and EnhancedPipeline on 6 representative queries. This notebook is the source for `demo_log.pdf`."

2. **[md]** How to read: legend for pipeline_trace tokens, note on the one live smoke-test cell.

3. **[code]** Imports + `load_dotenv()`. Define a `load_result(qid, pipeline)` helper that pulls from `baseline_results.json` / `enhanced_results.json`.

4. **[md]** "Smoke test (live)"

5. **[code]** `EnhancedPipeline().run("What was Brent crude price cited in the March 2026 MPC minutes?")` — this is q02 (simple, short, cheap; ~3 API calls). Display the `PipelineResult.answer`, `pipeline_trace`, `chunks_used`. Wrap in try/except; print clear error + skip if this fails.

6. **[md]** "Query 1 — Box D consumption (section-aware chunking wins)"

7. **[code]** Load q11 from both result files. Display: question, baseline answer (with chunks), enhanced answer (with chunks + trace + sources). Include a 2-line "what to notice" markdown after.

8-17. Repeat (pairs of md + code cells) for q01, q15, q24, q21, q13.

18. **[md]** "Cross-query summary"

19. **[code]** Build a pandas DataFrame: 6 rows × [qid, question, category, baseline_len, enhanced_len, trace, is_grounded, chunks_used, rewrite_fired, rerank_flipped]. Display.

20. **[md]** "What's next" — pointer to NB3.

### Export to PDF
After executing + saving: `jupyter nbconvert --to pdf notebooks/02_baseline_and_enhanced.ipynb --output-dir . --output demo_log.pdf`. Requires LaTeX. If LaTeX not available, fallback: `--to html` then print-to-PDF from browser.

### Expected marker-visible outputs
- One live pipeline call with trace showing `analyze_query → retrieve → grade_documents → rerank → generate → check_hallucination`
- 6 side-by-side answer comparisons from locked results
- At least one visible `rewrite_query` trace (q15), one `abstain_out_of_corpus` trace (q21), one rerank top-1 swap (q01 or q13)
- Summary DataFrame

---

## Notebook 3 — Evaluation (load-only)

**Target**: ~12 cells, ~20 s execution, $0 cost.

### Cell-by-cell spec

1. **[md]** Title + objective + `test_set_hash` reproducibility statement.

2. **[code]** Load all 6 evaluation files into dict. Print file sizes + mtimes.

3. **[md]** "Headline RAGAS metrics"

4. **[code]** Build the headline DataFrame: rows = {Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall}, cols = {baseline_mean, enhanced_mean, Δ, p_raw, p_holm, CI95}. From the current `ragas_aggregate.json`:
   - Faithfulness: 0.959 → 0.900, Δ=-0.057, p_holm=1.0, CI [-0.16, +0.01]
   - AnswerRelevancy: 0.582 → 0.529, Δ=-0.053, p_holm=1.0, CI [-0.25, +0.16]
   - ContextPrecision: 0.658 → 0.806, Δ=+0.148, p_holm=0.848, CI [-0.03, +0.38]
   - ContextRecall: 0.817 → 0.754, Δ=-0.063, p_holm=1.0, CI [-0.27, +0.14]

5. **[md]** "Interpretation" — one paragraph: no metric reaches α=0.05 post-Holm; Context Precision is the strongest signal for enhanced (+0.148). Enhanced's value = **selective abstention + retrieval precision**, not blanket quality.

6. **[md]** "CRAG-specific metrics"

7. **[code]** Render table from `crag_metrics.json`:
   - rewrite_trigger_rate=0.20 · rewrite_recovery_rate=0.40
   - hallucination_flag_rate=0.04 · hallucination_recovery_rate=0.80
   - metadata_filter_rate=0.92 · rerank_top1_change_rate=0.57
   - abstain_rate=0.16 · abstain_correctness=0.25 · **should_abstain_recall=1.00**
   - mean_chunks_retrieved=8.28 · mean_chunks_used=2.72

8. **[md]** CRAG interpretation — rerank earns its keep (top-1 flip 57% of the time); abstain recall perfect on the question that should abstain (q21); abstain precision weak (0.25) due to q06/q10/q24 false positives — documented limitation.

9. **[md]** "Per-category analysis"

10. **[code]** Load `per_category.csv`. Produce a grouped bar chart (baseline vs enhanced per category) using `matplotlib`. Save to `figures/per_category_comparison.png`. Also render the table inline. Call out: comparative-query Recall collapse is the largest within-category delta (this is the weakness we'd address with Rule 0.5 if Tier 1 #4 fires).

11. **[md]** "Cross-evaluator consistency (methodology caveat)"

12. **[code]** Load both `evaluator_divergence_*.csv`. Summarise in 2-3 sentences: Sonnet is stricter on context metrics than gpt-4o-mini, same sign of delta on Faithfulness/AnswerRelevancy. Methodology caveat, not a replacement judgment.

13. **[md]** "Evaluation summary" — one paragraph: baseline is a strong incumbent; enhanced wins on Context Precision + abstain recall; dominant weakness = over-aggressive abstention on 3 in-corpus questions. Flagged for iteration.

### Expected marker-visible outputs
- Reproducibility banner citing `test_set_hash`
- Headline RAGAS table with deltas, p-values, Holm-adj, CIs
- CRAG metrics table
- PNG chart (grouped bar by category)
- Cross-evaluator summary
- Written interpretation under every table

---

## Pre-execution checklist

Run every item before opening Jupyter. Stop and fix if any fails.

- [ ] `git status` clean (or stashed)
- [ ] `git tag pre-notebook-run` created
- [ ] `.venv` activated; `python -c "import boe_rag"` succeeds
- [ ] `pip install matplotlib jupyter nbconvert` (matplotlib not in current deps; jupyter/ipykernel already are)
- [ ] `.env` has all three keys (OPENAI for NB1 sanity query, ANTHROPIC + COHERE for NB2 smoke test)
- [ ] `pytest -x -q` green (confidence check)
- [ ] `data/evaluation_results/ragas_aggregate.json` parses; `test_set_hash` matches committed state
- [ ] Chosen query IDs (q01, q11, q13, q15, q21, q24 + q02 smoke) all present in `enhanced_results.json`
- [ ] `figures/` directory created at repo root

---

## Execution protocol

1. Author cells first (no execution). Commit: `wip: notebook cell content drafted`.
2. Execute NB3 first — easiest, no API calls. `jupyter nbconvert --to notebook --execute --inplace notebooks/03_evaluation.ipynb`. Timeout 120 s. Confirm `figures/per_category_comparison.png` written.
3. Execute NB1 — sanity query is one OpenAI call. Same command. Timeout 120 s.
4. Execute NB2 last — has the one live smoke test. `--ExecutePreprocessor.timeout=300`. Watch for Anthropic / Cohere / OpenAI failures; if smoke cell fails, re-run just that cell manually.
5. Visual review: open each notebook in Jupyter or VS Code. Confirm every cell has output, no stack traces, no leaked keys.
6. Export NB2 → PDF: `jupyter nbconvert --to pdf notebooks/02_baseline_and_enhanced.ipynb --output-dir . --output demo_log`. Falls back to HTML + browser-print if LaTeX missing.
7. Commit: `feat(notebooks): populated with saved outputs + demo_log.pdf`. Tag `submission-ready-notebooks`.

---

## Failure modes + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| NB1 sanity query returns empty | Low | Cosmetic | Collections already have 290+ chunks; embeddings exist. If empty, check collection name match in `config.py` |
| NB2 smoke test fails (API outage) | Medium | Cosmetic | Wrap in try/except; if fails, replace with a printed note "API unavailable at time of execution; see NB3 for full run results" |
| `nbconvert --to pdf` fails on no-LaTeX | High on fresh Mac | Blocks demo_log.pdf | Fallback: `--to html` → open in browser → Cmd-P → save as PDF. Documented in protocol. |
| Notebook kernel points at system Python, not `.venv` | Medium | Imports fail | Before first execution: `python -m ipykernel install --user --name boe-rag-venv --display-name "BoE RAG (.venv)"`; select this kernel in notebook metadata |
| Cell prints an API key | Catastrophic | Secrets leak | Never `print(os.environ)`. `load_dotenv()` followed by `assert os.getenv("...")` — never echo values |
| Notebook JSON bloats past 5 MB | Low | Git push pain | Truncate displayed text to 300 chars via `textwrap.shorten` |
| matplotlib pip install fails on Mac (rare) | Very low | Blocks NB3 chart | Fallback: use `df.plot.bar(...)` which internally requires matplotlib — same failure mode — if both fail, skip the chart and keep the table only |
| Query ID mismatch (e.g. q15 behaves differently at load time than I expected) | Low | Narrative breaks | Swap rules documented in demo slate. q03 is the backup rewrite case; q10 the backup false-positive abstain. |
| Git tag already exists | Low | Push blocks | `git tag -d pre-notebook-run; git tag pre-notebook-run` to re-point |

---

## Cost + time estimate (revised)

| Step | Cost | Time |
|---|---|---|
| Author cell content | $0 | ~35 min |
| Execute NB1 (1 OpenAI sanity query) | <$0.01 | ~1 min |
| Execute NB2 (1 smoke test = ~3 API calls) | ~$0.05 | ~1 min |
| Execute NB3 (load + chart) | $0 | ~30 s |
| PDF export + visual review | $0 | ~15 min |
| **Total** | **~$0.06** | **~55 min** |

vs v1 estimate: $1.55 and ~1h. We saved ~$1.50 AND removed integrity risk.

---

## Verification gates

- [ ] All 3 notebooks have saved outputs on every code cell (no `[*]` / `In []`)
- [ ] No `.env` or key contents appear in any cell output
- [ ] NB1 reproducibility banner shows git SHA matching `HEAD`
- [ ] NB2 has visible pipeline_trace with `rewrite_query` (q15) AND `abstain_out_of_corpus` (q21) AND pre_rerank_ids != post_rerank_ids (q01 or q13)
- [ ] NB2 summary DataFrame has 6 rows with correct categories
- [ ] NB3 headline RAGAS table deltas + p_holm values match `ragas_aggregate.json` exactly
- [ ] `figures/per_category_comparison.png` exists and opens
- [ ] `demo_log.pdf` exists at repo root with ≥6 example sections
- [ ] `git tag submission-ready-notebooks` applied
- [ ] `du -sh notebooks/` under 5 MB

---

## Open decisions still needing your sign-off

1. **Do you accept the v1→v2 scope reduction on live pipeline runs in NB2?** Ram's recommendation: **yes**. Integrity with the report is worth more than theatre.
2. **OK to unify NB2 with demo_log.pdf (one deliverable)?** Ram: **yes**, spec 09 allows it, removes duplicated work.
3. **OK to `pip install matplotlib` during pre-flight** (minor dependency addition)? Ram: **yes**, tiny footprint, chart materially improves NB3.
4. **VS Code Jupyter UI or command-line `nbconvert --execute` or Jupyter browser?** Ram: **VS Code** — immediate feedback, no kernel registration dance, most comfortable on your Mac.
5. **Exact 6 demo query slate approved?** (q11, q01, q15, q24, q21, q13 + q02 smoke)

---

## One-sentence version

NB1 reads committed state + 1 sanity query; NB2 loads locked results for the 6-query CRAG showcase with a single live smoke test on q02, then exports to `demo_log.pdf`; NB3 loads evaluation JSON + emits one chart — total ~$0.06, ~55 min, zero integrity risk, collapses the demo-log deliverable.
