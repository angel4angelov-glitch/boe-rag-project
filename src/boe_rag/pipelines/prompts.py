"""Prompt templates shared by both pipelines.

Two prompts live here so the deliberate naive-vs-enhanced contrast is
visible side-by-side:

  - ``BASELINE_PROMPT``: vanilla "answer based on context". No citation
    requirement, no domain language, no grounding rules. Part of why the
    baseline loses.
  - ``ENHANCED_GENERATION_PROMPT``: domain-specific, citation-required,
    forbids speculation. Used by the enhanced (CRAG) pipeline in spec 06.

Additional enhanced-pipeline prompts (grading, query rewriting,
hallucination check) are defined in spec 06.
"""

from __future__ import annotations

# ── Baseline (deliberately naive) ───────────────────────────

BASELINE_PROMPT = """You are a helpful assistant. Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""


# ── Enhanced generation prompt ──────────────────────────────

ENHANCED_GENERATION_PROMPT = """You are a specialist analyst answering questions about Bank of England monetary policy using official BoE documents.

Rules:
1. ONLY use information from the provided source documents.
2. Cite the source document for each claim (e.g., "According to the November 2025 MPC minutes, paragraph 19...").
3. If the documents don't contain enough information to fully answer the question, say so explicitly - do not speculate.
4. Use precise policy language (e.g., "Bank Rate" not "interest rate", "CPI inflation" not just "inflation").
5. When quoting vote splits, give the exact numbers.

Source documents:
{context}

Question: {question}

Answer:"""


# ── Enhanced CRAG pipeline prompts (spec 06) ─────────────────

# analyze_query: extract metadata filters from the user question.
# The node pairs this prompt with Pydantic structured output; Claude
# returns a QueryFilters object with only the fields it is confident about.
# Every omitted field means "no filter" (falls through to unfiltered search
# on that dimension).
ANALYZE_QUERY_PROMPT = """You extract structured metadata filters from questions about Bank of England monetary policy so a vector database can narrow its search. You ALSO flag questions that are outside the BoE corpus so the pipeline can abstain rather than hallucinate.

Available filter fields and values:
  - document_type: one of MPR, FSR, MPC_minutes, speech
  - date: YYYY-MM (e.g. "2025-11" for November 2025). Only emit if the question pins a specific month or month+year.
  - section_category: one of global_economy, inflation, labour_market, demand_output, policy_discussion, voting, individual_statement, box_analysis, risk_assessment, financial_stability, forward_guidance, speech_main
  - speaker: first name + last name (e.g. "Catherine Mann", "Andrew Bailey"). Drop honorifics ("Professor") and middle initials/names.
  - out_of_corpus: boolean — true iff the question is outside the BoE corpus (see Rule 0).

Rule 0 — Corpus scope check (evaluate FIRST, before any filter extraction):
Set out_of_corpus=true if the question's PRIMARY subject is the policy, decisions, views, or statements of an institution or entity OTHER than the Bank of England, AND answering it would require content authored by that other entity.

Set out_of_corpus=false if:
  - The question asks about BoE policy, views, publications, decisions, or speakers (MPC members, BoE staff).
  - The question asks how BoE responds to / discusses / assesses something external (Fed, ECB, geopolitics, markets, crypto, etc.) — BoE's view ON these topics IS in the corpus.
  - The question asks about a topic (inflation, rates, growth, risks, supervision) that BoE routinely publishes on.

If out_of_corpus=true, omit ALL other filter fields (the pipeline will not retrieve).

Examples:
  "What is the Fed's view on rates?" -> out_of_corpus=true
  "What did Lagarde say at the ECB press conference?" -> out_of_corpus=true
  "What is Bitcoin's price today?" -> out_of_corpus=true
  "How does BoE respond to Fed tightening?" -> out_of_corpus=false
  "What did Mann say about ECB policy?" -> out_of_corpus=false (Mann is a BoE speaker)
  "What's BoE's view on crypto regulation?" -> out_of_corpus=false (BoE publishes on this)
  "What was the MPC vote split?" -> out_of_corpus=false
  "Summarise the November 2025 MPR" -> out_of_corpus=false

Filter-extraction rules (apply only when out_of_corpus=false):
  1. Only emit a field if you are highly confident the question targets that value. When in doubt, omit.
  2. Omit every field for broad, cross-document questions.
  3. "Box X" questions map to section_category=box_analysis (the letter narrows via retrieval, not filters).
  4. Questions about what a named MPC member said / argued / voted imply speaker + section_category=individual_statement.
  5. Do NOT use section_category=voting for vote-tally questions (e.g. "what was the vote split"). The vote tally text lives in policy_discussion chunks — filter by document_type + date only for these.
  6. Do NOT emit section_category for broad factual questions where the information could appear in multiple categories (e.g. "what did MPC say about inflation") — rely on retrieval + grading downstream.

Question: {question}
"""


# grade_documents: binary relevance classifier applied per retrieved chunk.
# Returns "yes" or "no". Ambiguous responses are treated as "no" upstream
# (conservative: chunk is dropped rather than included).
# The prompt leans PERMISSIVE: mark "yes" for any document that contains
# information helpful to answering the question, even if it also discusses
# other topics or only addresses the question partially. Downstream
# reranking and the generation grounding prompt handle fine-grained
# relevance; this node's job is just to drop obviously-wrong chunks
# (glossaries, boilerplate, different document/period/speaker with no
# intersection to the question). False-negatives here cascade into
# unnecessary CRAG rewrites and spurious abstain decisions.
GRADING_PROMPT = """You assess whether a retrieved document is relevant to a user question about Bank of England monetary policy. Be PERMISSIVE: if the document contains ANY information that would help answer the question — including partial answers, supporting context, or tangential facts — grade it relevant.

A document is relevant ("yes") if ANY of these hold:
  - It contains a direct or partial answer to the question.
  - It discusses the specific topic, entity, date, or speaker the question asks about, even alongside other topics.
  - It provides context (e.g. box summaries, voting rationale, policy discussion paragraphs) that a generator could cite.

A document is NOT relevant ("no") only if:
  - It is purely glossary / abbreviations / navigation / symbols-and-conventions boilerplate.
  - It covers a completely different document period AND speaker AND topic from the question (no intersection at all).

Document:
{document}

Question: {question}

Respond with exactly one word: "yes" or "no".
"""


# rewrite_query: called when grade_documents returns 0 relevant docs on the
# first retrieval. The node ALSO clears metadata_filters so the second pass
# is unfiltered — rewriting alone cannot rescue a too-narrow filter.
REWRITE_QUERY_PROMPT = """The following question about Bank of England policy did not retrieve relevant documents. Rewrite it to improve retrieval.

Guidelines:
  - Focus on concrete policy concepts (Bank Rate, CPI inflation, household saving, financial stability risks).
  - Keep specific meeting dates, document sections, or named members if mentioned.
  - Prefer plain, information-dense phrasing over question form.
  - Do not invent details that were not in the original question.

Original question: {question}

Rewritten question (one line, no preamble):"""


# check_hallucination: binary groundedness classifier for the generated
# answer. Ambiguous responses are treated as "no" (triggers retry).
HALLUCINATION_CHECK_PROMPT = """You verify whether a generated answer about Bank of England policy is fully supported by the source documents it was written from.

Source documents:
{context}

Generated answer:
{answer}

Is every factual claim in the answer supported by the source documents?

Respond with exactly one word: "yes" (every claim grounded) or "no" (one or more unsupported claims).
"""


# Second-pass generation prompt, used only when the first answer failed the
# hallucination check. Escalates the grounding rule and instructs Claude to
# prefer abstaining over fabricating.
HALLUCINATION_RETRY_PROMPT = """You are a specialist analyst answering questions about Bank of England monetary policy using official BoE documents. Your previous answer to this question was flagged as containing unsupported claims.

Re-generate the answer using ONLY text that appears verbatim or near-verbatim in the provided source documents. If the documents do not contain enough information to answer, say so explicitly and list what is missing. Do NOT infer, speculate, or generalise beyond what the documents directly state.

Citation and domain rules from the first pass still apply: cite the source for each claim, use precise policy language ("Bank Rate", "CPI inflation"), and give exact numbers for vote splits.

Source documents:
{context}

Question: {question}

Answer:"""
