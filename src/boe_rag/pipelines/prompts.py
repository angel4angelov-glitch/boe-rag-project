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
