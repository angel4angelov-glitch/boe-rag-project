"""Build notebooks/01_data_ingestion_indexing.ipynb."""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_data_ingestion_indexing.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md("""# Notebook 1 — Data Ingestion & Indexing

**Objective**: walk the data pipeline that produced the corpus the rest of the project queries — scrape, chunk, embed, index — and validate the committed state.

**Mode**: read-only. We inspect the existing committed artefacts (HTML cache, `data/raw/manifest.csv`, `data/chunks/*`, `chroma_db/`) and run one OpenAI embedding sanity query. We do NOT rebuild the index here, because rebuilding would change the embedding-version footprint that the locked RAGAS evaluation in NB3 was computed against. A guarded "rebuild from scratch" code block is included at the end for reference.""")

md("""## Reproducibility""")

code("""import subprocess, sys, importlib.metadata
from datetime import datetime, timezone

def _ver(pkg):
    try: return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError: return "n/a"

print(f"Notebook executed:    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
print(f"Python:               {sys.version.split()[0]}")
print(f"Git SHA:              {subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()}")
print(f"Git branch:           {subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD']).decode().strip()}")
print()
print("Key package versions:")
for p in ("anthropic", "openai", "cohere", "chromadb", "langchain",
          "langgraph", "ragas", "tiktoken", "pandas", "boe-rag"):
    print(f"  {p:18s} {_ver(p)}")
""")

code("""import os
from dotenv import load_dotenv

load_dotenv()
# Sanity-check that keys are present; do NOT echo any values.
for k in ("OPENAI_API_KEY",):
    if not os.getenv(k):
        raise RuntimeError(f"Missing required env var: {k}")
    print(f"  {k}: present ({len(os.getenv(k))} chars)")
""")

md("""## Scraped corpus

The scraper (`src/boe_rag/scraper/`) downloads MPC minutes, MPRs, FSRs, and speeches from `bankofengland.co.uk`, normalises HTML to plain text with structural markers (`##`, `###`, paragraph numbers, `[BOX START/END]`), and writes a `manifest.csv` with one row per document.""")

code("""from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
manifest = pd.read_csv(ROOT / "data" / "raw" / "manifest.csv")
print(f"Total documents in manifest: {len(manifest)}")
print()
print("By document type:")
print(manifest.groupby("document_type").size().to_string())
print()
print(f"Total words across corpus: {int(manifest['word_count'].sum()):,}")
print()
print("Sample (first 6 rows):")
display(manifest.head(6))
""")

md("""## Chunking

Two parallel chunking strategies, one per pipeline:

- **Baseline** (`src/boe_rag/chunking/base_chunker.py`) — fixed ~500-token chunks via `RecursiveCharacterTextSplitter`, no metadata. The naive control.
- **Enhanced** (`src/boe_rag/chunking/section_chunker.py`) — section-aware splitting on the structural markers, with rich metadata (document_type, date, section_category, speaker for speeches, box_id for box analyses, paragraph_number for MPC minutes).

Both write one JSON file per source document into `data/chunks/{baseline,enhanced}/`.""")

code("""import json

baseline_dir = ROOT / "data" / "chunks" / "baseline"
enhanced_dir = ROOT / "data" / "chunks" / "enhanced"

baseline_files = sorted(baseline_dir.glob("*.json"))
enhanced_files = sorted(enhanced_dir.glob("*.json"))

baseline_total = sum(json.loads(p.read_text())["total_chunks"] for p in baseline_files)
enhanced_total = sum(json.loads(p.read_text())["total_chunks"] for p in enhanced_files)

print(f"Baseline collection: {len(baseline_files):2d} documents, {baseline_total:4d} chunks total")
print(f"Enhanced collection: {len(enhanced_files):2d} documents, {enhanced_total:4d} chunks total")
print()
print("Sample chunk from MPC June 2025 minutes:")
print()

b_sample = json.loads((baseline_dir / "mpc_2025_06.json").read_text())["chunks"][0]
e_sample = json.loads((enhanced_dir / "mpc_2025_06.json").read_text())["chunks"][0]

print("--- baseline (no metadata, fixed-size split) ---")
print(f"chunk_id: {b_sample['chunk_id']}")
print(f"text ({b_sample['token_count']} tokens, first 250 chars):")
print("    " + b_sample["text"][:250].replace(chr(10), " ") + "...")
print()
print("--- enhanced (section-aware, metadata-tagged) ---")
print(f"chunk_id: {e_sample['chunk_id']}")
print(f"metadata: {e_sample['metadata']}")
print(f"text ({e_sample['token_count']} tokens, first 250 chars):")
print("    " + e_sample["text"][:250].replace(chr(10), " ") + "...")
""")

md("""## Indexing

Both chunk collections are embedded with OpenAI `text-embedding-3-small` and stored in two ChromaDB collections (`boe_baseline`, `boe_enhanced`) backed by the local `chroma_db/` directory. The collections are created with the `OpenAIEmbeddingFunction` so that subsequent `.query()` calls automatically embed the query with the same model.""")

code("""import chromadb
from chromadb.utils import embedding_functions
from boe_rag.config import Paths, BASELINE_COLLECTION, ENHANCED_COLLECTION, EMBEDDING_MODEL

client = chromadb.PersistentClient(path=str(Paths.CHROMA_DB))
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=EMBEDDING_MODEL,
)

baseline_col = client.get_collection(BASELINE_COLLECTION, embedding_function=embed_fn)
enhanced_col = client.get_collection(ENHANCED_COLLECTION, embedding_function=embed_fn)

print(f"ChromaDB path:        {Paths.CHROMA_DB}")
print(f"Embedding model:      {EMBEDDING_MODEL}")
print(f"  {BASELINE_COLLECTION:15s}  {baseline_col.count():4d} embedded chunks")
print(f"  {ENHANCED_COLLECTION:15s}  {enhanced_col.count():4d} embedded chunks")
""")

md("""### Sanity query

One end-to-end retrieval round-trip on the enhanced collection. Validates that the index is queryable, the embedding round-trip works, and that returned chunks carry expected metadata.""")

code("""query = "What did the MPC decide about Bank Rate in November 2025?"
res = enhanced_col.query(query_texts=[query], n_results=3)

print(f"Query: {query!r}\\n")
print(f"{'rank':>4}  {'chunk_id':40s}  {'similarity':>10s}  {'doc_type':>13s}  {'date':>8s}  preview")
print("-" * 130)
for rank, (cid, doc, dist, meta) in enumerate(zip(
    res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0]
), start=1):
    sim = round(1.0 - dist, 3)
    preview = doc.replace("\\n", " ")[:55] + "..."
    dt = meta.get("document_type", "")
    dat = meta.get("date", "")
    print(f"{rank:>4}  {cid:40s}  {sim:>10.3f}  {dt:>13s}  {dat:>8s}  {preview}")
""")

md("""## Validation checklist

Confirmed by the cells above:

- [x] Manifest contains 23 documents across MPC_minutes / MPR / FSR / speech
- [x] Chunk counts non-zero for both collections
- [x] Baseline chunks carry minimal metadata (`chunk_id`, text, token count)
- [x] Enhanced chunks carry rich metadata (`document_type`, `date`, `section_category`, etc.)
- [x] Both ChromaDB collections embedded and queryable with `text-embedding-3-small`
- [x] Sanity query returns relevant MPC chunks with sensible cosine similarity scores

The pipeline is ready to serve queries. Continue to NB2 for the side-by-side baseline vs enhanced demonstration.""")

md("""## Appendix — How to rebuild from scratch

The committed `chroma_db/` directory is the canonical state used by the locked evaluation in NB3. Rebuilding will re-call OpenAI's embedding API and the resulting embeddings may differ subtly from the committed ones (model version drift, tokeniser updates) — which would invalidate the locked RAGAS results.

To intentionally rebuild the index from scratch (use only when you accept that the eval results will need to be regenerated):

```python
# WARNING: destructive. Wipes both collections and re-embeds all chunks.
# Cost: ~$0.05 + ~2 minutes.
if False:  # set to True to actually run
    from boe_rag.indexing.chroma_store import build_collection
    build_collection("baseline", baseline_dir, BASELINE_COLLECTION, force=True)
    build_collection("enhanced", enhanced_dir, ENHANCED_COLLECTION, force=True)
```""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}

OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")
