"""Render report.md -> report.html (then Chrome headless writes report.pdf).

Run from repo root:
    python scripts/build_report_pdf.py
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --headless --disable-gpu --no-pdf-header-footer \
        --print-to-pdf="$(pwd)/report.pdf" "file://$(pwd)/report.html"
"""
from __future__ import annotations

import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report.md"
OUT = ROOT / "report.html"

CSS = """
<style>
  @page { size: A4; margin: 22mm 18mm; }
  body {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
    max-width: 720px;
    margin: 0 auto;
  }
  h1 { font-size: 18pt; margin: 0 0 6pt 0; }
  h2 { font-size: 13pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
  h3 { font-size: 11pt; margin: 12pt 0 4pt 0; }
  p { margin: 4pt 0 8pt 0; text-align: justify; }
  table { border-collapse: collapse; margin: 6pt 0 12pt 0; font-size: 9.5pt; width: 100%; }
  th, td { border: 1px solid #999; padding: 3pt 6pt; vertical-align: top; }
  th { background: #f0f0f0; text-align: left; }
  code { font-family: "SF Mono", Menlo, monospace; font-size: 9.5pt; background: #f5f5f5; padding: 0 2pt; }
  img { max-width: 100%; height: auto; display: block; margin: 6pt auto; }
  hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
  strong { font-weight: 600; }
</style>
"""

html_body = markdown.markdown(
    SRC.read_text(),
    extensions=["tables", "fenced_code", "sane_lists"],
)

OUT.write_text(
    f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title>{CSS}</head><body>{html_body}</body></html>"
)
print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
