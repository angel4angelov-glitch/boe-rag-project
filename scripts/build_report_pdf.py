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
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #000;
  }
  h1 { font-family: Arial, Helvetica, sans-serif; font-size: 16pt; font-weight: bold; margin: 0 0 8pt 0; line-height: 1.3; }
  h2 { font-family: Arial, Helvetica, sans-serif; font-size: 13pt; font-weight: bold; margin: 16pt 0 6pt 0; line-height: 1.3; }
  h3 { font-family: Arial, Helvetica, sans-serif; font-size: 11.5pt; font-weight: bold; margin: 12pt 0 4pt 0; line-height: 1.3; }
  p { margin: 0 0 8pt 0; text-align: justify; }
  ul, ol { margin: 4pt 0 8pt 0; padding-left: 22pt; }
  li { margin-bottom: 3pt; }
  table { border-collapse: collapse; margin: 6pt 0 12pt 0; font-size: 10pt; width: 100%; line-height: 1.3; }
  th, td { border: 1px solid #000; padding: 4pt 6pt; vertical-align: top; }
  th { background: #e8e8e8; text-align: left; font-weight: bold; }
  code { font-family: "Courier New", Courier, monospace; font-size: 10pt; background: #f2f2f2; padding: 0 2pt; }
  img { max-width: 100%; height: auto; display: block; margin: 8pt auto; }
  hr { border: none; border-top: 1px solid #000; margin: 14pt 0; }
  strong { font-weight: bold; }
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
