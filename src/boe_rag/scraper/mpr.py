"""Scraper for Bank of England Monetary Policy Reports.

Handles chapter/section headings, box analyses (kept intact with markers),
chart placeholders (image discarded, title preserved), and tables (extracted
via pandas.read_html).
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup, Tag

from boe_rag.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class MPRScraper(BaseScraper):
    """Scraper for Monetary Policy Report pages."""

    def _extract_chart_titles(self, content: Tag) -> list[str]:
        """Collect chart titles before div.img-block is stripped.

        Each chart title (e.g. 'Chart 1.1: CPI inflation...') is captured
        from the h3.img-title inside div.img-block. Order is preserved so
        _walk_content_tree can re-insert them in sequence.
        """
        return [
            title.get_text(" ", strip=True)
            for title in content.select("div.img-block h3.img-title")
        ]

    def _extract_tables(self, content: Tag) -> list[str]:
        """Extract tables as plain-text via pandas.read_html.

        Returns one string per table in document order, formatted as
        '[TABLE: <title>]\n<plain-text table>'. The caller is responsible for
        stripping table elements from the content tree so they aren't emitted
        twice.
        """
        results: list[str] = []
        for table_el in content.select("table"):
            title_el = table_el.find_previous(class_=lambda c: c and "img-title" in c)
            title = title_el.get_text(" ", strip=True) if title_el else "Table"
            try:
                dfs = pd.read_html(StringIO(str(table_el)))
            except ValueError:
                logger.warning("pandas.read_html failed on %s", title)
                continue
            if not dfs:
                continue
            body = dfs[0].to_string(index=False)
            results.append(f"[TABLE: {title}]\n{body}")
        # Strip tables from the tree so _walk_content_tree doesn't emit them as noisy text.
        for table_el in content.select("table"):
            table_el.decompose()
        return results

    def _walk_content_tree(
        self,
        content: Tag,
        charts: list[str],
        tables: list[str],
    ) -> str:
        """Walk the content, emitting markdown headings, box markers, and passages."""
        lines: list[str] = []
        chart_iter = iter(charts)
        table_iter = iter(tables)

        # Iterate top-level children; handle sections and their box-highlights specially.
        for el in _descend_in_order(content):
            name = el.name

            if name == "h2":
                if _is_in_box(el):
                    # Box heading handled by box renderer; skip to avoid duplicates.
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"## {text}")
            elif name == "h3":
                if _is_in_box(el):
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"### {text}")
            elif name == "p":
                if _is_in_box(el):
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(text)
            elif name == "li":
                if _is_in_box(el):
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"- {text}")
            elif name == "div" and _has_class(el, "box-highlight"):
                lines.append(_render_box(el))
            elif name == "div" and _has_class(el, "img-block-placeholder"):
                # Placeholder we can insert [CHART: ...] markers at later.
                pass

        # Append any leftover chart/table markers at end of document for context.
        for chart_title in chart_iter:
            lines.append(f"[CHART: {chart_title}]")
        for table_text in table_iter:
            lines.append(table_text)

        return "\n\n".join(lines)


def _descend_in_order(root: Tag) -> list[Tag]:
    """Yield tags in document order, skipping descendants of box-highlight divs.

    We handle boxes as atomic units — the box renderer walks their inner
    content separately. Without this pruning we'd emit every box paragraph twice.
    """
    out: list[Tag] = []
    _walk(root, out, skip_inside_box=True)
    return out


def _walk(node: Tag, out: list[Tag], *, skip_inside_box: bool) -> None:
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if skip_inside_box and _has_class(child, "box-highlight"):
            out.append(child)  # emit the box itself, don't recurse
            continue
        out.append(child)
        _walk(child, out, skip_inside_box=skip_inside_box)


def _is_in_box(el: Tag) -> bool:
    """True if el is inside a div.box-highlight ancestor."""
    for parent in el.parents:
        if isinstance(parent, Tag) and _has_class(parent, "box-highlight"):
            return True
    return False


def _has_class(el: Tag, cls: str) -> bool:
    classes = el.get("class") or []
    return cls in classes


def _render_box(box: Tag) -> str:
    """Render a div.box-highlight with [BOX START: title] ... [BOX END] markers."""
    heading = box.find(class_=lambda c: c and "box__h2" in c) or box.find("h2")
    title = heading.get_text(" ", strip=True) if heading else "Box"

    summary_el = box.find("p", class_=lambda c: c and "text-box-highlight" in c)
    summary = summary_el.get_text(" ", strip=True) if summary_el else ""

    body_paragraphs: list[str] = []
    for p in box.find_all("p"):
        if p is summary_el:
            continue
        text = p.get_text(" ", strip=True)
        if text:
            body_paragraphs.append(text)

    parts = [f"[BOX START: {title}]"]
    if summary:
        parts.append(f"[Summary: {summary}]")
    parts.extend(body_paragraphs)
    parts.append("[BOX END]")
    return "\n\n".join(parts)
