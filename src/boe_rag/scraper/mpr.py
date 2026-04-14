"""Scraper for Bank of England Monetary Policy Reports.

Handles chapter/section headings, box analyses (kept intact with markers),
chart placeholders (image discarded, title preserved), and tables (extracted
via pandas.read_html).

Structural markers emitted (chunker contract):
  ## / ###              - section / sub-section headings
  [BOX START: ...]      - opening of an analysis box
  [BOX END]             - closing of an analysis box
  [CHART: <title>]      - chart title (trailing index, see note below)
  [TABLE: <title>]      - table title + plain-text body

Chart / table markers are appended at the end of the document rather than
inline. The chunker treats the trailing marker block as a dedicated
'chart index' chunk.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
from bs4 import Tag

from boe_rag.scraper.base import BaseScraper, has_class, is_in_ancestor

logger = logging.getLogger(__name__)


class MPRScraper(BaseScraper):
    """Scraper for Monetary Policy Report pages."""

    def _extract_chart_titles(self, content: Tag) -> list[str]:
        """Collect chart titles, excluding img-blocks that wrap a <table>.

        Returns titles in document order. Table titles are excluded here so
        they are not double-emitted (once as [CHART: ...] and once as
        [TABLE: ...]).
        """
        titles: list[str] = []
        for block in content.select("div.img-block"):
            if block.find("table") is not None:
                continue  # handled by _extract_tables
            title_el = block.select_one("h3.img-title")
            if title_el is None:
                continue
            titles.append(title_el.get_text(" ", strip=True))
        return titles

    def _extract_tables(self, content: Tag) -> list[str]:
        """Extract every <table> as '[TABLE: <title>]\\n<plain-text table>'.

        Tables can appear either inside div.img-block (MPR forecast table) or
        standalone with a preceding h3.img-title (FSR stress-test annex). We
        prefer the block-local title when available, otherwise fall back to
        the nearest preceding h3.img-title sibling.

        Also strips the preceding h3.img-title so it is not emitted as a
        duplicate '### heading' by the walker.
        """
        results: list[str] = []
        to_decompose: list[Tag] = []

        for table_el in content.select("table"):
            title, title_source = _find_table_title(table_el)
            try:
                dfs = pd.read_html(StringIO(str(table_el)))
            except ValueError:
                logger.warning("pandas.read_html failed on %s", title)
                to_decompose.append(table_el)
                if title_source is not None:
                    to_decompose.append(title_source)
                continue
            if not dfs:
                to_decompose.append(table_el)
                if title_source is not None:
                    to_decompose.append(title_source)
                continue
            body = dfs[0].to_string(index=False)
            results.append(f"[TABLE: {title}]\n{body}")
            to_decompose.append(table_el)
            if title_source is not None:
                to_decompose.append(title_source)

        # Strip tables + their titles from the tree so the walker does not
        # emit them as raw text or duplicate ### headings.
        for el in to_decompose:
            el.decompose()
        return results

    def _walk_content_tree(
        self,
        content: Tag,
        charts: list[str],
        tables: list[str],
    ) -> str:
        """Walk the content, emitting markdown headings, box markers, and passages."""
        lines: list[str] = []

        for el in _descend_in_order(content):
            name = el.name

            if name in ("h2", "h3"):
                if is_in_ancestor(el, class_name="box-highlight"):
                    continue  # handled by box renderer
                if is_in_ancestor(el, tag_name="li"):
                    continue  # handled by li renderer
                text = el.get_text(" ", strip=True)
                if text:
                    marker = "## " if name == "h2" else "### "
                    lines.append(f"{marker}{text}")
            elif name == "p":
                if is_in_ancestor(el, class_name="box-highlight"):
                    continue
                if is_in_ancestor(el, tag_name="li"):
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(text)
            elif name == "li":
                if is_in_ancestor(el, class_name="box-highlight"):
                    continue
                text = el.get_text(" ", strip=True)
                if text:
                    lines.append(f"- {text}")
            elif name == "div" and has_class(el, "box-highlight"):
                lines.append(_render_box(el))

        # Append chart / table markers as a trailing index (see module docstring).
        for chart_title in charts:
            lines.append(f"[CHART: {chart_title}]")
        for table_text in tables:
            lines.append(table_text)

        return "\n\n".join(lines)


def _find_table_title(table_el: Tag) -> tuple[str, Tag | None]:
    """Locate the best title for a <table> and the element it came from.

    Resolution order:
      1. An <h3 class="img-title"> inside the same <div class="img-block">
         ancestor as the table.
      2. The closest preceding <h3 class="img-title"> anywhere before the table.
      3. Literal fallback 'Table'.

    Returns:
        (title, source_element). source_element is returned so callers can
        decompose it after extraction to prevent double-emission.
    """
    for ancestor in table_el.parents:
        if isinstance(ancestor, Tag) and has_class(ancestor, "img-block"):
            title_el = ancestor.select_one("h3.img-title")
            if title_el is not None:
                return title_el.get_text(" ", strip=True), title_el
            break

    prev_title = table_el.find_previous(class_=lambda c: c and "img-title" in c)
    if isinstance(prev_title, Tag):
        return prev_title.get_text(" ", strip=True), prev_title

    return "Table", None


def _descend_in_order(root: Tag) -> list[Tag]:
    """Yield descendants in document order, treating box-highlight divs as atomic.

    Without this pruning, paragraphs inside a box would be emitted both by the
    outer walk and by _render_box.
    """
    out: list[Tag] = []
    _walk(root, out)
    return out


def _walk(node: Tag, out: list[Tag]) -> None:
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if has_class(child, "box-highlight"):
            out.append(child)  # emit the box itself, do not recurse
            continue
        out.append(child)
        _walk(child, out)


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
        # Avoid double-emission from <p> inside <li> within the box.
        if is_in_ancestor(p, tag_name="li"):
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
