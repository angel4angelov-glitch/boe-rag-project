# 02 — Data Ingestion

## Objective
Scrape all target BoE documents from bankofengland.co.uk and save clean text with structural markers to `data/raw/`. Every assertion in this spec is verified against actual BoE HTML as of April 2026.

## Depends on
01-PROJECT-SETUP (beautifulsoup4, requests, lxml installed)

## Deliverables
- [ ] Scraper module `src/boe_rag/scraper/` with per-document-type scrapers
- [ ] Raw text files in `data/raw/` with consistent naming
- [ ] Manifest CSV listing all scraped documents with metadata
- [ ] Every document verified: non-empty, correct type, reasonable length

---

## Verified BoE Website Structure (April 2026)

All three document types (MPC minutes, MPR, FSR) share the same core architecture:
- **Static HTML** — fully server-rendered, no JavaScript needed, `curl` gets everything
- **Server**: Microsoft IIS/10.0 behind Akamai CDN
- **Content container**: `div.page-content#content > div#output > section.page-section` (repeated)
- **Headings**: `<h2>` for chapters/sections, `<h3>` for sub-sections
- **Paragraphs**: Plain `<p>` tags throughout

### MPC Minutes
- **URL pattern**: `https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/{year}/{month-name}-{year}`
- **Content selector**: `div.col9 div.page-content` blocks — **note: TWO blocks exist.** First has only the `<h2>` summary heading. Second has all minutes text. Select ALL `div.page-content` elements within `div.col9` and concatenate content from both.
- **Paragraph numbering**: Inline text at paragraph start — `<p>1: Before turning...` (colon format, occasionally period)
- **Paragraph number regex**: `^\d+[:.]\s`
- **Headings in content**: `<h2>` (e.g., "Minutes of the Monetary Policy Committee meeting..."), `<h3>` (e.g., "The Committee's discussions", "The immediate policy decision")
- **Vote tallies**: Plain `<p>` text with member names, preceded by `<ul><li>` proposition. Vote groupings use `<p><strong>Votes to maintain/reduce...</strong></p>`
- **Individual member views**: `<p><strong>Name:</strong> rationale...</p>`
- **Artefacts to strip**: `<a id="_Hlk...">` anchors (Word paste artefacts). Note: cookie banners, sidebar `div.col3`, footer, and navigation are all **outside** `div.col9` and are excluded by the content selector — no explicit stripping needed.
- **HTML entities**: `&rsquo;` (curly apostrophes), `&ndash;` (en-dashes), `&nbsp;`

### Monetary Policy Reports
- **URL pattern**: `https://www.bankofengland.co.uk/monetary-policy-report/{year}/{month-name}-{year}`
- **Full report on ONE page** — single HTML file, ~3.4 MB, all chapters and boxes inline
- **Content selector**: `div.page-content#content > div#output > section.page-section` (repeated per chapter)
- **Chapter headings**: `<h2><a id="section1"></a>1: Current economic conditions</h2>`
- **Sub-sections**: `<h3>1.1: Inflation</h3>`
- **Box analyses**: Wrapped in `<div class="box-highlight">` with `<h2 class="box__h2"><a id="boxa"></a>Box A: ...</h2>`. Lead paragraph uses `<p class="text-box-highlight">`. **Boxes are in the HTML, not images.**
- **Charts**: Base64-encoded JPEG images (`<figure class="image"><img src="data:image/jpeg;base64,...">`) with titles in `<h3 class="h4 img-title">Chart 1.1: ...</h3>`. **Charts are opaque images — extract title/subtitle text only, not the image data.**
- **Tables**: Real HTML `<table class="table table--small">` elements — parseable. Only ~1 major table per report (e.g., Table 3.A: Forecast summary).
- **Sidebar TOC**: `nav.nav-chapters` (JS-populated, ignore)

### Financial Stability Reports
- **URL pattern**: `https://www.bankofengland.co.uk/financial-stability-report/{year}/{month-name}-{year}`
- **Full report on ONE page** — ~4.8 MB, 1,853 `<p>` tags, all chapters inline
- **Identical structure to MPR**: `section.page-section` per chapter, `<h2>` headings with anchor IDs, same box markup (`div.box-highlight`, `h2.box__h2`)
- **6 boxes** (A through F), each in its own `<section class="page-section">`
- **Annexes**: Same markup as chapters, distinguished by heading text ("Annex 1:", "Annex 2:", etc.)
- **Section numbering**: `section0` through `section17` (non-sequential, `section11` missing)

### Speeches
- **Listing page (`/news/speeches`) is AJAX-only** — loads via POST to `/_api/News/RefreshPagedNewsList`, protected by Akamai WAF. **Cannot scrape the listing page.**
- **Alternative: `/sitemap/speeches`** — static HTML, ~2,988 links organised by year. Use this as the index.
- **Speech detail pages are static HTML** — fully server-rendered.
- **URL pattern**: `https://www.bankofengland.co.uk/speech/{year}/{month}/{slug}`
- **Content selector**: `div.page-content#content > div#output > section.page-section`
- **Title**: `<h1 itemprop="name">`
- **Date**: `<div class="published-date">`
- **Speaker**: Sidebar `<div class="col3">` → `<a class="med-block-cta"> <h3>Andrew Bailey</h3>`
- **Summary**: `<div class="hero-paragraph"> <span class="hero"><p>...</p></span>`
- **Body**: `<h2>` section headings, `<p>` paragraphs, `<ul>/<ol>` lists
- **Footnotes**: `<div class="footnotes-container"> <ol>` at the end

---

## Target Documents

### MPC Minutes (7 documents)
| Meeting | URL |
|---------|-----|
| Jun 2025 | `.../monetary-policy-summary-and-minutes/2025/june-2025` |
| Aug 2025 | `.../monetary-policy-summary-and-minutes/2025/august-2025` |
| Sep 2025 | `.../monetary-policy-summary-and-minutes/2025/september-2025` |
| Nov 2025 | `.../monetary-policy-summary-and-minutes/2025/november-2025` |
| Dec 2025 | `.../monetary-policy-summary-and-minutes/2025/december-2025` |
| Feb 2026 | `.../monetary-policy-summary-and-minutes/2026/february-2026` |
| Mar 2026 | `.../monetary-policy-summary-and-minutes/2026/march-2026` |

**7 confirmed meetings.** If an April 2026 meeting is published before the deadline (16 April), add it. Otherwise 7 is the count. No TBDs.

### Monetary Policy Reports (4 documents)
| Report | URL |
|--------|-----|
| Feb 2025 | `.../monetary-policy-report/2025/february-2025` |
| May 2025 | `.../monetary-policy-report/2025/may-2025` |
| Aug 2025 | `.../monetary-policy-report/2025/august-2025` |
| Nov 2025 | `.../monetary-policy-report/2025/november-2025` |

### Financial Stability Reports (2 documents)
| Report | URL |
|--------|-----|
| Jul 2025 | `.../financial-stability-report/2025/july-2025` |
| Dec 2025 | `.../financial-stability-report/2025/december-2025` |

### Speeches (10 documents)
**Scoped to 10.** Selected by: MPC members only, 2025-2026, policy-relevant content.

**Selection process** (run once before scraping):
1. Fetch `/sitemap/speeches` (static HTML, all links)
2. Filter programmatically: links containing year `2025` or `2026`
3. Filter by speaker name in link text (case-insensitive grep for target names)
4. From the filtered list, manually pick 10 that are most policy-relevant (forward guidance, dissent rationale, inflation outlook)
5. Hardcode the 10 URLs into `src/boe_rag/scraper/speeches.py` as a constant list

Target speakers: Bailey, Lombardelli, Mann, Breeden, Dhingra, Greene, Pill, Taylor, Ramsden.

**Why 10 not 20:** Evaluation queries target MPC minutes and MPR/FSR content. Speeches are supporting material. 10 well-chosen speeches add corpus breadth without doubling ingestion/chunking work for content the eval barely tests.

**The 10 URLs will be finalised and hardcoded into the scraper before any scraping runs. No "figure it out later."**

---

## Scraping Implementation

### Shared extraction logic

All four document types use the same core HTML structure. The scraper base class handles:

```python
import unicodedata
from abc import ABC, abstractmethod


class BaseScraper(ABC):
    """Base class for all BoE document scrapers."""

    def scrape(self, html: str) -> str:
        """Extract clean text from a BoE publication page."""
        soup = BeautifulSoup(html, "lxml")

        # Subclasses extract page-level metadata BEFORE narrowing to content.
        # e.g. SpeechScraper extracts speaker from sidebar div.col3 h3 here.
        page_metadata = self._extract_page_metadata(soup)

        # Find the main content container
        content = self._find_content(soup)

        # STEP 1: Extract chart/table info BEFORE stripping containers.
        chart_markers = self._extract_chart_titles(content)  # MPR/FSR override
        table_markers = self._extract_tables(content)        # MPR/FSR override

        # STEP 2: Strip elements inside the content container
        for selector in ["div.img-block",           # Chart containers (base64 images)
                         "div.footnotes-container",  # Footnotes
                         "nav.nav-chapters",         # Sidebar TOC (JS-populated)
                         "div.pdf-form"]:            # "Convert to PDF" form
            for el in content.select(selector):
                el.decompose()

        # Strip Word paste artefacts
        for a in content.find_all("a", id=lambda x: x and x.startswith("_Hlk")):
            a.decompose()

        # STEP 3: Walk content tree (per-document-type logic)
        raw_text = self._walk_content_tree(content, chart_markers, table_markers)

        # STEP 4: Normalise unicode
        return _normalise_text(raw_text)

    def _find_content(self, soup: BeautifulSoup) -> Tag:
        """Locate the content container. MPC overrides this."""
        content = soup.select_one("div.page-content#content div#output")
        if not content:
            content = soup.select_one("div.col9")
        if not content:
            raise ScraperError(f"No content container found")
        return content

    def _extract_page_metadata(self, soup: BeautifulSoup) -> dict:
        """Override in subclasses that need metadata outside content container."""
        return {}

    def _extract_chart_titles(self, content: Tag) -> list[str]:
        """Override in MPR/FSR to extract h3.img-title text."""
        return []

    def _extract_tables(self, content: Tag) -> list[str]:
        """Override in MPR/FSR to extract tables via pandas.read_html()."""
        return []

    @abstractmethod
    def _walk_content_tree(self, content: Tag, charts: list, tables: list) -> str:
        """Subclass implements document-type-specific text extraction."""
        ...


def _normalise_text(text: str) -> str:
    """Normalise whitespace and unicode characters."""
    text = text.replace("\xa0", " ")           # &nbsp; → space
    text = text.replace("\u2019", "'")         # right single curly → straight
    text = text.replace("\u201c", '"')         # left double curly → straight
    text = text.replace("\u201d", '"')         # right double curly → straight
    text = text.replace("\u2013", "-")         # en-dash → hyphen
    text = text.replace("\u2014", " - ")       # em-dash → spaced hyphen
    text = unicodedata.normalize("NFKC", text) # canonical decomposition
    return text
```

**Key design decisions:**
- **Class hierarchy, not standalone functions.** `BaseScraper` defines the extraction pipeline. Subclasses (`MPCScraper`, `MPRScraper`, `FSRScraper`, `SpeechScraper`) override only the methods that differ.
- **Extraction order**: Step 1 (chart/table extraction) runs BEFORE Step 2 (container decomposition). MPC subclass returns empty lists (no charts/tables in minutes).
- **Page-level metadata first**: `_extract_page_metadata()` runs on the full soup BEFORE narrowing to the content container. The `SpeechScraper` uses this to extract the speaker name from sidebar `div.col3 h3`, which lives outside `div#output`.
- **Decompose targets are only elements INSIDE the content container.** `nav`, `footer`, `header` are ancestors/siblings excluded by the selector — they are never matched or stripped.
- **Unicode normalisation** prevents `\xa0` (non-breaking space) from poisoning embeddings and breaking exact-match retrieval.

### Per-document-type differences

| Aspect | MPC | MPR | FSR | Speech |
|--------|-----|-----|-----|--------|
| Content root | `div.col9 div.page-content` (both blocks) | `div#output section.page-section` | Same as MPR | `div#output section.page-section` |
| Paragraph numbering | Yes (`^\d+[:.]\s`) | No | No | No |
| Box detection | No | `div.box-highlight` | `div.box-highlight` | No |
| Chart handling | N/A | Strip `div.img-block`, keep `h3.img-title` text | Same as MPR | Rare |
| Table handling | N/A | Extract `<table>` as text | Same as MPR | N/A |
| Vote extraction | `<strong>Votes to...</strong>` | N/A | N/A | N/A |
| Speaker extraction | `<strong>Name:</strong>` pattern | N/A | N/A | Sidebar `div.col3 h3` |

This justifies separate scraper classes — the extraction logic genuinely diverges on paragraph numbering, box detection, vote parsing, and chart handling.

### Output format

Save as plain text with lightweight structural markers:

```
## Monetary Policy Summary, November 2025

At its meeting ending on 5 November 2025, the Monetary Policy Committee
voted by a majority of 5-4 to maintain Bank Rate at 4%.

## Minutes of the Monetary Policy Committee meeting ending on 5 November 2025

### The Committee's discussions

1: Before turning to its immediate policy decision, the Monetary Policy
Committee (MPC) discussed key economic developments...

2: The Committee's policy discussions covered: the extent to which
disinflation was continuing...

[...]

### The immediate policy decision

22: The Chair invited the Committee to vote on the proposition that:
- Bank Rate should be maintained at 4%.

Five members (Andrew Bailey, Megan Greene...) voted in favour.
Four members (Sarah Breeden...) voted against.

**Votes to maintain Bank Rate at 4%**
**Andrew Bailey:** [rationale text]
**Megan Greene:** [rationale text]

**Votes to reduce Bank Rate by 0.25 percentage points, to 3.75%**
**Sarah Breeden:** [rationale text]
```

For MPR/FSR, additionally preserve:
```
[BOX START: Box A - Developments in firms' costs and margins]
[Summary: Lead paragraph text from p.text-box-highlight]
Body text paragraphs...
[BOX END]

[CHART: Chart 1.1 - CPI inflation has been falling]

[TABLE: Table 3.A - Forecast summary]
(extracted via pandas.read_html() → DataFrame.to_string(), not hand-rolled parsing)
```

**Chart handling**: Before stripping `div.img-block`, extract the `h3.img-title` text and insert as a `[CHART: ...]` marker. The image data is discarded — this is an acknowledged limitation (see report Section 4: Failure Analysis).

**Table handling**: Use `pandas.read_html(str(table_element))` on the `<table>` HTML string to get a DataFrame, then `.to_string()` for plain text. This handles `rowspan`/`colspan` correctly. Only ~1 table per MPR, so this isn't a performance concern.

### Interface contract with the chunker

The output `.txt` format is the API between the scraper and the chunker (spec 03). The chunker parses these markers to detect section boundaries. **If you change a marker format here, the chunker breaks.**

| Marker | Meaning | Chunker uses it for |
|--------|---------|-------------------|
| `## Heading text` | H2 section/chapter boundary | Split point for chapters (MPR/FSR) and major sections (MPC) |
| `### Sub-heading text` | H3 sub-section boundary | Split point for sub-sections, "The immediate policy decision", etc. |
| `N: paragraph text` | Numbered MPC paragraph (N = integer) | Paragraph range tagging in metadata (`paragraph_range: "15-18"`) |
| `**Name:** text` | Individual MPC member statement | `section_category: individual_statement`, `speaker: Name` |
| `**Votes to ...**` | Vote grouping header | `section_category: voting` |
| `[BOX START: Box X - title]` | Start of MPR/FSR box analysis | `section_category: box_analysis`, kept as single chunk |
| `[BOX END]` | End of box analysis | Box boundary |
| `[CHART: Chart N.N - title]` | Chart placeholder (image discarded) | Preserved in chunk text for context |
| `[TABLE: Table N.N - title]` | Table with extracted text | Preserved in chunk text |

### Rate limiting and caching

```python
DELAY_SECONDS = 2.0
USER_AGENT = "BoE-RAG-Academic-Research/1.0 (University of Warwick MSc FinTech)"


def _url_to_cache_name(url: str) -> str:
    """Human-readable cache filename from URL slug.

    https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/november-2025
    → 'monetary-policy-summary-and-minutes_2025_november-2025.html'
    """
    parts = url.rstrip("/").split("/")
    # Take the last 3 path segments: document-type / year / slug
    return "_".join(parts[-3:]) + ".html"


def fetch_page(url: str, cache_dir: Path) -> str | None:
    """Fetch with caching, rate limiting, and graceful error handling.

    Returns HTML string on success, None on failure (404, timeout, etc.).
    """
    cache_path = cache_dir / _url_to_cache_name(url)
    if cache_path.exists():
        logger.debug("Cache hit: %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8")

    time.sleep(DELAY_SECONDS)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.warning("HTTP %s for %s — skipping", e.response.status_code, url)
        return None
    except (requests.ConnectionError, requests.Timeout):
        logger.warning("Connection/timeout failed for %s — retrying once", url)
        time.sleep(5)
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.error("Retry failed for %s — skipping", url)
            return None

    cache_path.write_text(resp.text, encoding="utf-8")
    return resp.text
```

**Key decisions:**
- **Returns `None` on failure** instead of raising — the scrape loop continues, records `status: missing` in manifest
- **Cache filenames are human-readable** — `monetary-policy-summary-and-minutes_2025_november-2025.html`, not an MD5 hash
- **Single retry on connection errors only** (not on 404 — a 404 won't become a 200 on retry)
- **30s timeout** to avoid hanging on slow Akamai responses

---

## Directory Layout

```
data/
├── html_cache/                         # Raw HTML responses (gitignored, large)
│   ├── monetary-policy-summary-and-minutes_2025_november-2025.html
│   ├── monetary-policy-report_2025_november-2025.html
│   └── ...
└── raw/                                # Processed text output (shipped in zip)
    ├── mpc_minutes/
    │   ├── mpc_2025_06.txt
    │   ├── mpc_2025_08.txt
    │   ├── mpc_2025_09.txt
    │   ├── mpc_2025_11.txt
    │   ├── mpc_2025_12.txt
    │   ├── mpc_2026_02.txt
    │   └── mpc_2026_03.txt
    ├── mpr/
    │   ├── mpr_2025_02.txt
    │   ├── mpr_2025_05.txt
    │   ├── mpr_2025_08.txt
    │   └── mpr_2025_11.txt
    ├── fsr/
    │   ├── fsr_2025_07.txt
    │   └── fsr_2025_12.txt
    ├── speeches/
    │   ├── speech_bailey_2025_09.txt
    │   ├── speech_lombardelli_2025_11.txt
    │   └── ...
    └── manifest.csv
```

`data/html_cache/` is gitignored (raw HTML files are large — MPRs are 3.4 MB each — and fully recreatable by re-running the scraper). `data/raw/` ships in the zip so the marker can inspect processed text.

The `fetch_page` function's `cache_dir` parameter points to `data/html_cache/`. The scrape loop's output goes to `data/raw/`.

## Manifest CSV

```csv
filename,document_type,date,title,source_url,word_count,status
mpc_2025_11.txt,MPC_minutes,2025-11,November 2025 MPC Minutes,https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2025/november-2025,4200,ok
speech_dhingra_2025_10.txt,speech,2025-10,Swati Dhingra on trade policy,https://www.bankofengland.co.uk/speech/2025/october/swati-dhingra-trade-policy,0,missing
```

The manifest serves two purposes:
1. **Validation**: Notebook 01 loads it and checks counts, lengths, completeness
2. **Downstream metadata**: The chunker reads `source_url` and `document_type` from the manifest rather than inferring from filenames

---

## Validation Checks

1. **Count check**: manifest has ~23 rows (7 MPC + 4 MPR + 2 FSR + 10 speeches). All with `status: ok`.
2. **Length check by type**:
   - MPC minutes: 3,000–6,000 words
   - MPR: 15,000–40,000 words (these are full reports, 3.4 MB HTML)
   - FSR: 15,000–40,000 words (4.8 MB HTML)
   - Speeches: 1,500–8,000 words
   - Flag anything outside these ranges
3. **Content check**: Print first 500 chars of each file. Verify document content, not boilerplate.
4. **Structure check**:
   - MPC files contain `##` headings AND numbered paragraphs (`\d+[:.]\s`)
   - MPR/FSR files contain `[BOX START` markers
   - MPR/FSR files contain `[CHART:` markers
5. **Domain check**:
   - MPC files contain "Bank Rate" and "voted"
   - MPR files contain "Box" and "Chart"
   - FSR files contain "financial stability" or "resilience"
6. **Encoding check**: No mojibake. Search for `â€™` or `Â£` (signs of encoding corruption).

---

## Acceptance Criteria

1. All target documents downloaded and saved to `data/raw/`
2. `manifest.csv` exists with correct metadata for every file, consumed by the chunker
3. All 6 validation checks pass
4. Scraper respects 2s delay between requests
5. Scraper is idempotent — re-running skips already-cached HTML (fetch level) AND already-existing `.txt` output files (processing level)
6. 404s and missing pages handled gracefully (logged, not crashed)
7. No PyMuPDF / PDF fallback needed — all content available as HTML
