"""
Times of India -> structured Excel news scraper

Run this script to scrape TOI category pages and generate an Excel news export.
The output directory defaults to 'toi_output/toi_market_news.xlsx' (or configured via TOI_OUTPUT_DIR).
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Dict, List
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------
BASE = "https://timesofindia.indiatimes.com"

# EXACT OUTPUT LOCATION REQUESTED
OUTPUT_DIR = Path(os.getenv("TOI_OUTPUT_DIR", "toi_output"))
OUTPUT_FILE = OUTPUT_DIR / "toi_market_news.xlsx"

REQUEST_DELAY_SECONDS = 1.25
TIMEOUT_SECONDS = 20
MAX_HEADLINES_PER_SECTION = 250

SECTIONS: Dict[str, str] = {
    # Business / markets
    "Business": "/business",
    "India Business": "/business/india-business",
    "International Business": "/business/international-business",
    "Economy": "/business/economy",
    "Markets": "/business/markets",

    # Technology / consumer technology
    "Technology": "/technology",
    "Tech News": "/technology/tech-news",
    "Gadgets": "/gadgets-news",

    # Education
    "Education": "/education",

    # Global / geopolitical information
    "World": "/world",
    "US": "/world/us",
    "Pakistan": "/world/pakistan",
    "South Asia": "/world/south-asia",
    "UK": "/world/uk",
    "Europe": "/world/europe",
    "China": "/world/china",
    "Middle East": "/world/middle-east",
    "Rest of World": "/world/rest-of-world",
}

USER_AGENT = (
    "TOI-News-Scraper/1.0 "
    "(personal research project; respects robots.txt)"
)

NEWS_HEADERS = [
    "scraped_at_utc",
    "source",
    "section",
    "headline_clean",
    "url",
    "category_page",
    "model_text",
]

# ---------------------------------------------------------------------------
# 4. HTTP / parsing helpers
# ---------------------------------------------------------------------------
def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
            "Cache-Control": "no-cache",
        }
    )
    return session


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalise_url(href: str) -> str:
    full = urljoin(BASE, href.strip())
    parsed = urlparse(full)
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned)


def allowed_by_robots(robots: RobotFileParser, url: str) -> bool:
    return robots.can_fetch(USER_AGENT, url)


def fetch_html(session: requests.Session, robots: RobotFileParser, url: str) -> str:
    if not allowed_by_robots(robots, url):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def looks_like_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE).netloc:
        return False

    path = parsed.path.lower()
    if "/articleshow/" not in path:
        return False

    blocked_fragments = (
        "/articleshowbriefs",
        "/articleshowajax",
        "/articleshow_new",
        "/articleshowprint",
        "/amp/",
    )
    return not any(fragment in path for fragment in blocked_fragments)


def headline_from_anchor(anchor) -> str:
    text = clean_text(anchor.get_text(" ", strip=True))
    if not text:
        text = clean_text(anchor.get("title", ""))
    if not text:
        text = clean_text(anchor.get("aria-label", ""))
    return text


def is_reasonable_headline(text: str) -> bool:
    if not text:
        return False
    if len(text) < 12 or len(text) > 300:
        return False

    lowered = text.lower()
    blocked_exact = {
        "read more",
        "more",
        "click here",
        "view all",
        "latest stories",
        "related stories",
        "photos",
        "videos",
    }
    if lowered in blocked_exact:
        return False

    return bool(re.search(r"[A-Za-z]", text))


def extract_articles(html: str, section: str, page_url: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[dict] = []
    seen_urls = set()

    # Keep headline extraction exactly based on TOI article URLs so menu labels
    # are not accidentally treated as news.
    for anchor in soup.select('a[href*="/articleshow/"]'):
        href = anchor.get("href")
        if not href:
            continue

        url = normalise_url(href)
        if not looks_like_article_url(url):
            continue
        if url in seen_urls:
            continue

        headline = headline_from_anchor(anchor)
        if not is_reasonable_headline(headline):
            continue

        seen_urls.add(url)
        rows.append(
            {
                "source": "Times of India",
                "section": section,
                "headline_clean": headline,
                "url": url,
                "category_page": page_url,
            }
        )

        if len(rows) >= MAX_HEADLINES_PER_SECTION:
            break

    return rows

# ---------------------------------------------------------------------------
# 5. Excel output
# ---------------------------------------------------------------------------
def set_column_widths(ws):
    for column_cells in ws.columns:
        max_length = 0
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, len(value))
        letter = column_cells[0].column_letter
        ws.column_dimensions[letter].width = min(max_length + 2, 70)


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    set_column_widths(ws)

    # Header formatting
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top")

    # Readability for long text columns.
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def make_output(records: List[dict], section_status: List[dict], run_started: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # De-duplicate article URLs across overlapping TOI pages.
    unique_records = []
    seen_urls = set()

    for record in records:
        url = record["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        unique_records.append(
            {
                "scraped_at_utc": run_started,
                "source": record["source"],
                "section": record["section"],
                "headline_clean": record["headline_clean"],
                "url": url,
                "category_page": record["category_page"],
                "model_text": f"{record['section']} | {record['headline_clean']}",
            }
        )

    unique_records.sort(
        key=lambda x: (x["section"].lower(), x["headline_clean"].lower())
    )

    wb = Workbook()
    default_ws = wb.active
    wb.remove(default_ws)

    # ----------------------------- news sheet -----------------------------
    ws_news = wb.create_sheet("news")
    ws_news.append(NEWS_HEADERS)

    for record in unique_records:
        ws_news.append([record[column] for column in NEWS_HEADERS])

    # --------------------------- sections sheet ---------------------------
    ws_sections = wb.create_sheet("sections")
    section_headers = ["section", "url", "status", "headlines", "seconds"]
    ws_sections.append(section_headers)
    for item in section_status:
        ws_sections.append(
            [
                item.get("section", ""),
                item.get("url", ""),
                item.get("status", ""),
                item.get("headlines", 0),
                item.get("seconds", 0),
            ]
        )

    # ------------------------- run_metadata sheet ------------------------
    ws_meta = wb.create_sheet("run_metadata")
    ws_meta.append(["field", "value"])
    metadata = [
        ("run_started_utc", run_started),
        ("output_file", str(OUTPUT_FILE)),
        ("output_directory", str(OUTPUT_DIR)),
        ("sections_configured", len(SECTIONS)),
        ("unique_headlines", len(unique_records)),
        ("request_delay_seconds", REQUEST_DELAY_SECONDS),
        ("max_headlines_per_section", MAX_HEADLINES_PER_SECTION),
        ("purpose", "Fresh news input for a market/stock-analysis pipeline"),
        ("content_scope", "Headline metadata only; no full article bodies"),
    ]
    for field, value in metadata:
        ws_meta.append([field, value])

    # ----------------------------- README sheet --------------------------
    ws_readme = wb.create_sheet("README")
    ws_readme.append(["field", "description"])
    readme_rows = [
        ("headline_clean", "Headline text captured from a TOI category page."),
        ("section", "TOI section that produced the headline."),
        ("url", "TOI article URL."),
        ("scraped_at_utc", "UTC time of the scraping run."),
        ("model_text", "Combined NLP input: section | headline."),
        ("run behavior", "Run this scraper immediately before the prediction pipeline so the model receives fresh news."),
        ("output location", r"D:\toi_output\toi_market_news.xlsx"),
    ]
    for row in readme_rows:
        ws_readme.append(list(row))

    # Formatting
    for ws in wb.worksheets:
        format_sheet(ws)

    ws_news.column_dimensions["D"].width = 80   # headline
    ws_news.column_dimensions["E"].width = 80   # url
    ws_news.column_dimensions["F"].width = 60   # category
    ws_news.column_dimensions["G"].width = 95   # model_text

    ws_readme.column_dimensions["A"].width = 25
    ws_readme.column_dimensions["B"].width = 110
    ws_meta.column_dimensions["A"].width = 30
    ws_meta.column_dimensions["B"].width = 100
    ws_sections.column_dimensions["A"].width = 25
    ws_sections.column_dimensions["B"].width = 65
    ws_sections.column_dimensions["C"].width = 45

    wb.save(OUTPUT_FILE)

    # Re-open once to make sure the generated workbook is readable.
    check_wb = load_workbook(OUTPUT_FILE, read_only=True)
    check_wb.close()

# ---------------------------------------------------------------------------
# 6. Main scraper
# ---------------------------------------------------------------------------
def main() -> None:
    run_started = datetime.now(timezone.utc).isoformat()
    session = build_session()

    robots = RobotFileParser()
    robots.set_url(f"{BASE}/robots.txt")
    robots.read()

    all_records: List[dict] = []
    section_status: List[dict] = []

    print("=" * 72)
    print("TIMES OF INDIA NEWS SCRAPER")
    print("=" * 72)
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Output file  : {OUTPUT_FILE}")
    print()

    for index, (section, path) in enumerate(SECTIONS.items(), start=1):
        url = urljoin(BASE, path)
        started = time.time()
        print(f"[{index}/{len(SECTIONS)}] Scraping {section} ...", end=" ")

        try:
            html = fetch_html(session, robots, url)
            records = extract_articles(html, section, url)
            all_records.extend(records)

            section_status.append(
                {
                    "section": section,
                    "url": url,
                    "status": "ok",
                    "headlines": len(records),
                    "seconds": round(time.time() - started, 2),
                }
            )
            print(f"OK ({len(records)} headlines)")

        except requests.HTTPError as exc:
            section_status.append(
                {
                    "section": section,
                    "url": url,
                    "status": f"http_error:{exc}",
                    "headlines": 0,
                    "seconds": round(time.time() - started, 2),
                }
            )
            print(f"FAILED - HTTP error: {exc}")

        except Exception as exc:
            section_status.append(
                {
                    "section": section,
                    "url": url,
                    "status": f"error:{type(exc).__name__}:{exc}",
                    "headlines": 0,
                    "seconds": round(time.time() - started, 2),
                }
            )
            print(f"FAILED - {type(exc).__name__}: {exc}")

        if index < len(SECTIONS):
            time.sleep(REQUEST_DELAY_SECONDS)

    make_output(all_records, section_status, run_started)

    unique_urls = {record["url"] for record in all_records}
    ok_count = sum(1 for item in section_status if item["status"] == "ok")

    print()
    print("=" * 72)
    print("TOI SCRAPE COMPLETED")
    print("=" * 72)
    print(f"Sections attempted : {len(SECTIONS)}")
    print(f"Sections successful: {ok_count}")
    print(f"Unique headlines   : {len(unique_urls)}")
    print(f"Excel output       : {OUTPUT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
