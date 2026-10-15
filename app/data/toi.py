from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import IST
from zoneinfo import ZoneInfo

BASE = "https://timesofindia.indiatimes.com"
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 20
MAX_HEADLINES_PER_SECTION = 150

SECTIONS: Dict[str, str] = {
    "Business": "/business",
    "India Business": "/business/india-business",
    "International Business": "/business/international-business",
    "Economy": "/business/economy",
    "Markets": "/business/markets",
    "Technology": "/technology",
    "Tech News": "/technology/tech-news",
    "Gadgets": "/gadgets-news",
    "Education": "/education",
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

USER_AGENT = "TOI-News-Scraper/1.0 (personal market-research project; respects robots.txt)"

@dataclass
class TOISnapshot:
    fetched_at: str
    headlines: list[dict]
    sections_ok: int
    sections_total: int
    error: str = ""

def _clean(v: str) -> str:
    return re.sub(r"\s+", " ", v or "").strip()

def _normalise(url: str) -> str:
    full = urljoin(BASE, url.strip())
    parsed = urlparse(full)
    return urlunparse(parsed._replace(query="", fragment=""))

def _is_article(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE).netloc:
        return False
    path = parsed.path.lower()
    if "/articleshow/" not in path:
        return False
    return not any(x in path for x in ("/articleshowbriefs", "/articleshowajax", "/articleshow_new", "/articleshowprint", "/amp/"))

def _reasonable(headline: str) -> bool:
    if not headline or len(headline) < 12 or len(headline) > 300:
        return False
    if _clean(headline).lower() in {"read more","more","click here","view all","latest stories","related stories","photos","videos"}:
        return False
    return bool(re.search(r"[A-Za-z]", headline))

class TOIScraper:
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset({"GET"}), respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
            "Cache-Control": "no-cache",
        })
        self.robots = RobotFileParser()
        self.robots.set_url(f"{BASE}/robots.txt")
        self._robots_loaded = False

    def _robots_ok(self, url: str) -> bool:
        if not self._robots_loaded:
            try:
                self.robots.read()
                self._robots_loaded = True
            except Exception:
                # Fail closed if robots.txt cannot be read.
                return False
        return self.robots.can_fetch(USER_AGENT, url)

    def _fetch(self, url: str) -> str:
        if not self._robots_ok(url):
            raise PermissionError("TOI robots.txt denied or could not be read.")
        r = self.session.get(url, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.text

    def _extract(self, html: str, section: str, page_url: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        rows, seen = [], set()
        for a in soup.select('a[href*="/articleshow/"]'):
            href = a.get("href")
            if not href: continue
            url = _normalise(href)
            if not _is_article(url) or url in seen: continue
            headline = _clean(a.get_text(" ", strip=True) or a.get("title", "") or a.get("aria-label", ""))
            if not _reasonable(headline): continue
            seen.add(url)
            rows.append({
                "source": "Times of India",
                "section": section,
                "title": headline,
                "url": url,
                "published_at": "",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            })
            if len(rows) >= MAX_HEADLINES_PER_SECTION: break
        return rows

    def scrape_all(self) -> TOISnapshot:
        started = datetime.now(timezone.utc).isoformat()
        all_rows, ok, error = [], 0, ""
        for idx, (section, path) in enumerate(SECTIONS.items(), start=1):
            url = urljoin(BASE, path)
            try:
                html = self._fetch(url)
                rows = self._extract(html, section, url)
                all_rows.extend(rows)
                ok += 1
            except Exception as exc:
                error = f"{section}: {type(exc).__name__}: {exc}"
            if idx < len(SECTIONS):
                time.sleep(REQUEST_DELAY_SECONDS)
        seen = set()
        unique = []
        for row in all_rows:
            if row["url"] in seen: continue
            seen.add(row["url"])
            unique.append(row)
        return TOISnapshot(started, unique, ok, len(SECTIONS), error)

class TOIBackground:
    def __init__(self, refresh_minutes: int = 10, state_dir: str = "data_cache"):
        self.refresh_seconds = max(300, refresh_minutes*60)
        self.scraper = TOIScraper()
        self.snapshot: TOISnapshot | None = None
        self._lock = threading.RLock()
        self._started = False
        self._thread = None
        self.state_file = Path(state_dir) / "toi_news.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_disk()

    def _load_disk(self):
        try:
            import json
            obj = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.snapshot = TOISnapshot(**obj)
        except Exception:
            self.snapshot = None

    def _save_disk(self):
        try:
            import json
            self.state_file.write_text(json.dumps(self.snapshot.__dict__, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def start(self):
        if self._started: return
        self._started = True
        self._thread = threading.Thread(target=self._loop, name="toi-refresh", daemon=True)
        self._thread.start()

    def refresh(self):
        snap = self.scraper.scrape_all()
        with self._lock:
            self.snapshot = snap
            self._save_disk()
        return snap

    def _loop(self):
        # Immediate first attempt, then every refresh interval.
        try: self.refresh()
        except Exception: pass
        while True:
            time.sleep(self.refresh_seconds)
            try: self.refresh()
            except Exception: pass

    def get(self) -> TOISnapshot | None:
        with self._lock:
            return self.snapshot
