from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.moneycontrol.com/stocksmarketsindia/"
MARKET_TERMS = {
    "sensex","nifty","nifty 50","bank nifty","banknifty","bse","nse","stocks",
    "stock","equity","market","markets","rbi","sebi","rupee","earnings","profit",
    "revenue","ipo","fii","dii","bond","yield","oil","crude","gdp","inflation",
    "interest rate","tariff","broker","target price","buy","sell","q1","q2","q3","q4",
}

@dataclass
class MoneyControlSnapshot:
    fetched_at: str
    headlines: list[dict]
    error: str = ""


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def is_moneycontrol_url(url: str) -> bool:
    parsed = urlparse(url)
    return not parsed.netloc or parsed.netloc.endswith("moneycontrol.com")


def market_relevance(title: str) -> float:
    low = title.lower()
    return sum(1.0 for term in MARKET_TERMS if term in low)


class MoneyControlScraper:
    def __init__(self, state_dir: str = "data_cache"):
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.75,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}),
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
        self.session.headers.update({
            "User-Agent": "RadheStockMarketPredictor/4.0 (market research)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        })
        self.state_file = Path(state_dir) / "moneycontrol_news.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot: MoneyControlSnapshot | None = None
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        try:
            self.snapshot = MoneyControlSnapshot(**json.loads(self.state_file.read_text(encoding="utf-8")))
        except Exception:
            self.snapshot = None

    def _save(self):
        if not self.snapshot:
            return
        try:
            self.state_file.write_text(json.dumps(self.snapshot.__dict__, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def scrape(self) -> MoneyControlSnapshot:
        try:
            r = self.session.get(BASE, timeout=25)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            rows = []
            seen = set()
            for a in soup.select('a[href]'):
                title = clean(a.get_text(" ", strip=True))
                href = a.get("href") or ""
                url = urljoin(BASE, href)
                if not title or len(title) < 25 or len(title) > 300:
                    continue
                if not is_moneycontrol_url(url):
                    continue
                if market_relevance(title) <= 0:
                    continue
                key = (title.lower(), url.split("?")[0])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "source": "Moneycontrol",
                    "title": title,
                    "url": url,
                    "published_at": "",
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "relevance": market_relevance(title),
                })
                if len(rows) >= 50:
                    break
            rows.sort(key=lambda x: (-float(x.get("relevance", 0)), x.get("title", "")))
            snap = MoneyControlSnapshot(datetime.now(timezone.utc).isoformat(), rows, "")
        except Exception as exc:
            previous = self.snapshot.headlines if self.snapshot else []
            snap = MoneyControlSnapshot(
                datetime.now(timezone.utc).isoformat(), previous,
                f"Moneycontrol refresh failed: {type(exc).__name__}: {exc}"
            )
        with self._lock:
            self.snapshot = snap
            self._save()
        return snap

    def get(self) -> MoneyControlSnapshot | None:
        with self._lock:
            return self.snapshot


class MoneyControlBackground:
    def __init__(self, refresh_minutes: int = 30, state_dir: str = "data_cache"):
        self.refresh_seconds = max(900, int(refresh_minutes) * 60)
        self.scraper = MoneyControlScraper(state_dir)
        self._started = False
        self.thread = None

    def start(self):
        if self._started:
            return
        self._started = True
        self.thread = threading.Thread(target=self._loop, name="moneycontrol-refresh", daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            self.scraper.scrape()
        except Exception:
            pass
        while True:
            time.sleep(self.refresh_seconds)
            try:
                self.scraper.scrape()
            except Exception:
                pass

    def get(self) -> MoneyControlSnapshot | None:
        return self.scraper.get()
