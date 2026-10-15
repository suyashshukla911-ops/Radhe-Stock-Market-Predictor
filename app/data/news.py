from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import requests
from pathlib import Path


from app.config import MARKET_CLOSE_MINUTES, MARKET_OPEN_MINUTES, IST, settings
from zoneinfo import ZoneInfo

POS = {"beat","beats","beating","growth","profit","profits","surge","surges","rally","rallies","rise","rises","rising","gain","gains","strong","upgrade","upgraded","outperform","record","boost","boosted","approval","approved","contract","deal","partnership","launch","launched","expansion","demand","recovery","acquire","acquisition","buyback","dividend","order","orders","guidance","optimistic","bullish","breakout"}
NEG = {"miss","missed","loss","losses","fall","falls","drop","drops","decline","declined","plunge","plunged","crash","down","weak","downgrade","downgraded","underperform","lawsuit","investigation","fraud","scandal","default","debt","warning","penalty","fine","layoff","layoffs","delay","delayed","recall","probe","resign","resignation","cut","cuts","regulatory","ban","strike","fire","slump","concern","bearish","selloff","breakdown"}

EVENT_PATTERNS = {
    "earnings": r"\b(results?|profit|revenue|earnings?|q[1-4])\b",
    "guidance": r"\b(guidance|outlook|forecast|expects?|target)\b",
    "m&a": r"\b(acqui(?:re|red|res|sition)|merger|takeover|stake)\b",
    "contract": r"\b(order|contract|deal|won|award)\b",
    "regulatory": r"\b(sebi|rbi|regulator|penalty|ban|approval)\b",
    "litigation": r"\b(lawsuit|probe|investigation|fraud|scandal)\b",
    "management": r"\b(ceo|md|chairman|resign|resignation|appoints?)\b",
    "capital": r"\b(dividend|buyback|fund[- ]?raise|debt|capital)\b",
    "product": r"\b(launch|product|new model|platform)\b",
    "operations": r"\b(plant|shutdown|production|capacity|strike)\b",
    "macro": r"\b(inflation|rates?|oil prices?|rupee|tariff|gdp|economy)\b",
    "analyst": r"\b(upgrade|downgrade|broker|price target|analyst)\b",
}
EVENT_WEIGHTS = {"earnings":1.0,"guidance":0.95,"m&a":0.9,"contract":0.75,"regulatory":0.9,"litigation":0.9,"management":0.65,"capital":0.75,"product":0.45,"operations":0.45,"macro":0.55,"analyst":0.35,"other":0.2}
SOURCE_WEIGHTS = {"reuters":1.0,"economic times":0.95,"moneycontrol":0.9,"business standard":0.9,"livemint":0.9,"financial express":0.85,"the hindu":0.85,"hindustan times":0.8}


def lexical_sentiment(text: str) -> float:
    words = re.findall(r"[a-zA-Z][a-zA-Z-]*", text.lower())
    p = sum(w in POS for w in words)
    n = sum(w in NEG for w in words)
    total = p + n
    return 0.0 if total == 0 else round((p-n)/total, 3)


def classify_event(text: str) -> tuple[str, float]:
    low = text.lower()
    for label, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, low):
            return label, EVENT_WEIGHTS[label]
    return "other", EVENT_WEIGHTS["other"]


def source_weight(publisher: str) -> float:
    low = (publisher or "").lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name in low:
            return weight
    return 0.65


@dataclass
class NewsQuota:
    day_key: str
    used: int = 0
    last_request_ts: float = 0.0
    daily_limit: int = settings.news_daily_quota

    def reset_if_needed(self):
        key = datetime.now(ZoneInfo(IST)).strftime("%Y-%m-%d")
        if key != self.day_key:
            self.day_key = key
            self.used = 0
            self.last_request_ts = 0.0

    @property
    def remaining(self) -> int:
        self.reset_if_needed()
        return max(0, self.daily_limit - self.used)

    def wait_for_slot(self):
        self.reset_if_needed()
        if self.used >= self.daily_limit:
            return False
        now = time.monotonic()
        wait = settings.news_min_interval_seconds - (now - self.last_request_ts)
        if wait > 0:
            time.sleep(wait)
        return True

    def mark(self):
        self.reset_if_needed()
        self.used += 1
        self.last_request_ts = time.monotonic()


class NewsService:
    BASE = "https://api.freenewsapi.io/v1/news"

    def __init__(self, api_key: str = "", max_records: int = 12, state_dir: str = "data_cache"):
        self.api_key = api_key.strip()
        self.max_records = max_records
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RadheStockMarketPredictor/2.0"})
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._lock = threading.RLock()
        self.state_file = Path(state_dir) / "news_quota.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.quota = self._load_quota()
        self.last_api_error = ""

    def _load_quota(self) -> NewsQuota:
        today = datetime.now(ZoneInfo(IST)).strftime("%Y-%m-%d")
        try:
            data = __import__("json").loads(self.state_file.read_text(encoding="utf-8"))
            if data.get("day_key") == today:
                return NewsQuota(day_key=today, used=int(data.get("used", 0)), last_request_ts=0.0)
        except Exception:
            pass
        return NewsQuota(day_key=today)

    def _save_quota(self) -> None:
        try:
            payload = {"day_key": self.quota.day_key, "used": self.quota.used}
            self.state_file.write_text(__import__("json").dumps(payload), encoding="utf-8")
        except Exception:
            pass

    def _market_is_open(self) -> bool:
        now = datetime.now(ZoneInfo(IST))
        if now.weekday() >= 5:
            return False
        minute = now.hour * 60 + now.minute
        return MARKET_OPEN_MINUTES <= minute < MARKET_CLOSE_MINUTES

    def _fetch(self, term: str, refresh_seconds: int = 30) -> list[dict]:
        if not self.api_key:
            return []
        cache_key = term.upper()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and time.time() - cached[0] < refresh_seconds:
                return cached[1]
            if not self.quota.wait_for_slot():
                return cached[1] if cached else []
            params = {
                "language": "en",
                "country": "IN",
                "order_by": "recent",
                "in_title": term[:80],
            }
            try:
                response = self.session.get(self.BASE, params=params, headers={"x-api-key": self.api_key}, timeout=15)
                self.quota.mark()
                self._save_quota()
            except requests.RequestException as exc:
                self.quota.mark()
                self._save_quota()
                self.last_api_error = str(exc)
                return cached[1] if cached else []
            if response.status_code >= 400:
                self.last_api_error = f"FreeNewsApi HTTP {response.status_code}"
                return cached[1] if cached else []
            try:
                data = response.json().get("data", []) or []
            except Exception as exc:
                self.last_api_error = str(exc)
                return cached[1] if cached else []
            rows = []
            now = datetime.now(timezone.utc)
            for item in data[: self.max_records]:
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                published = item.get("published_at") or ""
                try:
                    ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    age_hours = max(0.0, (now - ts).total_seconds() / 3600)
                except Exception:
                    age_hours = 24.0
                tone = lexical_sentiment(title)
                event, event_weight = classify_event(title)
                recency = math.exp(-age_hours / 18.0)
                impact = max(-1.0, min(1.0, tone * event_weight * source_weight(item.get("publisher", "")) * (0.5 + 0.5 * recency)))
                rows.append({
                    "uuid": item.get("uuid", ""),
                    "title": title,
                    "url": item.get("url") or item.get("original_url") or "",
                    "publisher": item.get("publisher", "unknown"),
                    "published_at": published,
                    "sentiment": round(tone, 3),
                    "event": event,
                    "impact": round(impact, 4),
                    "recency": round(recency, 4),
                })
            rows.sort(key=lambda x: abs(x["impact"]), reverse=True)
            self._cache[cache_key] = (time.time(), rows)
            return rows

    def search(self, company: str, symbol: str, timespan: str = "24h") -> list[dict]:
        term = company.strip() if company.strip() else symbol.strip()
        if symbol.upper() in {"SENSEX", "NIFTY 50", "NIFTY BANK", "INDIA VIX"}:
            term = symbol
        term = re.sub(r"\s+LIMITED$", "", term, flags=re.I).strip()
        return self._fetch(term, refresh_seconds=max(5, int(settings.news_window_seconds)))

    @staticmethod
    def aggregate(articles: Iterable[dict]) -> dict:
        articles = list(articles)
        if not articles:
            return {"headline_count":0,"sentiment":0.0,"impact":0.0,"positive":0,"negative":0,"neutral":0,"top_event":"none"}
        counts = {"positive":0,"negative":0,"neutral":0}
        weighted = []
        event_scores = {}
        for a in articles:
            s = float(a.get("sentiment",0))
            label = "positive" if s > 0.15 else "negative" if s < -0.15 else "neutral"
            counts[label] += 1
            weighted.append(float(a.get("impact",0)))
            event_scores[a.get("event","other")] = event_scores.get(a.get("event","other"),0)+abs(float(a.get("impact",0)))
        return {
            "headline_count": len(articles),
            "sentiment": round(sum(float(a.get("sentiment",0)) for a in articles)/len(articles),4),
            "impact": round(max(-1.0,min(1.0,sum(weighted)/len(weighted))),4),
            **counts,
            "top_event": max(event_scores, key=event_scores.get) if event_scores else "none",
        }

    def diagnostics(self) -> dict:
        return {
            "provider": settings.news_provider,
            "configured": bool(self.api_key),
            "used_today": self.quota.used,
            "remaining_today": self.quota.remaining,
            "daily_limit": self.quota.daily_limit,
            "last_error": self.last_api_error,
        }
