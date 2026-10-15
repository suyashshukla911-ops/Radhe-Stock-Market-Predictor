from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

from app.config import IST, settings
from app.data.toi import TOIBackground

POS = {"beat","beats","growth","profit","profits","surge","rally","rise","rises","rising","gain","gains","strong","upgrade","upgraded","outperform","record","boost","approval","approved","contract","deal","partnership","launch","expansion","demand","recovery","acquire","acquisition","buyback","dividend","order","orders","guidance","optimistic","bullish","breakout","upside"}
NEG = {"miss","missed","loss","losses","fall","falls","drop","drops","decline","declined","plunge","plunged","crash","down","weak","downgrade","downgraded","underperform","lawsuit","investigation","fraud","scandal","default","debt","warning","penalty","fine","layoff","delay","recall","probe","resign","resignation","cut","cuts","regulatory","ban","strike","slump","concern","bearish","selloff","breakdown","downside"}
MARKET_TERMS = {
    "sensex","nifty","nifty 50","nifty bank","bank nifty","bse","nse","sebi","rbi","rupee","inflation",
    "interest rate","repo rate","oil","crude","tariff","gdp","india stock market","share market","stocks",
    "equity","earnings","markets","market","ipo","fii","dii","bond","yield","budget",
}
EVENT_PATTERNS = {
    "earnings": r"\b(results?|profit|revenue|earnings?|q[1-4])\b",
    "guidance": r"\b(guidance|outlook|forecast|expects?|target)\b",
    "m&a": r"\b(acqui(?:re|red|res|sition)|merger|takeover|stake)\b",
    "contract": r"\b(order|contract|deal|won|award)\b",
    "regulatory": r"\b(sebi|rbi|regulator|penalty|ban|approval)\b",
    "litigation": r"\b(lawsuit|probe|investigation|fraud|scandal)\b",
    "macro": r"\b(inflation|rates?|oil prices?|rupee|tariff|gdp|economy)\b",
    "analyst": r"\b(upgrade|downgrade|broker|price target|analyst)\b",
}
EVENT_WEIGHTS = {"earnings":1.0,"guidance":0.95,"m&a":0.9,"contract":0.75,"regulatory":0.9,"litigation":0.9,"macro":0.65,"analyst":0.4,"other":0.2}
SOURCE_WEIGHTS = {"reuters":1.0,"economic times":0.95,"moneycontrol":0.9,"business standard":0.9,"livemint":0.9,"financial express":0.85,"the hindu":0.85,"hindustan times":0.8,"times of india":0.78}

def lexical_sentiment(text: str) -> float:
    words = re.findall(r"[a-zA-Z][a-zA-Z-]*", text.lower())
    p = sum(w in POS for w in words)
    n = sum(w in NEG for w in words)
    if p+n == 0: return 0.0
    return round((p-n)/(p+n), 3)

def classify_event(text: str):
    low = text.lower()
    for label, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, low):
            return label, EVENT_WEIGHTS[label]
    return "other", EVENT_WEIGHTS["other"]

def source_weight(publisher: str) -> float:
    low = (publisher or "").lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name in low: return weight
    return 0.65

DISPLAY_SECTIONS = {"business","india business","international business","economy","markets","technology","tech news"}

def relevance_score(title: str, symbol: str, company: str, section: str = "") -> float:
    low = title.lower()
    sec = (section or "").lower().strip()
    score = 0.0
    terms = {symbol.lower(), company.lower(), company.lower().replace(" limited", "").strip()}
    for term in terms:
        if term and len(term) > 2 and term in low:
            score += 3.0
    if any(term in low for term in MARKET_TERMS):
        score += 1.7
    if sec in DISPLAY_SECTIONS:
        score += 2.5
    elif sec.startswith("world") or sec in {"us","pakistan","south asia","uk","europe","china","middle east","rest of world"}:
        score -= 2.5
    return score

def display_rank_score(row: dict, symbol: str, company: str) -> float:
    title = row.get("title", "")
    section = row.get("section", "")
    score = relevance_score(title, symbol, company, section)
    published = row.get("published_at") or row.get("scraped_at") or ""
    try:
        ts = datetime.fromisoformat(published.replace("Z", "+00:00"))
        age_h = max(0.0, (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 3600)
        freshness = math.exp(-age_h / 18.0)
    except Exception:
        freshness = 0.35
    return score + freshness

@dataclass
class NewsQuota:
    day_key: str
    used: int = 0
    last_request_ts: float = 0.0
    daily_limit: int = settings.news_daily_quota

    def reset_if_needed(self):
        key = datetime.now(ZoneInfo(IST)).strftime("%Y-%m-%d")
        if key != self.day_key:
            self.day_key, self.used, self.last_request_ts = key, 0, 0.0

    @property
    def remaining(self):
        self.reset_if_needed()
        return max(0, self.daily_limit-self.used)

    def wait_for_slot(self):
        self.reset_if_needed()
        if self.used >= self.daily_limit: return False
        wait = settings.news_min_interval_seconds - (time.monotonic()-self.last_request_ts)
        if wait > 0: time.sleep(wait)
        return True

    def mark(self):
        self.reset_if_needed()
        self.used += 1
        self.last_request_ts = time.monotonic()

class NewsService:
    BASE = "https://api.freenewsapi.io/v1/news"
    X_BASE = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, api_key: str = "", max_records: int = 20, state_dir: str = "data_cache"):
        self.api_key = api_key.strip()
        self.max_records = max_records
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset({"GET"}), respect_retry_after_header=True)
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
        self.session.headers.update({"User-Agent":"RadheStockMarketPredictor/3.0"})
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float,list[dict]]] = {}
        self._toi = TOIBackground(settings.toi_refresh_minutes, state_dir=state_dir)
        self._x_cache: tuple[float,list[dict]] | None = None
        self.quota = NewsQuota(datetime.now(ZoneInfo(IST)).strftime("%Y-%m-%d"))
        self.last_api_error = ""
        self.last_x_error = ""
        self.thread = None
        self.started = False

    def start(self):
        if self.started: return
        self.started = True
        self._toi.start()
        self.thread = threading.Thread(target=self._loop, name="news-refresh", daemon=True)
        self.thread.start()

    def _loop(self):
        # FreeNews/X refresh on the slow, deliberate cadence. TOI has its own
        # 10-minute background worker.
        self._refresh_global()
        while True:
            time.sleep(max(60, settings.news_refresh_minutes*60))
            try: self._refresh_global()
            except Exception as exc: self.last_api_error = str(exc)

    def _load_freenews(self, term: str) -> list[dict]:
        if not self.api_key or not self.quota.wait_for_slot():
            return []
        try:
            response = self.session.get(
                self.BASE,
                params={"language":"en","country":"IN","order_by":"recent","in_title":term[:80]},
                headers={"x-api-key":self.api_key},
                timeout=15,
            )
            self.quota.mark()
            if response.status_code >= 400:
                self.last_api_error = f"FreeNewsApi HTTP {response.status_code}"
                return []
            data = response.json().get("data", []) or []
        except Exception as exc:
            self.quota.mark()
            self.last_api_error = str(exc)
            return []

        now = datetime.now(timezone.utc)
        rows=[]
        for item in data[:self.max_records]:
            title = (item.get("title") or "").strip()
            if not title: continue
            published = item.get("published_at") or ""
            try:
                ts = datetime.fromisoformat(published.replace("Z","+00:00"))
                age_h = max(0.0,(now-ts).total_seconds()/3600)
            except Exception:
                age_h = 24.0
            tone = lexical_sentiment(title)
            event, ew = classify_event(title)
            impact = float(max(-1,min(1,tone*ew*source_weight(item.get("publisher",""))*(0.5+0.5*math.exp(-age_h/18)))))
            rows.append({
                "source":"FreeNewsApi","title":title,"url":item.get("url") or item.get("original_url") or "",
                "publisher":item.get("publisher","unknown"),"published_at":published,
                "sentiment":round(tone,3),"event":event,"impact":round(impact,4),
                "relevance":0.0,
            })
        return rows

    def _load_x(self) -> list[dict]:
        if not settings.x_bearer_token:
            return []
        try:
            r = self.session.get(
                self.X_BASE,
                params={"query":"(NIFTY OR SENSEX OR NSE OR BSE OR \"Indian stock market\") lang:en -is:retweet",
                        "max_results":100, "tweet.fields":"created_at,public_metrics,author_id"},
                headers={"Authorization":f"Bearer {settings.x_bearer_token}"},
                timeout=20,
            )
            if r.status_code >= 400:
                self.last_x_error = f"X API HTTP {r.status_code}"
                return []
            rows=[]
            for item in (r.json().get("data") or [])[:100]:
                title = (item.get("text") or "").replace("\n"," ").strip()
                tone=lexical_sentiment(title); event,ew=classify_event(title)
                rows.append({
                    "source":"X","title":title,"url":f"https://x.com/i/web/status/{item.get('id','')}",
                    "publisher":"X","published_at":item.get("created_at",""),
                    "sentiment":tone,"event":event,"impact":round(tone*ew*0.45,4),"relevance":0.2,
                })
            self._x_cache=(time.time(),rows)
            return rows
        except Exception as exc:
            self.last_x_error=str(exc)
            return []

    def _refresh_global(self):
        global_news = self._load_freenews("India stock market Nifty Sensex")
        with self._lock:
            self._cache["__global__"]=(time.time(),global_news)
        self._load_x()

    def _selected_news(self, symbol: str, company: str):
        key=(symbol or company or "").upper()
        with self._lock:
            cached=self._cache.get(key)
            if cached and time.time()-cached[0] < settings.news_refresh_minutes*60:
                return cached[1]
        rows=self._load_freenews(company if company else symbol)
        with self._lock:
            self._cache[key]=(time.time(),rows)
        return rows

    def snapshot(self, symbol: str, company: str) -> dict:
        global_rows = []
        with self._lock:
            cached=self._cache.get("__global__")
            if cached: global_rows.extend(cached[1])
        selected=self._selected_news(symbol,company) if self.api_key else []
        toi=self._toi.get()
        toi_rows=toi.headlines if toi else []
        combined=[]
        seen=set()
        for row in selected+global_rows+toi_rows:
            title=row.get("title","")
            url=row.get("url","")
            key=(title.lower(),url)
            if key in seen: continue
            seen.add(key)
            row=dict(row)
            row["relevance"]=relevance_score(title,symbol,company,row.get("section",""))
            # TOI headlines are fresher when fetched by the category page; cap
            # their lexical impact rather than letting a single headline dominate.
            if row.get("source")=="Times of India":
                tone=lexical_sentiment(title); event,ew=classify_event(title)
                row["sentiment"]=tone; row["event"]=event; row["impact"]=round(tone*ew*SOURCE_WEIGHTS["times of india"],4)
            combined.append(row)

        with self._lock:
            if self._x_cache:
                combined.extend([dict(x) for x in self._x_cache[1]])

        combined.sort(key=lambda x: (float(x.get("relevance",0))*2 + abs(float(x.get("impact",0)))), reverse=True)
        top=combined[: self.max_records]
        summary=self.aggregate(top)
        display=self.display_snapshot(symbol,company)
        return {
            "summary":summary,
            "articles":top,
            "display_articles":display["articles"],
            "download_url":display["download_url"],
            "diagnostics":self.diagnostics(),
            "toi":{"fetched_at":display["fetched_at"],"sections_ok":display["sections_ok"],"sections_total":display["sections_total"]},
        }

    @staticmethod
    def aggregate(articles: Iterable[dict]) -> dict:
        items=list(articles)
        if not items:
            return {"headline_count":0,"sentiment":0.0,"impact":0.0,"positive":0,"negative":0,"neutral":0,"top_event":"none"}
        counts={"positive":0,"negative":0,"neutral":0}
        impacts=[]; events={}
        for a in items:
            s=float(a.get("sentiment",0))
            label="positive" if s>0.15 else "negative" if s<-0.15 else "neutral"
            counts[label]+=1
            impacts.append(float(a.get("impact",0)))
            events[a.get("event","other")]=events.get(a.get("event","other"),0)+abs(float(a.get("impact",0)))
        return {
            "headline_count":len(items),
            "sentiment":round(sum(float(a.get("sentiment",0)) for a in items)/len(items),4),
            "impact":round(max(-1,min(1,sum(impacts)/len(impacts))),4),
            **counts,"top_event":max(events,key=events.get) if events else "none"
        }

    def export_toi_excel(self) -> Path:
        """Export the current TOI snapshot in the same shape as the provided scraper."""
        snap=self._toi.get()
        if snap is None:
            try:
                snap=self._toi.refresh()
            except Exception:
                snap=self._toi.get()
        if snap is None:
            raise RuntimeError("Times of India news is not available yet.")
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Font, Alignment
        out=Path(self._toi.state_file).with_name("toi_market_news.xlsx")
        out.parent.mkdir(parents=True, exist_ok=True)
        wb=Workbook(); ws=wb.active; ws.title="news"
        headers=["scraped_at_utc","source","section","headline_clean","url","category_page","model_text"]
        ws.append(headers)
        for row in snap.headlines:
            ws.append([row.get("scraped_at",""),row.get("source","Times of India"),row.get("section",""),row.get("title",""),row.get("url",""),row.get("category_page",""),f"{row.get('section','')} | {row.get('title','')}"])
        for c in ws[1]:
            c.font=Font(bold=True); c.alignment=Alignment(vertical="top")
        for row in ws.iter_rows():
            for cell in row: cell.alignment=Alignment(vertical="top",wrap_text=True)
        ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
        for col,w in {"A":26,"B":20,"C":24,"D":90,"E":90,"F":70,"G":100}.items(): ws.column_dimensions[col].width=w
        meta=wb.create_sheet("run_metadata")
        meta.append(["field","value"]); meta.append(["fetched_at_utc",snap.fetched_at]); meta.append(["sections_ok",snap.sections_ok]); meta.append(["sections_total",snap.sections_total]); meta.append(["headline_count",len(snap.headlines)])
        wb.save(out)
        check=load_workbook(out,read_only=True); check.close()
        return out

    def display_snapshot(self, symbol: str, company: str) -> dict:
        """User-facing news: Times of India headlines only."""
        toi=self._toi.get()
        rows=toi.headlines if toi else []
        ranked=[]
        for row in rows:
            item=dict(row)
            item["relevance"]=relevance_score(item.get("title",""),symbol,company,item.get("section",""))
            ranked.append(item)
        market_rows=[x for x in ranked if (x.get("section","").lower() in DISPLAY_SECTIONS)]
        ranked_pool=market_rows if len(market_rows) >= 5 else ranked
        ranked_pool.sort(key=lambda x:display_rank_score(x,symbol,company),reverse=True)
        return {"articles":ranked_pool[:10],"fetched_at":toi.fetched_at if toi else None,"sections_ok":toi.sections_ok if toi else 0,"sections_total":toi.sections_total if toi else len(__import__('app.data.toi',fromlist=['SECTIONS']).SECTIONS),"download_url":"/api/news/download"}

    def refresh_toi(self):
        snap=self._toi.refresh()
        return {
            "fetched_at":snap.fetched_at,
            "sections_ok":snap.sections_ok,
            "sections_total":snap.sections_total,
            "headline_count":len(snap.headlines),
        }

    def diagnostics(self):
        toi=self._toi.get()
        return {
            "provider":settings.news_provider,
            "configured":bool(self.api_key),
            "used_today":self.quota.used,
            "remaining_today":self.quota.remaining,
            "daily_limit":self.quota.daily_limit,
            "last_error":self.last_api_error,
            "x_configured":bool(settings.x_bearer_token),
            "x_error":self.last_x_error,
            "toi_fetched_at":toi.fetched_at if toi else None,
            "toi_sections_ok":toi.sections_ok if toi else 0,
            "toi_sections_total":toi.sections_total if toi else 18,
            "next_news_refresh_minutes":settings.news_refresh_minutes,
            "next_toi_refresh_minutes":settings.toi_refresh_minutes,
        }
