from __future__ import annotations

import gzip
import io
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import INDEX_INSTRUMENTS, IST, MARKET_CLOSE_MINUTES, MARKET_OPEN_MINUTES, NSE_HOLIDAYS_SET

try:
    import upstox_client
except Exception:
    upstox_client = None

UPSTOX_API = "https://api.upstox.com"
INSTRUMENT_URLS = {
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
}

@dataclass
class Instrument:
    instrument_key: str
    exchange: str
    segment: str
    instrument_type: str
    name: str
    trading_symbol: str
    short_name: str
    isin: str = ""

class UpstoxError(RuntimeError):
    pass

class UpstoxClient:
    def __init__(self, access_token: str = "", cache_dir: str = "data_cache"):
        self.access_token = access_token.strip()
        self.session = requests.Session()
        retry = Retry(
            total=3, connect=3, read=3, backoff_factor=0.25,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(pool_connections=40, pool_maxsize=40, max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.headers.update({"Accept": "application/json", "User-Agent": "SuyashStockMarketPredictor/3.0"})
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._instruments: list[Instrument] | None = None
        self._lock = threading.RLock()
        self._rest_cache: dict[str, tuple[float, object]] = {}

    @property
    def connected(self) -> bool:
        return bool(self.access_token)

    def set_access_token(self, token: str) -> None:
        self.access_token = token.strip()

    def _headers(self):
        if not self.access_token:
            raise UpstoxError("Upstox access token is missing.")
        return {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}

    def _get_json(self, url: str, params: Optional[dict] = None):
        try:
            r = self.session.get(url, headers=self._headers(), params=params, timeout=25)
        except requests.RequestException as exc:
            raise UpstoxError(f"Upstox request failed: {exc}") from exc
        if r.status_code >= 400:
            raise UpstoxError(f"Upstox HTTP {r.status_code}: {r.text[:400]}")
        try:
            return r.json()
        except Exception as exc:
            raise UpstoxError("Upstox returned non-JSON data.") from exc

    # ---------- instrument directory ----------
    def _download_instruments(self) -> list[Instrument]:
        rows: list[Instrument] = []
        for exchange, url in INSTRUMENT_URLS.items():
            try:
                r = self.session.get(url, timeout=60)
                r.raise_for_status()
                with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                    payload = json.loads(gz.read().decode("utf-8"))
            except Exception as exc:
                raise UpstoxError(f"Instrument master download failed for {exchange}: {exc}") from exc

            for item in payload:
                segment = str(item.get("segment", ""))
                if segment not in {"NSE_EQ", "BSE_EQ"}:
                    continue
                instrument_type = str(item.get("instrument_type", "")).upper()
                if instrument_type not in {"EQ", "BE", "BZ", "ST", "SM"}:
                    continue
                key = str(item.get("instrument_key", ""))
                symbol = str(item.get("trading_symbol", ""))
                if not key or not symbol:
                    continue
                rows.append(Instrument(
                    instrument_key=key,
                    exchange=str(item.get("exchange", exchange)),
                    segment=segment,
                    instrument_type=instrument_type,
                    name=str(item.get("name", "")),
                    trading_symbol=symbol,
                    short_name=str(item.get("short_name", "")),
                    isin=str(item.get("isin", "")),
                ))
        # Stable, de-duplicated universe. This keeps unavailable securities visible.
        unique = {}
        for item in rows:
            unique[item.instrument_key] = item
        return list(unique.values())

    @staticmethod
    def is_market_open(dt: Optional[datetime] = None) -> bool:
        if dt is None:
            dt = datetime.now(ZoneInfo(IST))
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(IST))
        else:
            dt = dt.astimezone(ZoneInfo(IST))
        if dt.weekday() >= 5:
            return False
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in NSE_HOLIDAYS_SET:
            return False
        minute = dt.hour * 60 + dt.minute
        return MARKET_OPEN_MINUTES <= minute < MARKET_CLOSE_MINUTES

    def load_instruments(self, force: bool = False) -> list[Instrument]:
        with self._lock:
            if not force and self._instruments is not None:
                return self._instruments
            cache_file = self.cache_dir / "instruments_equity.json"
            if not force and cache_file.exists() and time.time() - cache_file.stat().st_mtime < 12 * 3600:
                try:
                    raw = json.loads(cache_file.read_text(encoding="utf-8"))
                    self._instruments = [Instrument(**x) for x in raw]
                    return self._instruments
                except Exception:
                    pass
            rows = self._download_instruments()
            try:
                cache_file.write_text(json.dumps([x.__dict__ for x in rows], ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            self._instruments = rows
            return rows

    def resolve_symbol(self, query: str) -> Instrument:
        raw = query.strip()
        q = raw.upper()
        aliases = {
            "SENSEX": "SENSEX", "BSE SENSEX": "SENSEX",
            "NIFTY 50": "NIFTY 50", "NIFTY": "NIFTY 50", "NSE NIFTY": "NIFTY 50",
            "NIFTY BANK": "NIFTY BANK", "BANKNIFTY": "NIFTY BANK", "NIFTYBANK": "NIFTY BANK",
            "INDIA VIX": "INDIA VIX", "VIX": "INDIA VIX",
        }
        if q in aliases:
            item = INDEX_INSTRUMENTS[aliases[q]]
            return Instrument(instrument_type="INDEX", isin="", **item)
        matches = self.search_instruments(raw, limit=25)
        if not matches:
            raise UpstoxError(f"Instrument not found: {query}")
        return matches[0]

    def search_instruments(self, query: str, limit: int = 50) -> list[Instrument]:
        q = query.strip().upper()
        if not q:
            return []
        exact, fuzzy = [], []
        for ins in self.load_instruments():
            fields = [ins.trading_symbol.upper(), ins.name.upper(), ins.short_name.upper(), ins.isin.upper()]
            if q == ins.trading_symbol.upper():
                exact.append(ins)
            elif any(q in field for field in fields):
                fuzzy.append(ins)
        exact.sort(key=lambda x: (0 if x.exchange == "NSE" else 1, x.trading_symbol))
        fuzzy.sort(key=lambda x: (0 if x.exchange == "NSE" else 1, x.trading_symbol))
        return (exact + fuzzy)[:limit]

    def universe_summary(self):
        items = self.load_instruments()
        return {
            "total": len(items),
            "nse": sum(x.exchange == "NSE" for x in items),
            "bse": sum(x.exchange == "BSE" for x in items),
            "results": [
                {"symbol": x.trading_symbol, "name": x.name, "exchange": x.exchange, "segment": x.segment, "instrument_key": x.instrument_key}
                for x in items
            ],
        }

    # ---------- candles ----------
    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
        if not candles:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
        for c in cols[1:]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    @staticmethod
    def aggregate_minutes(df: pd.DataFrame, interval: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        data = df.copy().sort_values("timestamp")
        for c in ["open","high","low","close","volume"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        ts = pd.to_datetime(data["timestamp"], utc=False).dt.tz_convert(IST)
        mins = ts.dt.hour * 60 + ts.dt.minute
        mask = (mins >= MARKET_OPEN_MINUTES) & (mins < MARKET_CLOSE_MINUTES) & (ts.dt.dayofweek < 5)
        data = data.loc[mask].copy()
        if data.empty:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","oi"])
        ts = pd.to_datetime(data["timestamp"], utc=False).dt.tz_convert(IST)
        session_open = ts.dt.normalize() + pd.to_timedelta(MARKET_OPEN_MINUTES, unit="m")
        elapsed = ((ts - session_open).dt.total_seconds() // 60).astype(int)
        bucket = (elapsed // max(1, int(interval))) * max(1, int(interval))
        data["timestamp"] = session_open + pd.to_timedelta(bucket, unit="m")
        out = data.groupby("timestamp", sort=True).agg(
            open=("open","first"), high=("high","max"), low=("low","min"),
            close=("close","last"), volume=("volume","sum")
        ).reset_index()
        out["oi"] = 0.0
        return out[["timestamp","open","high","low","close","volume","oi"]].drop_duplicates("timestamp")

    def get_intraday(self, instrument_key: str, interval) -> pd.DataFrame:
        if interval == "1d":
            encoded = quote(instrument_key, safe="")
            url = f"{UPSTOX_API}/v3/historical-candle/intraday/{encoded}/days/1"
            payload = self._get_json(url)
            return self._to_df(payload.get("data", {}).get("candles", []))
        base = self._get_intraday_1m(instrument_key)
        return self.aggregate_minutes(base, int(interval))

    def _get_intraday_1m(self, instrument_key: str) -> pd.DataFrame:
        encoded = quote(instrument_key, safe="")
        url = f"{UPSTOX_API}/v3/historical-candle/intraday/{encoded}/minutes/1"
        payload = self._get_json(url)
        return self._to_df(payload.get("data", {}).get("candles", []))

    def get_historical(self, instrument_key: str, interval: int, from_date: date, to_date: date) -> pd.DataFrame:
        # V3 minute data is limited to one month for intervals through 15m and
        # one quarter for larger minute intervals. We chunk conservatively.
        max_days = 30 if interval <= 15 else 92
        frames = []
        current_to = to_date
        encoded = quote(instrument_key, safe="")
        while current_to >= from_date:
            current_from = max(from_date, current_to - timedelta(days=max_days - 1))
            url = f"{UPSTOX_API}/v3/historical-candle/{encoded}/minutes/{int(interval)}/{current_to.isoformat()}/{current_from.isoformat()}"
            try:
                payload = self._get_json(url)
            except UpstoxError:
                # only tolerate failures on very old chunks; recent holes would corrupt features
                if current_from == from_date or (to_date - current_to).days < 60:
                    raise
                current_to = current_from - timedelta(days=1)
                continue
            frames.append(self._to_df(payload.get("data", {}).get("candles", [])))
            if current_from <= from_date:
                break
            current_to = current_from - timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","oi"])
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def get_daily_history(self, instrument_key: str, calendar_days: int = 365) -> pd.DataFrame:
        now = datetime.now(ZoneInfo(IST))
        to_date = now.date()
        from_date = to_date - timedelta(days=max(7, calendar_days))
        encoded = quote(instrument_key, safe="")
        url = f"{UPSTOX_API}/v3/historical-candle/{encoded}/days/1/{to_date.isoformat()}/{from_date.isoformat()}"
        payload = self._get_json(url)
        return self._to_df(payload.get("data", {}).get("candles", []))

    def get_recent_history(self, instrument_key: str, interval, calendar_days: int) -> pd.DataFrame:
        now = datetime.now(ZoneInfo(IST))
        to_date = now.date()
        from_date = to_date - timedelta(days=calendar_days)
        if interval == "1d":
            return self.get_daily_history(instrument_key, calendar_days)
        if int(interval) != 1:
            base = self.get_historical(instrument_key, 1, from_date, to_date)
            return self.aggregate_minutes(base, int(interval))
        return self.get_historical(instrument_key, 1, from_date, to_date)

    def get_quote_batch(self, instrument_keys: Iterable[str]) -> dict:
        keys = list(dict.fromkeys(instrument_keys))
        out = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i+500]
            payload = self._get_json(f"{UPSTOX_API}/v3/market-quote/quotes", params={"instrument_key": ",".join(batch)})
            out.update(payload.get("data", {}))
        return out

class LiveMarketHub:
    """Low-frequency browser polling + continuous server-side WebSocket.

    The WebSocket is the live data source. Browser polling is intentionally
    decoupled from the stream so the server does not hammer Upstox every second.
    """

    def __init__(self, upstox: UpstoxClient):
        self.upstox = upstox
        self._lock = threading.RLock()
        self._streamer = None
        self._started = False
        self._connected = False
        self._requested: set[str] = set()
        self._minute_bars: dict[str, dict[datetime, dict]] = {}
        self._minutes: dict[str, deque[dict]] = {}
        self._current: dict[str, dict] = {}
        self._latest_session: dict[str, dict] = {}
        self._last_error = ""
        self._worker = None

    @property
    def connected(self): 
        with self._lock: return self._connected

    @property
    def last_error(self):
        with self._lock: return self._last_error

    def start(self):
        if self._started or not self.upstox.access_token or upstox_client is None:
            return
        self._started = True
        self._worker = threading.Thread(target=self._run, name="upstox-stream", daemon=True)
        self._worker.start()

    def subscribe(self, instrument_keys: Iterable[str]):
        keys = {x for x in instrument_keys if x}
        if not keys: return
        self.start()
        with self._lock:
            new_keys = keys - self._requested
            self._requested.update(keys)
            streamer = self._streamer
            connected = self._connected
        if streamer is not None and connected and new_keys:
            try: streamer.subscribe(sorted(new_keys), "full")
            except Exception as exc:
                with self._lock: self._last_error = str(exc)

    def _run(self):
        while True:
            try:
                configuration = upstox_client.Configuration()
                configuration.access_token = self.upstox.access_token
                client = upstox_client.ApiClient(configuration)
                streamer = upstox_client.MarketDataStreamerV3(client)
                try: streamer.auto_reconnect(True, 5, 60)
                except Exception: pass
                self._streamer = streamer

                def on_open():
                    with self._lock:
                        self._connected = True
                        self._last_error = ""
                        keys = sorted(self._requested)
                    if keys:
                        try: streamer.subscribe(keys, "full")
                        except Exception as exc:
                            with self._lock: self._last_error = str(exc)

                def on_close():
                    with self._lock: self._connected = False

                def on_error(err):
                    with self._lock:
                        self._connected = False
                        self._last_error = str(err)

                def on_message(message):
                    try: self._ingest(message)
                    except Exception as exc:
                        with self._lock: self._last_error = f"feed parse: {exc}"

                streamer.on("open", on_open)
                streamer.on("close", on_close)
                streamer.on("error", on_error)
                streamer.on("message", on_message)
                streamer.connect()
                return
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                time.sleep(10)

    @staticmethod
    def _float(v):
        try: return float(v)
        except Exception: return None

    @staticmethod
    def _epoch_ms(v):
        try: return int(float(v))
        except Exception: return None

    def _upsert(self, key: str, bar: dict, is_authoritative: bool = False):
        ts = pd.Timestamp(bar["timestamp"])
        if ts.tz is None: ts = ts.tz_localize(IST)
        else: ts = ts.tz_convert(IST)
        minute_ts = ts.floor("min").to_pydatetime()
        op = float(bar["open"])
        hi = float(bar["high"])
        lo = float(bar["low"])
        cl = float(bar["close"])
        vol = float(bar.get("volume") or 0.0)

        with self._lock:
            bars = self._minute_bars.setdefault(key, {})
            existing = bars.get(minute_ts)
            if existing is None:
                bars[minute_ts] = {
                    "timestamp": minute_ts,
                    "open": op, "high": hi, "low": lo, "close": cl,
                    "volume": vol, "is_feed": is_authoritative,
                }
            else:
                if is_authoritative:
                    existing["open"] = op
                    existing["high"] = max(existing["high"], hi)
                    existing["low"] = min(existing["low"], lo)
                    existing["close"] = cl
                    existing["volume"] = max(existing.get("volume", 0.0), vol)
                    existing["is_feed"] = True
                else:
                    existing["high"] = max(existing["high"], hi)
                    existing["low"] = min(existing["low"], lo)
                    existing["close"] = cl

            if len(bars) > 6000:
                oldest = sorted(bars.keys())[:1000]
                for old in oldest:
                    del bars[old]

            sorted_items = sorted(bars.values(), key=lambda r: r["timestamp"])
            if sorted_items:
                self._current[key] = dict(sorted_items[-1])
                self._minutes[key] = deque([dict(x) for x in sorted_items[:-1]], maxlen=8000)

    def _ingest(self, message: dict):
        if not isinstance(message, dict): return
        feeds = message.get("feeds") or {}
        for key, feed in feeds.items():
            if not isinstance(feed, dict): continue
            full = feed.get("fullFeed") or feed.get("ff") or {}
            ff = full.get("marketFF") or full.get("indexFF") or {}
            if not isinstance(ff, dict): ff = {}
            ltpc = ff.get("ltpc") or feed.get("ltpc") or {}
            ltp = self._float(ltpc.get("ltp"))
            # IMPORTANT: ltt is trade time; currentTs can be a heartbeat time.
            ltt = self._epoch_ms(ltpc.get("ltt"))
            cp = self._float(ltpc.get("cp"))
            if ltp is None or ltt is None:
                continue
            trade_ts = datetime.fromtimestamp(ltt/1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(IST))
            now = datetime.now(ZoneInfo(IST))
            minute = trade_ts.hour*60 + trade_ts.minute
            if trade_ts.weekday() >= 5 or minute < MARKET_OPEN_MINUTES or minute >= MARKET_CLOSE_MINUTES:
                # Never overwrite the last valid cash-session trade with a heartbeat.
                continue
            market_ohlc = ff.get("marketOHLC") or feed.get("marketOHLC") or {}
            for candle in market_ohlc.get("ohlc") or []:
                if not isinstance(candle, dict) or candle.get("interval") != "I1":
                    continue
                ts = self._epoch_ms(candle.get("ts"))
                op = self._float(candle.get("open")); hi = self._float(candle.get("high"))
                lo = self._float(candle.get("low")); cl = self._float(candle.get("close"))
                if None in (ts, op, hi, lo, cl): continue
                self._upsert(key, {
                    "timestamp": datetime.fromtimestamp(ts/1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(IST)),
                    "open": op, "high": hi, "low": lo, "close": cl, "volume": self._float(candle.get("vol")) or 0.0
                }, is_authoritative=True)
            self._upsert(key, {"timestamp": trade_ts, "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": 0.0}, is_authoritative=False)
            with self._lock:
                self._latest_session[key] = {
                    "price": ltp, "previous_close": cp, "trade_time": trade_ts.isoformat(),
                    "updated_at": now.isoformat(), "instrument_key": key, "session_trade": True,
                }

    def latest(self, key: str):
        with self._lock:
            return dict(self._latest_session[key]) if key in self._latest_session else None

    def live_bars(self, key: str, interval: int = 1):
        with self._lock:
            bars = self._minute_bars.get(key)
            if not bars:
                rows = list(self._minutes.get(key, ()))
                current = self._current.get(key)
                if current is not None: rows.append(dict(current))
            else:
                rows = sorted(bars.values(), key=lambda r: r["timestamp"])
        if not rows: return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","oi"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False).dt.tz_convert(IST)
        df["oi"] = 0.0
        cols = ["timestamp","open","high","low","close","volume","oi"]
        clean_df = df[[c for c in cols if c in df.columns]].copy()
        if interval == 1: return clean_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        return UpstoxClient.aggregate_minutes(clean_df, interval)

    def latest_or_rest(self, key: str, interval: int):
        tick = self.latest(key)
        bars = self.live_bars(key, interval)
        return tick, bars
