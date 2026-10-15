from __future__ import annotations

import json
import time
import threading
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

from app.config import INDEX_INSTRUMENTS, IST, MARKET_CLOSE_MINUTES, MARKET_OPEN_MINUTES

try:
    import upstox_client
except Exception:  # optional until installed
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
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=Retry(total=2, backoff_factor=0.15, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"})))
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "RadheStockMarketPredictor/2.0",
        })
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._instruments: list[Instrument] | None = None
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        return bool(self.access_token)

    def set_access_token(self, token: str) -> None:
        self.access_token = token.strip()

    def _headers(self) -> dict[str, str]:
        if not self.access_token:
            raise UpstoxError("Upstox is not connected. Set UPSTOX_ACCESS_TOKEN in the environment.")
        return {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}

    def _get_json(self, url: str, params: Optional[dict] = None) -> dict:
        try:
            response = self.session.get(url, headers=self._headers(), params=params, timeout=25)
        except requests.RequestException as exc:
            raise UpstoxError(f"Upstox request failed: {exc}") from exc
        if response.status_code >= 400:
            raise UpstoxError(f"Upstox HTTP {response.status_code}: {response.text[:500]}")
        try:
            return response.json()
        except Exception as exc:
            raise UpstoxError("Upstox returned a non-JSON response.") from exc

    # ---------- Instruments ----------

    def _download_instruments(self) -> list[Instrument]:
        import gzip
        import io
        rows: list[Instrument] = []
        for exchange, url in INSTRUMENT_URLS.items():
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise UpstoxError(f"Instrument master download failed for {exchange}: {exc}") from exc
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                payload = json.loads(gz.read().decode("utf-8"))
            for item in payload:
                if item.get("segment") not in {"NSE_EQ", "BSE_EQ"}:
                    continue
                instrument_type = str(item.get("instrument_type", "")).upper()
                if instrument_type not in {"EQ", "BE", "BZ", "ST", "SM"}:
                    continue
                rows.append(Instrument(
                    instrument_key=item.get("instrument_key", ""),
                    exchange=item.get("exchange", exchange),
                    segment=item.get("segment", ""),
                    instrument_type=instrument_type,
                    name=item.get("name", "") or "",
                    trading_symbol=item.get("trading_symbol", "") or "",
                    short_name=item.get("short_name", "") or "",
                    isin=item.get("isin", "") or "",
                ))
        return [r for r in rows if r.instrument_key and r.trading_symbol]

    def load_instruments(self, force: bool = False) -> list[Instrument]:
        cache_file = self.cache_dir / "instruments_equity.json"
        with self._lock:
            if not force and cache_file.exists():
                age = time.time() - cache_file.stat().st_mtime
                if age < 20 * 3600:
                    try:
                        raw = json.loads(cache_file.read_text(encoding="utf-8"))
                        self._instruments = [Instrument(**x) for x in raw]
                        return self._instruments
                    except Exception:
                        pass
            rows = self._download_instruments()
            cache_file.write_text(json.dumps([r.__dict__ for r in rows], ensure_ascii=False), encoding="utf-8")
            self._instruments = rows
            return rows

    def resolve_symbol(self, symbol_or_name: str) -> Instrument:
        raw = symbol_or_name.strip()
        q = raw.upper()
        for aliases, key in [
            (["SENSEX", "BSE SENSEX"], "SENSEX"),
            (["NIFTY 50", "NIFTY", "NSE NIFTY"], "NIFTY 50"),
            (["NIFTY BANK", "BANKNIFTY", "NIFTYBANK"], "NIFTY BANK"),
            (["INDIA VIX", "VIX"], "INDIA VIX"),
        ]:
            if q in {a.upper() for a in aliases}:
                item = INDEX_INSTRUMENTS[key]
                return Instrument(instrument_type="INDEX", isin="", **item)

        matches = self.search_instruments(raw, limit=25)
        if not matches:
            raise UpstoxError(f"Indian instrument not found in the Upstox universe: {symbol_or_name}")
        return matches[0]

    def search_instruments(self, query: str, limit: int = 25) -> list[Instrument]:
        q = query.strip().upper()
        if not q:
            return []
        items = self.load_instruments()
        exact: list[Instrument] = []
        fuzzy: list[Instrument] = []
        for ins in items:
            fields = [ins.trading_symbol.upper(), ins.name.upper(), ins.short_name.upper(), ins.isin.upper()]
            if q == ins.trading_symbol.upper():
                exact.append(ins)
            elif any(q in field for field in fields):
                fuzzy.append(ins)
        ordered = exact + fuzzy
        ordered.sort(key=lambda x: (0 if x.exchange == "NSE" else 1, x.trading_symbol))
        return ordered[:limit]

    # ---------- Market data ----------

    @staticmethod
    def _to_df(candles: list) -> pd.DataFrame:
        columns = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
        if not candles:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(candles, columns=columns[:len(candles[0])])
        for c in ["open", "high", "low", "close", "volume", "oi"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    @staticmethod
    def _aggregate_minutes(df: pd.DataFrame, interval: int) -> pd.DataFrame:
        if df.empty or interval <= 1:
            return df.copy()
        data = df.copy().sort_values("timestamp")
        ts = pd.to_datetime(data["timestamp"], utc=False).dt.tz_convert(IST)
        mins = ts.dt.hour * 60 + ts.dt.minute
        # Indian cash-market session: 09:15 through 15:30 IST. Ignore auction/after-close noise.
        keep = (mins >= MARKET_OPEN_MINUTES) & (mins < MARKET_CLOSE_MINUTES)
        data = data.loc[keep].copy()
        if data.empty:
            return data
        ts = pd.to_datetime(data["timestamp"], utc=False).dt.tz_convert(IST)
        session_open = ts.dt.normalize() + pd.Timedelta(minutes=MARKET_OPEN_MINUTES)
        delta = ((ts - session_open).dt.total_seconds() // 60).astype(int)
        data["bucket"] = session_open + pd.to_timedelta((delta // int(interval)) * int(interval), unit="m")
        grouped = data.groupby("bucket", sort=True)
        out = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index().rename(columns={"bucket":"timestamp"})
        out["oi"] = 0.0
        return out[["timestamp","open","high","low","close","volume","oi"]]

    def get_intraday(self, instrument_key: str, interval: int) -> pd.DataFrame:
        encoded = quote(instrument_key, safe="")
        try:
            url = f"{UPSTOX_API}/v3/historical-candle/intraday/{encoded}/minutes/{int(interval)}"
            payload = self._get_json(url)
            df = self._to_df(payload.get("data", {}).get("candles", []))
        except UpstoxError:
            df = pd.DataFrame()
        if not df.empty:
            return self._aggregate_minutes(df, interval)
        if interval != 1:
            # V3 supports custom intervals, but a 1-minute fallback is more resilient for indices.
            url = f"{UPSTOX_API}/v3/historical-candle/intraday/{encoded}/minutes/1"
            payload = self._get_json(url)
            base = self._to_df(payload.get("data", {}).get("candles", []))
            return self._aggregate_minutes(base, interval)
        return df

    def get_historical(self, instrument_key: str, interval: int, from_date: date, to_date: date) -> pd.DataFrame:
        # V3 allows a maximum of one month for 1-15 minute candles and one quarter for larger minute intervals.
        if interval <= 15:
            max_days = 30
        else:
            max_days = 92
        frames = []
        current_to = to_date
        while current_to >= from_date:
            current_from = max(from_date, current_to - timedelta(days=max_days - 1))
            encoded = quote(instrument_key, safe="")
            url = f"{UPSTOX_API}/v3/historical-candle/{encoded}/minutes/{int(interval)}/{current_to.isoformat()}/{current_from.isoformat()}"
            payload = self._get_json(url)
            frames.append(self._to_df(payload.get("data", {}).get("candles", [])))
            if current_from <= from_date:
                break
            current_to = current_from - timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def get_recent_history(self, instrument_key: str, interval: int, calendar_days: int) -> pd.DataFrame:
        now = datetime.now(ZoneInfo(IST))
        to_date = now.date()
        from_date = to_date - timedelta(days=calendar_days)
        try:
            direct = self.get_historical(instrument_key, interval, from_date, to_date)
        except UpstoxError:
            direct = pd.DataFrame()
        if not direct.empty:
            return self._aggregate_minutes(direct, interval)
        if interval != 1:
            # Fallback to 1-minute history and build the requested timeframe locally.
            base = self.get_historical(instrument_key, 1, from_date, to_date)
            return self._aggregate_minutes(base, interval)
        return direct

    def get_quote_batch(self, instrument_keys: Iterable[str]) -> dict:
        keys = list(dict.fromkeys(instrument_keys))
        if not keys:
            return {}
        data: dict = {}
        for start in range(0, len(keys), 500):
            batch = keys[start:start + 500]
            payload = self._get_json(f"{UPSTOX_API}/v3/market-quote/quotes", params={"instrument_key": ",".join(batch)})
            data.update(payload.get("data", {}))
        return data


class LiveMarketHub:
    """Server-side Upstox V3 feed with a rolling 1-minute bar store.

    The access token stays server-side. The V3 WebSocket is the primary live source;
    its full feed includes LTPC plus 1-minute/30-minute OHLC snapshots. Tick data is
    also folded into the current 1-minute bar so the forming candle moves immediately.
    """

    def __init__(self, upstox: UpstoxClient):
        self.upstox = upstox
        self._lock = threading.RLock()
        self._streamer = None
        self._started = False
        self._connected = False
        self._requested: set[str] = set()
        self._minutes: dict[str, deque[dict]] = {}
        self._current: dict[str, dict] = {}
        self._latest: dict[str, dict] = {}
        self._last_error = ""
        self._worker = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        if self._started or not self.upstox.access_token or upstox_client is None:
            return
        self._started = True
        self._worker = threading.Thread(target=self._run, name="upstox-stream", daemon=True)
        self._worker.start()

    def subscribe(self, instrument_keys: Iterable[str]) -> None:
        keys = {k for k in instrument_keys if k}
        if not keys:
            return
        self.start()
        with self._lock:
            new_keys = keys - self._requested
            self._requested.update(keys)
            if self._streamer is not None and self._connected and new_keys:
                try:
                    self._streamer.subscribe(sorted(new_keys), "full")
                except Exception as exc:
                    self._last_error = str(exc)

    def _run(self) -> None:
        while True:
            try:
                configuration = upstox_client.Configuration()
                configuration.access_token = self.upstox.access_token
                client = upstox_client.ApiClient(configuration)
                streamer = upstox_client.MarketDataStreamerV3(client)
                try:
                    streamer.auto_reconnect(True, 5, 100)
                except Exception:
                    pass
                self._streamer = streamer

                def on_open():
                    with self._lock:
                        self._connected = True
                        self._last_error = ""
                        keys = sorted(self._requested)
                    if keys:
                        try:
                            streamer.subscribe(keys, "full")
                        except Exception as exc:
                            with self._lock:
                                self._last_error = str(exc)

                def on_close():
                    with self._lock:
                        self._connected = False

                def on_error(err):
                    with self._lock:
                        self._connected = False
                        self._last_error = str(err)

                def on_message(message):
                    try:
                        self._ingest(message)
                    except Exception as exc:
                        with self._lock:
                            self._last_error = f"feed parse: {exc}"

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
    def _as_float(value):
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _as_epoch_ms(value):
        try:
            return int(float(value))
        except Exception:
            return None

    def _upsert_minute_bar(self, key: str, bar: dict) -> None:
        """Merge a websocket OHLC snapshot or tick into the local 1-minute store."""
        ts = pd.Timestamp(bar["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        ts = ts.floor("min")
        row = {
            "timestamp": ts.to_pydatetime(),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume") or 0.0),
        }
        with self._lock:
            current = self._current.get(key)
            if current is None or current["timestamp"] != row["timestamp"]:
                if current is not None:
                    self._minutes.setdefault(key, deque(maxlen=4000)).append(dict(current))
                self._current[key] = row
                return
            # Snapshot OHLC is authoritative for the current minute; merge later ticks.
            current["open"] = float(current.get("open", row["open"]))
            current["high"] = max(float(current.get("high", row["high"])), row["high"])
            current["low"] = min(float(current.get("low", row["low"])), row["low"])
            current["close"] = row["close"]
            if row.get("volume") is not None:
                current["volume"] = max(float(current.get("volume", 0.0)), float(row["volume"]))

    def _extract_full_feed(self, feed: dict) -> dict:
        # SDK versions/documentation use both fullFeed and ff.
        full = feed.get("fullFeed") or feed.get("ff") or {}
        ff = full.get("marketFF") or full.get("indexFF") or {}
        return ff if isinstance(ff, dict) else {}

    def _ingest(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
        feeds = message.get("feeds") or {}
        for key, feed in feeds.items():
            if not isinstance(feed, dict):
                continue
            direct = feed.get("ltpc") or {}
            ff = self._extract_full_feed(feed)
            ff_ltpc = ff.get("ltpc") or {}
            ltpc = ff_ltpc or direct
            ltp = self._as_float(ltpc.get("ltp"))
            ltt = self._as_epoch_ms(ltpc.get("ltt") or message.get("currentTs"))
            cp = self._as_float(ltpc.get("cp"))
            ltq = self._as_float(ltpc.get("ltq"))
            if ltp is None:
                continue
            if ltt is None:
                ltt = int(time.time() * 1000)
            trade_ts = datetime.fromtimestamp(ltt / 1000.0, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(IST))
            now_ist = datetime.now(ZoneInfo(IST))
            if trade_ts.year < 2020:
                return
            minute_of_day = trade_ts.hour * 60 + trade_ts.minute
            # Ignore heartbeats/late packets outside the NSE/BSE cash session when building candles.
            if trade_ts.weekday() >= 5 or minute_of_day < MARKET_OPEN_MINUTES or minute_of_day >= MARKET_CLOSE_MINUTES:
                with self._lock:
                    self._latest[key] = {
                        "price": ltp, "previous_close": cp, "trade_time": trade_ts.isoformat(),
                        "updated_at": now_ist.isoformat(), "volume": self._as_float(ff.get("vtt")),
                        "ltq": ltq, "instrument_key": key, "session_trade": False,
                    }
                return

            # First consume the authoritative 1-minute OHLC snapshot if present.
            market_ohlc = ff.get("marketOHLC") or feed.get("marketOHLC") or {}
            for candle in (market_ohlc.get("ohlc") or []):
                if not isinstance(candle, dict) or candle.get("interval") != "I1":
                    continue
                cts = self._as_epoch_ms(candle.get("ts"))
                op = self._as_float(candle.get("open")); hi = self._as_float(candle.get("high"))
                lo = self._as_float(candle.get("low")); cl = self._as_float(candle.get("close"))
                if cts is None or op is None or hi is None or lo is None or cl is None:
                    continue
                self._upsert_minute_bar(key, {
                    "timestamp": datetime.fromtimestamp(cts / 1000.0, tz=ZoneInfo("UTC")).astimezone(ZoneInfo(IST)),
                    "open": op, "high": hi, "low": lo, "close": cl,
                    "volume": self._as_float(candle.get("vol")) or 0.0,
                })

            # Tick-level update makes the forming minute move immediately.
            self._upsert_minute_bar(key, {
                "timestamp": trade_ts,
                "open": ltp,
                "high": ltp,
                "low": ltp,
                "close": ltp,
                "volume": self._as_float(ff.get("vtt")) or 0.0,
            })

            with self._lock:
                self._latest[key] = {
                    "price": ltp,
                    "previous_close": cp,
                    "trade_time": trade_ts.isoformat(),
                    "updated_at": now_ist.isoformat(),
                    "volume": self._as_float(ff.get("vtt")),
                    "ltq": ltq,
                    "instrument_key": key,
                    "session_trade": True,
                }

    def latest(self, instrument_key: str) -> dict | None:
        with self._lock:
            value = self._latest.get(instrument_key)
            return dict(value) if value else None

    def one_minute_bars(self, instrument_key: str) -> pd.DataFrame:
        with self._lock:
            rows = list(self._minutes.get(instrument_key, deque()))
            current = self._current.get(instrument_key)
            if current:
                rows.append(dict(current))
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        return pd.DataFrame(rows).assign(oi=0.0).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    def live_bars(self, instrument_key: str, interval: int) -> pd.DataFrame:
        one = self.one_minute_bars(instrument_key)
        if one.empty:
            return one
        one = one.copy()
        ts = pd.to_datetime(one["timestamp"], utc=False).dt.tz_convert(IST)
        mins = ts.dt.hour * 60 + ts.dt.minute
        one = one.loc[(mins >= MARKET_OPEN_MINUTES) & (mins < MARKET_CLOSE_MINUTES)].copy()
        if one.empty:
            return one
        ts = pd.to_datetime(one["timestamp"], utc=False).dt.tz_convert(IST)
        session_open = ts.dt.normalize() + pd.Timedelta(minutes=MARKET_OPEN_MINUTES)
        delta_min = ((ts - session_open).dt.total_seconds() // 60).astype(int)
        one["bucket"] = session_open + pd.to_timedelta((delta_min // int(interval)) * int(interval), unit="m")
        grouped = one.groupby("bucket", sort=True)
        out = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).reset_index().rename(columns={"bucket":"timestamp"})
        out["oi"] = 0.0
        return out

