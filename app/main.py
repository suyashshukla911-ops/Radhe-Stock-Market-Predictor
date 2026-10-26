from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import CACHE_DIR, IST, NSE_HOLIDAYS_LIST, PRIORITY_STOCKS, get_cache_dir, settings
from app.data.news import NewsService
from app.data.upstox import LiveMarketHub, UpstoxAuthError, UpstoxClient, UpstoxError
from app.services.engine import AnalysisEngine
from app.services.stock_forecast import StockForecastService

logger = logging.getLogger("suyash.app")

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

def _token_file() -> Path:
    return get_cache_dir() / "upstox_token.json"

def _load_saved_token():
    """Re-use today's OAuth token after a restart (Upstox tokens expire ~03:30 IST next day)."""
    try:
        tfile = _token_file()
        if not tfile.exists():
            return ""
        obj = json.loads(tfile.read_text(encoding="utf-8"))
        issued = datetime.fromisoformat(obj["issued_at"])
        now = datetime.now(ZoneInfo(IST))
        expiry = issued.replace(hour=3, minute=30, second=0, microsecond=0)
        if expiry <= issued:
            expiry += timedelta(days=1)
        return obj["access_token"] if now < expiry else ""
    except Exception:
        return ""

upstox = UpstoxClient(_load_saved_token() or settings.upstox_access_token, cache_dir=get_cache_dir())
live = LiveMarketHub(upstox)
news = NewsService(settings.news_api_key, max_records=20, state_dir=get_cache_dir())
engine = AnalysisEngine(upstox, live, news)
stock_forecaster = StockForecastService(upstox, news)
oauth_states: dict[str, float] = {}

def _eod_locker():
    """Create the fixed EOD outlook at ~09:20-09:45 IST even if nobody has the page open."""
    while True:
        try:
            now = datetime.now(ZoneInfo(IST))
            m = now.hour * 60 + now.minute
            if upstox.connected and UpstoxClient.is_market_open() and 9 * 60 + 20 <= m <= 11 * 60:
                for name in ("SENSEX", "NIFTY 50"):
                    ins = upstox.resolve_symbol(name)
                    if f"{ins.instrument_key}|{engine._today()}" not in engine._eod_cache:
                        logger.info("EOD locker: locking morning outlook for %s", name)
                        try:
                            engine.analyze(ins, 5)
                            logger.info("EOD locker: successfully locked %s", name)
                        except Exception as exc:
                            logger.warning("EOD locker: failed to lock %s: %s", name, exc)
        except Exception as exc:
            logger.warning("EOD locker loop error: %s", exc)
        time.sleep(30)

def _token_validator_and_warmup():
    """Periodic background token validation (every 60s), morning alert (08:50 IST), and pre-open warmup (09:00 IST)."""
    last_val_time = 0.0
    last_alert_day = ""
    last_warmup_day = ""
    while True:
        try:
            now = datetime.now(ZoneInfo(IST))
            today_str = now.strftime("%Y-%m-%d")
            minute = now.hour * 60 + now.minute

            if upstox.connected and time.time() - last_val_time >= 60:
                last_val_time = time.time()
                try:
                    upstox.validate_token(allow_network=True)
                except Exception:
                    pass

            if now.weekday() < 5 and 8 * 60 + 50 <= minute < 9 * 60 and last_alert_day != today_str:
                last_alert_day = today_str
                is_valid = upstox.validate_token(allow_network=True) if upstox.connected else False
                if not is_valid:
                    logger.warning("MORNING TOKEN ALERT: Upstox session token missing or expired before 09:15 open! Please authenticate via /auth/upstox/start")

            if now.weekday() < 5 and 9 * 60 <= minute < 9 * 60 + 15 and last_warmup_day != today_str:
                last_warmup_day = today_str
                try:
                    logger.info("PRE-OPEN WARMUP: Loading instrument master and pre-subscribing key indices...")
                    upstox.load_instruments()
                    keys = []
                    for sym in ("SENSEX", "NIFTY 50", "NIFTY BANK", "INDIA VIX"):
                        try:
                            keys.append(upstox.resolve_symbol(sym).instrument_key)
                        except Exception:
                            pass
                    if keys:
                        live.subscribe(keys)
                except Exception as exc:
                    logger.warning("PRE-OPEN WARMUP failed: %s", exc)

        except Exception as exc:
            logger.warning("Background validator/warmup loop error: %s", exc)
        time.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    live.start()
    news.start()
    threading.Thread(target=_eod_locker, name="eod-locker", daemon=True).start()
    threading.Thread(target=_token_validator_and_warmup, name="token-validator", daemon=True).start()
    yield

app = FastAPI(title=settings.app_name, version="3.0.0", lifespan=lifespan)

def _require_admin(request: Request):
    if settings.admin_key:
        auth_header = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
        if auth_header != settings.admin_key:
            raise HTTPException(403, "Admin authorization required.")

@app.middleware("http")
async def add_security_and_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/health")
def health():
    clock = UpstoxClient.market_clock_status()
    token_age = None
    token_issued = None
    token_expiry = None
    is_expired = False
    tfile = _token_file()
    if tfile.exists():
        try:
            info = json.loads(tfile.read_text(encoding="utf-8"))
            issued = datetime.fromisoformat(info["issued_at"])
            token_issued = issued.isoformat()
            now = datetime.now(ZoneInfo(IST))
            token_age = int((now - issued).total_seconds())
            expiry = issued.replace(hour=3, minute=30, second=0, microsecond=0)
            if expiry <= issued:
                expiry += timedelta(days=1)
            token_expiry = expiry.isoformat()
            if now >= expiry:
                is_expired = True
        except Exception:
            pass

    # Instant check: allow_network=False avoids blocking Render health checks
    token_valid = bool(upstox.connected and not is_expired and not upstox.last_auth_error and upstox.validate_token(allow_network=False))
    models_ready = [f"{k[0]}:{k[1]}m" for k in engine.models.keys()]

    return {
        "ok": True,
        "app_name": settings.app_name,
        "upstox_connected": upstox.connected,
        "token_valid": token_valid,
        "token_issued_at": token_issued,
        "token_expires_at": token_expiry,
        "token_age_seconds": token_age,
        "last_ok_at": upstox.last_ok_at,
        "eod_locked": engine.eod_locked_status(),
        "market_clock": clock,
        "stream_connected": live.connected,
        "stream_error": live.last_error,
        "models_ready": models_ready,
        "cache_dir": str(get_cache_dir()),
        "news": news.diagnostics(),
        "live_poll_seconds": settings.live_poll_seconds,
        "prediction_refresh_seconds": settings.prediction_refresh_seconds,
        "model_retrain_minutes": settings.model_retrain_minutes,
        "toi_refresh_minutes": settings.toi_refresh_minutes,
        "moneycontrol_refresh_minutes": settings.moneycontrol_refresh_minutes,
    }

@app.get("/api/priority")
def priority():
    return {"stocks": PRIORITY_STOCKS}

@app.get("/api/stocks")
def stocks(q: str = Query("", max_length=80), limit: int = Query(50, ge=1, le=100)):
    try:
        if not q.strip():
            items = upstox.load_instruments()
            return {"count": len(items), "results": [
                {"name": x.name, "short_name": x.short_name, "symbol": x.trading_symbol, "exchange": x.exchange,
                 "segment": x.segment, "instrument_key": x.instrument_key}
                for x in items[:limit]
            ]}
        matches = upstox.search_instruments(q, limit=limit)
        return {"count": len(matches), "results": [
            {"name": x.name, "short_name": x.short_name, "symbol": x.trading_symbol, "exchange": x.exchange,
             "segment": x.segment, "instrument_key": x.instrument_key}
            for x in matches
        ]}
    except UpstoxError as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/universe")
def universe(limit: int = Query(100, ge=10, le=5000)):
    try:
        summary = upstox.universe_summary()
        summary["results"] = summary["results"][:limit]
        return summary
    except Exception as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/market-pulse")
def market_pulse(limit: int = Query(30, ge=10, le=50)):
    try:
        resolved = []
        for item in PRIORITY_STOCKS:
            try:
                resolved.append(upstox.resolve_symbol(item["symbol"]))
            except Exception:
                pass
        unique = {x.instrument_key: x for x in resolved}
        quotes = upstox.get_quote_batch(unique.keys())
    except Exception as exc:
        raise HTTPException(503, str(exc))
    rows = []
    for key, q in quotes.items():
        ins = unique.get(key) or unique.get(q.get("instrument_token"))
        if not ins:
            continue
        ohlc = q.get("ohlc") or {}
        last = q.get("last_price") or ohlc.get("close")
        prev = q.get("prev_close_price")
        net = q.get("net_change")
        if net is None and last is not None and prev not in (None, 0):
            net = last - prev
        change_pct = (net / prev * 100) if net is not None and prev not in (None, 0) else None
        if last is not None:
            rows.append({
                "symbol": ins.trading_symbol, "name": ins.name, "exchange": ins.exchange,
                "instrument_key": ins.instrument_key, "price": last, "change": net,
                "change_pct": change_pct, "volume": q.get("volume") or ohlc.get("volume")
            })
    rows.sort(key=lambda r: r["change_pct"] if r["change_pct"] is not None else -999, reverse=True)
    return {"count": len(rows), "results": rows[:limit]}

@app.get("/api/news")
def news_endpoint(symbol: str = "SENSEX", name: str = "SENSEX"):
    return news.snapshot(symbol, name)

@app.get("/api/news/download")
def news_download():
    try:
        path = news.export_toi_excel()
        return FileResponse(path, filename="toi_market_news.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/news/refresh")
def news_refresh(request: Request):
    _require_admin(request)
    try:
        result = news.refresh_toi()
        return {"ok": True, **result}
    except Exception as exc:
        raise HTTPException(503, str(exc))

@app.get("/api/news/quota")
def news_quota():
    return news.diagnostics()

@app.get("/api/analyze")
def analyze(request: Request, symbol: str = Query(..., min_length=1, max_length=100), interval: str = Query("5"), retrain: bool = False):
    if retrain:
        _require_admin(request)
    if interval not in {"1", "5", "10", "15", "30", "60"}:
        raise HTTPException(422, "Prediction timeframes: 1m, 5m, 10m, 15m, 30m and 60m. Use 1D for previous-session history.")
    try:
        ins = upstox.resolve_symbol(symbol)
        if not ins.segment.endswith("INDEX") and ins.trading_symbol.upper() not in {"SENSEX", "NIFTY 50"}:
            raise HTTPException(422, "Intraday model dashboard is reserved for SENSEX and NIFTY 50. Use the next-session stock forecast for listed equities.")
        return engine.analyze(ins, interval=int(interval), force_retrain=retrain)
    except UpstoxError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Analysis failed for %s %s: %s", symbol, interval, exc)
        raise HTTPException(500, f"Analysis failed: {exc}")

@app.get("/api/stock-forecast")
def stock_forecast(symbol: str = Query(..., min_length=1, max_length=100), retrain: bool = False):
    try:
        ins = upstox.resolve_symbol(symbol)
        if ins.segment.endswith("INDEX"):
            raise HTTPException(422, "Use the Sensex/Nifty intraday dashboards for index trading. Stock forecast is for listed equities.")
        return stock_forecaster.forecast(ins, force=retrain)
    except UpstoxError as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Stock forecast failed for %s: %s", symbol, exc)
        raise HTTPException(500, f"Stock forecast failed: {exc}")

@app.get("/api/live")
def live_state(symbol: str = Query(..., min_length=1, max_length=100), interval: str = Query("5")):
    if interval not in {"1", "5", "10", "15", "30", "60", "1d"}:
        raise HTTPException(422, "Supported views: 1m, 5m, 10m, 15m, 30m, 60m and 1D.")
    clock = UpstoxClient.market_clock_status()
    try:
        ins = upstox.resolve_symbol(symbol)
        return engine.live_state(ins, interval if interval == "1d" else int(interval))
    except UpstoxAuthError as exc:
        logger.warning("Auth error in /api/live for symbol=%s interval=%s: %s", symbol, interval, exc)
        return {
            "now": datetime.now(ZoneInfo(IST)).isoformat(),
            "instrument_key": "",
            "symbol": symbol,
            "tick": None,
            "candles": [],
            "current_candle": None,
            "seconds_to_candle_close": 0,
            "prediction": None,
            "market_data": {
                "stream_connected": live.connected,
                "stream_error": str(exc),
                "last_update_age_seconds": None,
                "source": "unavailable",
                "market_status": clock["market_status"],
                "market_clock_label": clock["label"],
                "data_status": "TOKEN_EXPIRED",
                "data_message": "Upstox session expired (401/403). Please re-login.",
                "last_market_price": None,
                "last_market_candle": None,
            },
        }
    except Exception as exc:
        logger.exception("Error in /api/live for symbol=%s interval=%s: %s", symbol, interval, exc)
        err_str = str(exc)
        if not upstox.access_token:
            data_status = "NO_TOKEN"
            data_msg = "Upstox access token missing. Please connect Upstox."
        elif upstox.last_auth_error:
            data_status = "TOKEN_EXPIRED"
            data_msg = "Upstox session expired (401/403). Please re-login."
        elif clock["is_open"]:
            data_status = "FEED_DOWN"
            data_msg = f"Live feed recovering: {err_str}"
        else:
            data_status = "ERROR"
            data_msg = f"Error: {err_str}"

        return {
            "now": datetime.now(ZoneInfo(IST)).isoformat(),
            "instrument_key": "",
            "symbol": symbol,
            "tick": None,
            "candles": [],
            "current_candle": None,
            "seconds_to_candle_close": 0,
            "prediction": None,
            "market_data": {
                "stream_connected": live.connected,
                "stream_error": err_str,
                "last_update_age_seconds": None,
                "source": "unavailable",
                "market_status": clock["market_status"],
                "market_clock_label": clock["label"],
                "data_status": data_status,
                "data_message": data_msg,
                "last_market_price": None,
                "last_market_candle": None,
            },
        }

@app.get("/auth/upstox/start")
def upstox_start():
    global oauth_states
    if not settings.upstox_client_id or not settings.upstox_redirect_uri:
        raise HTTPException(503, "OAuth fields are not configured. Check UPSTOX_CLIENT_ID and UPSTOX_REDIRECT_URI.")
    now_ts = time.time()
    oauth_states = {s: t for s, t in oauth_states.items() if now_ts - t < 600}
    state = secrets.token_urlsafe(24)
    oauth_states[state] = now_ts
    params = {"response_type": "code", "client_id": settings.upstox_client_id, "redirect_uri": settings.upstox_redirect_uri, "state": state}
    return RedirectResponse(f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}")

@app.get("/auth/upstox/callback")
def upstox_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    global oauth_states
    if error:
        return RedirectResponse(f"/?auth_error={quote(error)}")
    now_ts = time.time()
    oauth_states = {s: t for s, t in oauth_states.items() if now_ts - t < 600}
    if not code or state not in oauth_states:
        raise HTTPException(400, "Invalid or expired OAuth callback state. Please try connecting again.")
    del oauth_states[state]
    try:
        response = upstox.session.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={"code": code, "client_id": settings.upstox_client_id, "client_secret": settings.upstox_client_secret,
                  "redirect_uri": settings.upstox_redirect_uri, "grant_type": "authorization_code"},
            timeout=25,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("No access token returned by Upstox.")
        upstox.set_access_token(token)
        try:
            _token_file().write_text(json.dumps({"access_token": token, "issued_at": datetime.now(ZoneInfo(IST)).isoformat()}), encoding="utf-8")
        except Exception:
            pass
        live.restart(token)
        return RedirectResponse("/?connected=1")
    except Exception as exc:
        logger.exception("Upstox OAuth exchange failed: %s", exc)
        return RedirectResponse(f"/?auth_error={quote(str(exc))}")

@app.get("/api/status")
def status_endpoint():
    val_status = upstox.validate_token(allow_network=True) if upstox.connected else False
    return {
        "ok": True,
        "app_name": settings.app_name,
        "upstox_connected": upstox.connected,
        "token_valid": val_status,
        "last_auth_error": upstox.last_auth_error,
        "last_ok_at": upstox.last_ok_at,
        "stream_connected": live.connected,
        "stream_last_error": live.last_error,
        "reconnect_attempts": getattr(live, "reconnect_attempts", 0),
        "eod_locked": engine.eod_locked_status(),
        "cached_models": [f"{k[0]}:{k[1]}m" for k in engine.models.keys()],
        "news_diagnostics": news.diagnostics(),
        "cache_dir": str(get_cache_dir()),
    }

@app.get("/api/journal")
def journal_endpoint(limit: int = Query(50, ge=1, le=200)):
    items = engine.get_journal(limit)
    return {"count": len(items), "predictions": items}

@app.get("/api/config")
def config():
    return {
        "app_name": settings.app_name,
        "browser_poll_seconds": settings.live_poll_seconds,
        "prediction_refresh_seconds": settings.prediction_refresh_seconds,
        "model_retrain_minutes": settings.model_retrain_minutes,
        "news_refresh_minutes": settings.news_refresh_minutes,
        "toi_refresh_minutes": settings.toi_refresh_minutes,
        "moneycontrol_refresh_minutes": settings.moneycontrol_refresh_minutes,
        "supports": {"timeframes": ["1", "5", "10", "15", "30", "60", "1d"]},
        "nse_holidays": NSE_HOLIDAYS_LIST,
    }
