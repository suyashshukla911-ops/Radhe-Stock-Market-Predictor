from __future__ import annotations

import json
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

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import IST, PRIORITY_STOCKS, settings
from app.data.news import NewsService
from app.data.upstox import LiveMarketHub, UpstoxClient, UpstoxError
from app.services.engine import AnalysisEngine
from app.services.stock_forecast import StockForecastService

BASE_DIR=Path(__file__).resolve().parents[1]
FRONTEND_DIR=BASE_DIR/"frontend"

CACHE_DIR=Path(os.getenv("CACHE_DIR") or os.getenv("DATA_DIR") or BASE_DIR/"data_cache")
try: CACHE_DIR.mkdir(parents=True,exist_ok=True)
except OSError: CACHE_DIR=BASE_DIR/"data_cache"; CACHE_DIR.mkdir(parents=True,exist_ok=True)
TOKEN_FILE=CACHE_DIR/"upstox_token.json"

def _load_saved_token():
    """Re-use today's OAuth token after a restart (Upstox tokens expire ~03:30 IST next day)."""
    try:
        obj=json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        issued=datetime.fromisoformat(obj["issued_at"]); now=datetime.now(ZoneInfo(IST))
        expiry=issued.replace(hour=3,minute=30,second=0,microsecond=0)
        if expiry<=issued: expiry+=timedelta(days=1)
        return obj["access_token"] if now<expiry else ""
    except Exception: return ""

upstox=UpstoxClient(_load_saved_token() or settings.upstox_access_token,cache_dir=str(CACHE_DIR))
live=LiveMarketHub(upstox)
news=NewsService(settings.news_api_key,max_records=20,state_dir=str(CACHE_DIR))
engine=AnalysisEngine(upstox,live,news)
stock_forecaster=StockForecastService(upstox,news)
oauth_states:set[str]=set()

def _eod_locker():
    """Create the fixed EOD outlook at ~09:20-09:45 IST even if nobody has the page open."""
    while True:
        try:
            now=datetime.now(ZoneInfo(IST)); m=now.hour*60+now.minute
            if upstox.connected and UpstoxClient.is_market_open() and 9*60+20<=m<=11*60:
                for name in ("SENSEX","NIFTY 50"):
                    ins=upstox.resolve_symbol(name)
                    if f"{ins.instrument_key}|{engine._today()}" not in engine._eod_cache:
                        try: engine.analyze(ins,5)
                        except Exception: pass
        except Exception: pass
        time.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    live.start()
    news.start()
    threading.Thread(target=_eod_locker,name="eod-locker",daemon=True).start()
    yield

app=FastAPI(title=settings.app_name, version="3.0.0", lifespan=lifespan)
app.mount("/static",StaticFiles(directory=FRONTEND_DIR),name="static")

@app.get("/",include_in_schema=False)
def index(): return FileResponse(FRONTEND_DIR/"index.html")

@app.get("/api/health")
def health():
    return {
        "ok":True,
        "app_name":settings.app_name,
        "upstox_connected":upstox.connected,
        "stream_connected":live.connected,
        "stream_error":live.last_error,
        "news":news.diagnostics(),
        "live_poll_seconds":settings.live_poll_seconds,
        "prediction_refresh_seconds":settings.prediction_refresh_seconds,
        "model_retrain_minutes":settings.model_retrain_minutes,
        "toi_refresh_minutes":settings.toi_refresh_minutes,
        "moneycontrol_refresh_minutes":settings.moneycontrol_refresh_minutes,
    }

@app.get("/api/priority")
def priority(): return {"stocks":PRIORITY_STOCKS}

@app.get("/api/stocks")
def stocks(q:str=Query("",max_length=80),limit:int=Query(50,ge=1,le=100)):
    try:
        if not q.strip():
            # Returning the complete universe is intentionally supported; the
            # frontend renders a compact subset but searchable access remains full.
            items=upstox.load_instruments()
            return {"count":len(items),"results":[
                {"name":x.name,"short_name":x.short_name,"symbol":x.trading_symbol,"exchange":x.exchange,
                 "segment":x.segment,"instrument_key":x.instrument_key}
                for x in items[:limit]
            ]}
        matches=upstox.search_instruments(q,limit=limit)
        return {"count":len(matches),"results":[
            {"name":x.name,"short_name":x.short_name,"symbol":x.trading_symbol,"exchange":x.exchange,
             "segment":x.segment,"instrument_key":x.instrument_key}
            for x in matches
        ]}
    except UpstoxError as exc:
        raise HTTPException(503,str(exc))

@app.get("/api/universe")
def universe(limit: int = Query(100, ge=10, le=5000)):
    try:
        summary = upstox.universe_summary()
        summary["results"] = summary["results"][:limit]
        return summary
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/market-pulse")
def market_pulse(limit:int=Query(30,ge=10,le=50)):
    try:
        resolved=[]
        for item in PRIORITY_STOCKS:
            try: resolved.append(upstox.resolve_symbol(item["symbol"]))
            except Exception: pass
        unique={x.instrument_key:x for x in resolved}
        quotes=upstox.get_quote_batch(unique.keys())
    except Exception as exc:
        raise HTTPException(503,str(exc))
    rows=[]
    for key,q in quotes.items():
        ins=unique.get(key) or unique.get(q.get("instrument_token"))
        if not ins: continue
        ohlc=q.get("ohlc") or {}
        last=q.get("last_price") or ohlc.get("close")
        prev=q.get("prev_close_price")
        net=q.get("net_change")
        if net is None and last is not None and prev not in (None,0): net=last-prev
        change_pct=(net/prev*100) if net is not None and prev not in (None,0) else None
        if last is not None:
            rows.append({"symbol":ins.trading_symbol,"name":ins.name,"exchange":ins.exchange,"instrument_key":ins.instrument_key,"price":last,"change":net,"change_pct":change_pct,"volume":q.get("volume") or ohlc.get("volume")})
    rows.sort(key=lambda r:r["change_pct"] if r["change_pct"] is not None else -999,reverse=True)
    return {"count":len(rows),"results":rows[:limit]}

@app.get("/api/news")
def news_endpoint(symbol:str="SENSEX",name:str="SENSEX"):
    return news.snapshot(symbol,name)

@app.get("/api/news/download")
def news_download():
    try:
        path=news.export_toi_excel()
        return FileResponse(path,filename="toi_market_news.xlsx",media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        raise HTTPException(503,str(exc))

@app.get("/api/news/refresh")
def news_refresh():
    try:
        result=news.refresh_toi()
        return {"ok":True,**result}
    except Exception as exc:
        raise HTTPException(503,str(exc))

@app.get("/api/news/quota")
def news_quota(): return news.diagnostics()

@app.get("/api/analyze")
def analyze(symbol:str=Query(...,min_length=1,max_length=100),interval:str=Query("5"),retrain:bool=False):
    if interval not in {"1","5","10","15","30","60"}:
        raise HTTPException(422,"Prediction timeframes: 1m, 5m, 10m, 15m, 30m and 60m. Use 1D for previous-session history.")
    try:
        ins=upstox.resolve_symbol(symbol)
        if not ins.segment.endswith("INDEX") and ins.trading_symbol.upper() not in {"SENSEX","NIFTY 50"}:
            raise HTTPException(422,"Intraday model dashboard is reserved for SENSEX and NIFTY 50. Use the next-session stock forecast for listed equities.")
        return engine.analyze(ins,interval=int(interval),force_retrain=retrain)
    except UpstoxError as exc: raise HTTPException(503,str(exc))
    except ValueError as exc: raise HTTPException(422,str(exc))
    except HTTPException: raise
    except Exception as exc: raise HTTPException(500,f"Analysis failed: {exc}")

@app.get("/api/stock-forecast")
def stock_forecast(symbol: str = Query(..., min_length=1, max_length=100), retrain: bool=False):
    try:
        ins=upstox.resolve_symbol(symbol)
        if ins.segment.endswith("INDEX"):
            raise HTTPException(422,"Use the Sensex/Nifty intraday dashboards for index trading. Stock forecast is for listed equities.")
        return stock_forecaster.forecast(ins, force=retrain)
    except UpstoxError as exc:
        raise HTTPException(503,str(exc))
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500,f"Stock forecast failed: {exc}")

@app.get("/api/live")
def live_state(symbol:str=Query(...,min_length=1,max_length=100),interval:str=Query("5")):
    if interval not in {"1","5","10","15","30","60","1d"}:
        raise HTTPException(422,"Supported views: 1m, 5m, 10m, 15m, 30m, 60m and 1D.")
    try:
        ins=upstox.resolve_symbol(symbol)
        return engine.live_state(ins,interval if interval=="1d" else int(interval))
    except Exception as exc:
        return {
            "now": datetime.now(ZoneInfo(IST)).isoformat(),
            "instrument_key":"",
            "symbol":symbol,
            "tick":None,
            "candles":[],
            "current_candle":None,
            "seconds_to_candle_close":0,
            "prediction":None,
            "market_data":{
                "stream_connected":live.connected,
                "stream_error":str(exc),
                "last_update_age_seconds":None,
                "source":"unavailable",
                "market_status":"UNKNOWN",
                "last_market_price":None,
                "last_market_candle":None,
            },
        }

@app.get("/auth/upstox/start")
def upstox_start():
    if not settings.upstox_client_id or not settings.upstox_redirect_uri: raise HTTPException(503,"OAuth fields are not configured.")
    state=secrets.token_urlsafe(24); oauth_states.add(state)
    params={"response_type":"code","client_id":settings.upstox_client_id,"redirect_uri":settings.upstox_redirect_uri,"state":state}
    return RedirectResponse(f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}")

@app.get("/auth/upstox/callback")
def upstox_callback(code:Optional[str]=None,state:Optional[str]=None,error:Optional[str]=None):
    if error: return RedirectResponse(f"/?auth_error={quote(error)}")
    if not code or state not in oauth_states: raise HTTPException(400,"Invalid OAuth callback")
    oauth_states.discard(state)
    try:
        response=upstox.session.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"accept":"application/json","Content-Type":"application/x-www-form-urlencoded"},
            data={"code":code,"client_id":settings.upstox_client_id,"client_secret":settings.upstox_client_secret,
                  "redirect_uri":settings.upstox_redirect_uri,"grant_type":"authorization_code"},
            timeout=25,
        )
        response.raise_for_status()
        token=response.json().get("access_token")
        if not token: raise ValueError("No access token returned")
        upstox.set_access_token(token)
        try: TOKEN_FILE.write_text(json.dumps({"access_token":token,"issued_at":datetime.now(ZoneInfo(IST)).isoformat()}),encoding="utf-8")
        except Exception: pass
        live.start()
        return RedirectResponse("/?connected=1")
    except Exception as exc:
        return RedirectResponse(f"/?auth_error={quote(str(exc))}")

@app.get("/api/config")
def config():
    return {
        "app_name":settings.app_name,
        "browser_poll_seconds":settings.live_poll_seconds,
        "prediction_refresh_seconds":settings.prediction_refresh_seconds,
        "model_retrain_minutes":settings.model_retrain_minutes,
        "news_refresh_minutes":settings.news_refresh_minutes,
        "toi_refresh_minutes":settings.toi_refresh_minutes,
        "moneycontrol_refresh_minutes":settings.moneycontrol_refresh_minutes,
        "supports":{"timeframes":["1","5","10","15","30","60","1d"]},
    }
