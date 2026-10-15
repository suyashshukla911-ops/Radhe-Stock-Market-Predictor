from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import PRIORITY_STOCKS, settings
from app.data.news import NewsService
from app.data.upstox import LiveMarketHub, UpstoxClient, UpstoxError
from app.services.engine import AnalysisEngine

BASE_DIR=Path(__file__).resolve().parents[1]
FRONTEND_DIR=BASE_DIR/"frontend"

app=FastAPI(title=settings.app_name, version="3.0.0")
app.mount("/static",StaticFiles(directory=FRONTEND_DIR),name="static")

upstox=UpstoxClient(settings.upstox_access_token,cache_dir=str(BASE_DIR/"data_cache"))
live=LiveMarketHub(upstox)
news=NewsService(settings.news_api_key,max_records=20,state_dir=str(BASE_DIR/"data_cache"))
engine=AnalysisEngine(upstox,live,news)
oauth_states:set[str]=set()

@app.on_event("startup")
def startup_event():
    live.start()
    news.start()

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
def universe():
    try: return upstox.universe_summary()
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

@app.get("/api/news/quota")
def news_quota(): return news.diagnostics()

@app.get("/api/analyze")
def analyze(symbol:str=Query(...,min_length=1,max_length=100),interval:int=Query(5,ge=1,le=60),retrain:bool=False):
    if interval not in {1,5,10,15,30,60}: raise HTTPException(422,"Supported timeframes: 1,5,10,15,30,60 minutes.")
    try:
        ins=upstox.resolve_symbol(symbol)
        return engine.analyze(ins,interval=interval,force_retrain=retrain)
    except UpstoxError as exc: raise HTTPException(503,str(exc))
    except ValueError as exc: raise HTTPException(422,str(exc))
    except Exception as exc: raise HTTPException(500,f"Analysis failed: {exc}")

@app.get("/api/live")
def live_state(symbol:str=Query(...,min_length=1,max_length=100),interval:int=Query(5,ge=1,le=60)):
    try:
        ins=upstox.resolve_symbol(symbol)
        return engine.live_state(ins,interval)
    except Exception as exc:
        # Live state is a dashboard endpoint: never turn a temporary provider
        # problem into a blank/erroring page. Return a structured 200 response
        # so the UI can keep the last known state and retry on its normal cadence.
        return {
            "now": __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("Asia/Kolkata")).isoformat(),
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
        "supports":{"timeframes":[1,5,10,15,30,60]},
    }
