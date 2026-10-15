from __future__ import annotations

import secrets
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import PRIORITY_STOCKS, settings
from app.data.news import NewsService
from app.data.upstox import LiveMarketHub, UpstoxClient, UpstoxError
from app.services.engine import AnalysisEngine

BASE_DIR=Path(__file__).resolve().parents[1]
FRONTEND_DIR=BASE_DIR/"frontend"
app=FastAPI(title=settings.app_name,version="2.0.0")
app.mount("/static",StaticFiles(directory=FRONTEND_DIR),name="static")

upstox=UpstoxClient(settings.upstox_access_token,cache_dir=str(BASE_DIR/"data_cache"))
live=LiveMarketHub(upstox)
news=NewsService(settings.news_api_key,max_records=12,state_dir=str(BASE_DIR/"data_cache"))
engine=AnalysisEngine(upstox,live,news)
oauth_states:set[str]=set()

@app.on_event("startup")
def startup_event():
    live.start()

@app.get("/",include_in_schema=False)
def index(): return FileResponse(FRONTEND_DIR/"index.html")

@app.get("/api/health")
def health():
    return {"ok":True,"app_name":settings.app_name,"upstox_connected":upstox.connected,"stream_connected":live.connected,"stream_error":live.last_error,"news":news.diagnostics(),"live_poll_seconds":settings.live_poll_seconds,"prediction_refresh_seconds":settings.prediction_refresh_seconds}

@app.get("/api/priority")
def priority(): return {"stocks":PRIORITY_STOCKS}

@app.get("/api/stocks")
def stocks(q:str=Query("",max_length=80),limit:int=Query(25,ge=1,le=50)):
    if not q.strip():
        out=[]
        for item in PRIORITY_STOCKS:
            try: ins=upstox.resolve_symbol(item["symbol"])
            except Exception: continue
            out.append({"name":ins.name,"symbol":ins.trading_symbol,"exchange":ins.exchange,"segment":ins.segment,"instrument_key":ins.instrument_key})
        return {"results":out}
    try: matches=upstox.search_instruments(q,limit=limit)
    except UpstoxError as exc: raise HTTPException(503,str(exc))
    return {"results":[{"name":x.name,"short_name":x.short_name,"symbol":x.trading_symbol,"exchange":x.exchange,"segment":x.segment,"instrument_key":x.instrument_key} for x in matches]}

@app.get("/api/universe")
def universe():
    try:
        items=upstox.load_instruments(); return {"total_equities":len(items),"nse":sum(x.exchange=="NSE" for x in items),"bse":sum(x.exchange=="BSE" for x in items)}
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/market-pulse")
def market_pulse(limit:int=Query(50,ge=10,le=200)):
    # Keep this dashboard endpoint inexpensive: priority names are always visible;
    # the searchable universe remains available through /api/stocks.
    try:
        resolved=[]
        for item in PRIORITY_STOCKS:
            try: resolved.append(upstox.resolve_symbol(item["symbol"]))
            except Exception: pass
        unique={x.instrument_key:x for x in resolved}
        quotes=upstox.get_quote_batch(list(unique))
    except Exception as exc: raise HTTPException(503,str(exc))
    rows=[]; by_key={x.instrument_key:x for x in unique.values()}
    for key,q in quotes.items():
        token=q.get("instrument_token") or key
        ins=by_key.get(token)
        ohlc=q.get("ohlc") or {}; last=q.get("last_price") or ohlc.get("close"); prev=q.get("prev_close_price"); net=q.get("net_change")
        if net is None and last is not None and prev: net=last-prev
        change_pct=(net/prev*100) if net is not None and prev else None
        if ins and last is not None and change_pct is not None:
            rows.append({"symbol":ins.trading_symbol,"name":ins.name,"exchange":ins.exchange,"instrument_key":ins.instrument_key,"price":last,"change":net,"change_pct":change_pct,"volume":q.get("volume") or ohlc.get("volume")})
    rows.sort(key=lambda r:r["change_pct"],reverse=True); return {"count":len(rows),"results":rows[:limit]}

@app.get("/api/analyze")
def analyze(symbol:str=Query(...,min_length=1,max_length=100),interval:int=Query(5,ge=1,le=300),retrain:bool=False):
    try:
        ins=upstox.resolve_symbol(symbol)
        return engine.analyze(ins,interval=interval,force_retrain=retrain)
    except UpstoxError as exc: raise HTTPException(503,str(exc))
    except ValueError as exc: raise HTTPException(422,str(exc))
    except Exception as exc: raise HTTPException(500,f"Analysis failed: {exc}")

@app.get("/api/live")
def live_state(symbol:str=Query(...,min_length=1,max_length=100),interval:int=Query(5,ge=1,le=300)):
    try:
        ins=upstox.resolve_symbol(symbol)
        return engine.live_state(ins,interval=interval)
    except Exception as exc: raise HTTPException(503,str(exc))

@app.get("/api/news/quota")
def news_quota(): return news.diagnostics()

@app.get("/auth/upstox/start")
def upstox_start():
    if not settings.upstox_client_id or not settings.upstox_redirect_uri: raise HTTPException(503,"Set UPSTOX_CLIENT_ID and UPSTOX_REDIRECT_URI first.")
    state=secrets.token_urlsafe(24); oauth_states.add(state)
    params={"response_type":"code","client_id":settings.upstox_client_id,"redirect_uri":settings.upstox_redirect_uri,"state":state}
    from urllib.parse import urlencode
    return RedirectResponse(f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}")

@app.get("/auth/upstox/callback")
def upstox_callback(code:Optional[str]=None,state:Optional[str]=None,error:Optional[str]=None):
    if error: return RedirectResponse(f"/?auth_error={quote(error)}")
    if not code or state not in oauth_states: raise HTTPException(400,"Invalid OAuth callback")
    oauth_states.discard(state)
    try:
        # OAuth remains supported for deployments that prefer the standard short-lived token.
        response=upstox.session.post("https://api.upstox.com/v2/login/authorization/token",headers={"accept":"application/json","Content-Type":"application/x-www-form-urlencoded"},data={"code":code,"client_id":settings.upstox_client_id,"client_secret":settings.upstox_client_secret,"redirect_uri":settings.upstox_redirect_uri,"grant_type":"authorization_code"},timeout=25)
        response.raise_for_status(); token=response.json().get("access_token")
        if not token: raise ValueError("No access token returned")
        upstox.set_access_token(token); live.start()
        return RedirectResponse("/?connected=1")
    except Exception as exc: return RedirectResponse(f"/?auth_error={quote(str(exc))}")

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app.main:app",host="0.0.0.0",port=8000,reload=True)
