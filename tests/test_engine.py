import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from app.config import IST
from app.data.news import NewsService
from app.data.upstox import Instrument, LiveMarketHub, UpstoxClient
from app.services.engine import AnalysisEngine

def mock_candles(days_count=10):
    rows = []
    rng = np.random.default_rng(42)
    cur = 1000.0
    for d in pd.date_range("2026-08-03", periods=days_count, freq="D"):
        if d.weekday() >= 5: continue
        ts = pd.date_range(d.replace(hour=9, minute=15), periods=75, freq="5min", tz="Asia/Kolkata")
        for t in ts:
            ret = float(rng.normal(0, 0.002))
            op = cur
            cur = max(10.0, cur * (1 + ret))
            hi = max(op, cur) + 1.0
            lo = min(op, cur) - 1.0
            rows.append([t, op, hi, lo, cur, 1000.0, 0.0])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

def test_engine_init_has_rest_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    engine = AnalysisEngine(UpstoxClient(""), LiveMarketHub(UpstoxClient("")), NewsService("", state_dir=str(tmp_path)))
    assert hasattr(engine, "_rest_live_cache")
    assert isinstance(engine._rest_live_cache, dict)

def test_engine_session_state_save_and_prune(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    engine = AnalysisEngine(UpstoxClient(""), LiveMarketHub(UpstoxClient("")), NewsService("", state_dir=str(tmp_path)))
    
    # Store an entry
    engine._eod_cache["NSE_INDEX|Nifty 50|2026-10-24"] = {
        "predicted_eod_close": 24500.0,
        "eod_return": 0.005,
        "opening_reference_price": 24400.0,
        "news_impact": 0.1,
        "created_at": "2026-10-24T09:30:00+05:30",
    }
    # Save using the method called in analyze
    engine._save_session_state()
    assert (tmp_path / "eod_outlook.json").exists()

    # Load again in new engine instance
    engine2 = AnalysisEngine(UpstoxClient(""), LiveMarketHub(UpstoxClient("")), NewsService("", state_dir=str(tmp_path)))
    assert "NSE_INDEX|Nifty 50|2026-10-24" in engine2._eod_cache

def test_context_failure_does_not_raise_keyerror(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    upstox = UpstoxClient("")
    hub = LiveMarketHub(upstox)
    news = NewsService("", state_dir=str(tmp_path))
    engine = AnalysisEngine(upstox, hub, news)

    instrument = Instrument(
        instrument_key="BSE_INDEX|SENSEX", exchange="BSE", segment="BSE_INDEX",
        instrument_type="INDEX", name="SENSEX", trading_symbol="SENSEX", short_name="SENSEX"
    )

    # Mock history with valid DataFrame
    hist = mock_candles(10)
    monkeypatch.setattr(upstox, "get_recent_history", lambda k, interval, days: hist.copy())

    # Mock context history where one instrument returns empty DataFrame
    monkeypatch.setattr(engine, "_get_context_history", lambda interval, days, force=False: {
        "nifty": pd.DataFrame(),  # empty frame simulating download failure
        "banknifty": pd.DataFrame(),
        "vix": pd.DataFrame(),
    })

    # Must succeed and not crash with KeyError: 'timestamp'
    cache = engine._get_training_cache(instrument, 5, force=True)
    assert cache is not None
    assert cache.model is not None

def test_analyze_returns_top_level_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    upstox = UpstoxClient("")
    hub = LiveMarketHub(upstox)
    news = NewsService("", state_dir=str(tmp_path))
    engine = AnalysisEngine(upstox, hub, news)

    instrument = Instrument(
        instrument_key="BSE_INDEX|SENSEX", exchange="BSE", segment="BSE_INDEX",
        instrument_type="INDEX", name="SENSEX", trading_symbol="SENSEX", short_name="SENSEX"
    )

    # Force market open
    monkeypatch.setattr(engine, "_market_open", lambda: True)

    # Mock history and live
    candles = mock_candles(10)
    monkeypatch.setattr(upstox, "get_recent_history", lambda k, interval, days: candles.copy())
    monkeypatch.setattr(engine, "_live_snapshot", lambda ins, interval, ttl_seconds=None: candles.tail(20).copy())
    monkeypatch.setattr(hub, "latest", lambda key: {"price": 80000.0, "updated_at": datetime.now(ZoneInfo(IST)).isoformat()})

    engine._get_training_cache(instrument, 5, force=True)   # analyze() no longer trains inside the request
    res = engine.analyze(instrument, interval=5)
    assert "symbol" in res
    assert res["symbol"] == "SENSEX"
    assert res["instrument"]["symbol"] == "SENSEX"
    assert "forecast_candles" in res
    assert len(res["forecast_candles"]) == 2
