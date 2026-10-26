import time
from datetime import date, datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from app.config import IST, NSE_HOLIDAYS_SET
from app.data.news import NewsService, lexical_sentiment
from app.data.upstox import Instrument, LiveMarketHub, UpstoxClient
from app.ml.features import FEATURES, add_targets, build_features, prepare_training_frame
from app.services.engine import AnalysisEngine
from app.services.stock_forecast import StockForecastService
from tests.test_engine import mock_candles

INS = Instrument("BSE_INDEX|SENSEX", "BSE", "BSE_INDEX", "INDEX", "SENSEX", "SENSEX", "SENSEX")

def _engine(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    up = UpstoxClient("")
    return up, AnalysisEngine(up, LiveMarketHub(up), NewsService("", state_dir=str(tmp_path)))

def test_only_vix_context_missing_still_trains(tmp_path, monkeypatch):
    up, eng = _engine(tmp_path, monkeypatch)
    hist = mock_candles(60)
    monkeypatch.setattr(up, "get_recent_history", lambda k, i, d: hist.copy())
    ctx = hist.copy()
    monkeypatch.setattr(eng, "_get_context_history", lambda i, d, force=False: {"nifty": ctx, "banknifty": ctx, "vix": pd.DataFrame()})
    assert eng._get_training_cache(INS, 5, force=True).model is not None

def test_build_features_always_has_context_columns():
    out = build_features(mock_candles(10), context={"nifty": mock_candles(10)})
    assert {"nifty_return", "banknifty_return", "vix_return"} <= set(out.columns)

def test_t2_target_never_crosses_session():
    df = mock_candles(12)
    t = add_targets(build_features(df))
    day = t.timestamp.dt.date
    second_last = t.groupby(day).tail(2).groupby(day).head(1)
    assert second_last["next_2_return"].isna().all()

def test_holidays_2026_are_correct():
    assert "2026-02-16" not in NSE_HOLIDAYS_SET and "2026-09-04" not in NSE_HOLIDAYS_SET
    for d in ("2026-03-31", "2026-06-26", "2026-09-14", "2026-11-10"):
        assert d in NSE_HOLIDAYS_SET
    mon_open = datetime(2026, 2, 16, 10, 0, tzinfo=ZoneInfo(IST))
    ganesh = datetime(2026, 9, 14, 10, 0, tzinfo=ZoneInfo(IST))
    assert UpstoxClient.is_market_open(mon_open) and not UpstoxClient.is_market_open(ganesh)

def test_target_session_logic():
    f = StockForecastService._next_trading_session
    ist = ZoneInfo(IST)
    assert f(date(2026, 10, 26), now=datetime(2026, 10, 26, 8, 0, tzinfo=ist)) == date(2026, 10, 26)   # before open: today
    assert f(date(2026, 10, 26), now=datetime(2026, 10, 26, 11, 0, tzinfo=ist)) == date(2026, 10, 26)  # intraday: today's close
    assert f(date(2026, 10, 26), now=datetime(2026, 10, 26, 16, 0, tzinfo=ist)) == date(2026, 10, 27)  # after close: tomorrow
    assert f(date(2026, 11, 9), now=datetime(2026, 11, 9, 16, 0, tzinfo=ist)) == date(2026, 11, 11)    # skips 10 Nov holiday
    assert f(date(2026, 10, 23), now=datetime(2026, 10, 23, 16, 0, tzinfo=ist)) == date(2026, 10, 26)  # skips weekend

@pytest.mark.parametrize("text,sign", [
    ("Sensex ends lower as banks drag", -1), ("Nifty ends higher led by IT stocks", 1),
    ("Sensex tanks 800 points on FII selling", -1), ("Nifty slips below 25,000", -1),
    ("Sensex, Nifty snap four-day losing streak", 1), ("Inflation cools to 4-month low", 1),
    ("FIIs turn net sellers, pull out Rs 3,000 crore", -1), ("Crude oil prices rise, rupee weakens", -1),
    ("Bank NPA rise worries investors", -1),
])
def test_market_headline_sentiment(text, sign):
    assert lexical_sentiment(text) * sign > 0.15, (text, lexical_sentiment(text))

def test_model_for_request_does_not_block_and_then_serves(tmp_path, monkeypatch):
    up, eng = _engine(tmp_path, monkeypatch)
    hist = mock_candles(60)
    monkeypatch.setattr(up, "get_recent_history", lambda k, i, d: hist.copy())
    monkeypatch.setattr(eng, "_get_context_history", lambda i, d, force=False: {})
    t0 = time.time()
    with pytest.raises(ValueError):
        eng._model_for_request(INS, 5)
    assert time.time() - t0 < 2
    for _ in range(120):
        if (INS.instrument_key, 5) in eng.models: break
        time.sleep(0.5)
    assert eng._model_for_request(INS, 5).model is not None

def test_market_clock_status():
    ist = ZoneInfo(IST)
    # Weekend
    sat = datetime(2026, 10, 24, 11, 0, tzinfo=ist)
    assert UpstoxClient.market_clock_status(sat)["market_status"] == "WEEKEND"
    # Holiday (Ganesh Chaturthi 2026-09-14)
    ganesh = datetime(2026, 9, 14, 11, 0, tzinfo=ist)
    assert UpstoxClient.market_clock_status(ganesh)["market_status"] == "HOLIDAY"
    # Pre-market Monday 08:30 IST
    mon_pre = datetime(2026, 10, 26, 8, 30, tzinfo=ist)
    assert UpstoxClient.market_clock_status(mon_pre)["market_status"] == "PRE_OPEN"
    # Intraday open Monday 10:15 IST
    mon_open = datetime(2026, 10, 26, 10, 15, tzinfo=ist)
    assert UpstoxClient.market_clock_status(mon_open)["market_status"] == "OPEN"
    assert UpstoxClient.market_clock_status(mon_open)["is_open"] is True
    # Post-market close Monday 16:00 IST
    mon_close = datetime(2026, 10, 26, 16, 0, tzinfo=ist)
    assert UpstoxClient.market_clock_status(mon_close)["market_status"] == "CLOSED"

def test_to_df_empty_has_datetime_dtype():
    df = UpstoxClient._to_df([])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # .dt.date must not raise TypeError or AttributeError
    dates = df["timestamp"].dt.date
    assert len(dates) == 0

def test_get_historical_handles_empty_chunk(monkeypatch):
    up = UpstoxClient("test_token")
    hist = mock_candles(2)
    # Return valid candles for chunk 1, empty for chunk 2
    calls = []
    def mock_get_json(url, params=None):
        calls.append(url)
        if len(calls) == 1:
            raw_candles = [[r.timestamp.isoformat(), r.open, r.high, r.low, r.close, r.volume, 0.0] for r in hist.itertuples()]
            return {"data": {"candles": raw_candles}}
        return {"data": {"candles": []}}
    monkeypatch.setattr(up, "_get_json", mock_get_json)
    res = up.get_historical("BSE_INDEX|SENSEX", 1, date(2026, 9, 25), date(2026, 10, 26))
    assert not res.empty
    assert pd.api.types.is_datetime64_any_dtype(res["timestamp"])
    assert len(res.timestamp.dt.date) > 0

def test_live_market_hub_restart():
    up = UpstoxClient("old_token")
    hub = LiveMarketHub(up)
    hub._connected = True
    hub._started = True
    hub.restart("new_token")
    assert up.access_token == "new_token"
    assert hub._run_gen == 1
    assert hub._connected is False

def test_live_endpoint_returns_market_status_and_data_status(monkeypatch):
    from starlette.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    # Even if resolve_symbol or Upstox raises an error, live endpoint should not return UNKNOWN market_status
    r = client.get("/api/live?symbol=SENSEX&interval=5")
    assert r.status_code == 200
    data = r.json()
    assert "market_data" in data
    mkt = data["market_data"]
    assert mkt["market_status"] in {"OPEN", "PRE_OPEN", "CLOSED", "WEEKEND", "HOLIDAY"}
    assert "data_status" in mkt
    assert mkt["data_status"] in {"OK", "NO_TOKEN", "TOKEN_EXPIRED", "FEED_DOWN", "ERROR"}

