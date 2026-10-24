from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from app.config import IST
from app.data.upstox import LiveMarketHub, UpstoxClient

def test_feed_candle_authoritative_not_flattened_by_ticks():
    hub = LiveMarketHub(UpstoxClient(""))
    key = "NSE_INDEX|Nifty 50"
    ts = datetime(2026, 10, 24, 10, 1, 0, tzinfo=ZoneInfo(IST))

    # Real 10:01 candle: O104, H105, L104, C104.5
    hub._upsert(key, {
        "timestamp": ts,
        "open": 104.0,
        "high": 105.0,
        "low": 104.0,
        "close": 104.5,
        "volume": 500.0,
    }, is_authoritative=True)

    # Subsequent tick arrives at 10:01 with ltp=104.5
    hub._upsert(key, {
        "timestamp": ts,
        "open": 104.5,
        "high": 104.5,
        "low": 104.5,
        "close": 104.5,
        "volume": 0.0,
    }, is_authoritative=False)

    bars = hub.live_bars(key, interval=1)
    assert len(bars) == 1
    row = bars.iloc[0]
    # Candle must NOT be flattened to flat 104.5
    assert row["open"] == 104.0
    assert row["high"] == 105.0
    assert row["low"] == 104.0
    assert row["close"] == 104.5
    assert row["volume"] == 500.0

def test_out_of_order_candles_preserved():
    hub = LiveMarketHub(UpstoxClient(""))
    key = "NSE_INDEX|Nifty 50"
    ts1 = datetime(2026, 10, 24, 10, 0, 0, tzinfo=ZoneInfo(IST))
    ts2 = datetime(2026, 10, 24, 10, 1, 0, tzinfo=ZoneInfo(IST))

    # 10:01 arrives first
    hub._upsert(key, {"timestamp": ts2, "open": 102.0, "high": 103.0, "low": 101.5, "close": 102.5, "volume": 100.0}, is_authoritative=True)
    # 10:00 arrives second
    hub._upsert(key, {"timestamp": ts1, "open": 100.0, "high": 102.0, "low": 99.5, "close": 101.8, "volume": 150.0}, is_authoritative=True)

    bars = hub.live_bars(key, interval=1)
    assert len(bars) == 2
    assert bars.iloc[0]["timestamp"] == pd.Timestamp(ts1)
    assert bars.iloc[1]["timestamp"] == pd.Timestamp(ts2)
    assert bars.iloc[0]["open"] == 100.0
    assert bars.iloc[1]["open"] == 102.0

def test_resolve_symbol_prefers_nse():
    client = UpstoxClient("")
    # Mock instruments with RELIANCE in both BSE and NSE
    from app.data.upstox import Instrument
    client._instruments = [
        Instrument(instrument_key="BSE_EQ|INE002A01018", exchange="BSE", segment="BSE_EQ",
                   instrument_type="EQ", name="RELIANCE INDUSTRIES LTD", trading_symbol="RELIANCE", short_name="RELIANCE"),
        Instrument(instrument_key="NSE_EQ|INE002A01018", exchange="NSE", segment="NSE_EQ",
                   instrument_type="EQ", name="RELIANCE INDUSTRIES LTD", trading_symbol="RELIANCE", short_name="RELIANCE"),
    ]
    ins = client.resolve_symbol("RELIANCE")
    assert ins.exchange == "NSE", f"Expected NSE listing for RELIANCE, got {ins.exchange}"
