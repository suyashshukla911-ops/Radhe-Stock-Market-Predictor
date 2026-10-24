import pytest
from fastapi.testclient import TestClient
from app.main import app, news, stock_forecaster, upstox
from app.data.upstox import Instrument
from app.services.stock_forecast import StockForecast

@pytest.fixture
def client():
    return TestClient(app)

def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["app_name"] == "Suyash Stock Market Predictor"

def test_api_news_download_no_type_error(client, tmp_path, monkeypatch):
    # Mock excel export
    mock_file = tmp_path / "toi_market_news.xlsx"
    mock_file.write_text("dummy excel content")
    monkeypatch.setattr(news, "export_toi_excel", lambda: mock_file)

    r = client.get("/api/news/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def test_api_analyze_non_index_raises_422(client, monkeypatch):
    # Mock resolve_symbol to return an equity instrument (not INDEX)
    eq_ins = Instrument(
        instrument_key="NSE_EQ|INE002A01018", exchange="NSE", segment="NSE_EQ",
        instrument_type="EQ", name="RELIANCE INDUSTRIES LTD", trading_symbol="RELIANCE", short_name="RELIANCE"
    )
    monkeypatch.setattr(upstox, "resolve_symbol", lambda sym: eq_ins)

    # Must return 422 Unprocessable Entity, NOT 500 Internal Server Error
    r = client.get("/api/analyze?symbol=RELIANCE&interval=5")
    assert r.status_code == 422
    assert "Intraday model dashboard is reserved for SENSEX and NIFTY 50" in r.json()["detail"]

def test_api_stock_forecast_cache_consistency(client, monkeypatch):
    eq_ins = Instrument(
        instrument_key="NSE_EQ|INE002A01018", exchange="NSE", segment="NSE_EQ",
        instrument_type="EQ", name="RELIANCE INDUSTRIES LTD", trading_symbol="RELIANCE", short_name="RELIANCE"
    )
    monkeypatch.setattr(upstox, "resolve_symbol", lambda sym: eq_ins)

    # Mock stock forecaster
    forecast_obj = StockForecast(
        symbol="RELIANCE", company="Reliance Industries", current_price=1300.0,
        next_day_close=1310.0, next_day_high=1315.0, next_day_low=1295.0,
        up_probability=0.62, signal="UP", news_impact=0.1, training_rows=150,
        generated_at="2026-10-24T10:00:00+05:30"
    )
    result_dict = forecast_obj.__dict__ | {"validation_accuracy": 68.5, "next_session_date": "2026-10-27"}
    stock_forecaster.cache[eq_ins.instrument_key] = (10000000000.0, result_dict)

    # First call
    r1 = client.get("/api/stock-forecast?symbol=RELIANCE")
    assert r1.status_code == 200
    d1 = r1.json()
    assert "validation_accuracy" in d1
    assert "next_session_date" in d1
    assert d1["validation_accuracy"] == 68.5
    assert d1["next_session_date"] == "2026-10-27"

    # Second call (from cache)
    r2 = client.get("/api/stock-forecast?symbol=RELIANCE")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2 == d1
