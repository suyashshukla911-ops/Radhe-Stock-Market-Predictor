from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

IST = "Asia/Kolkata"
MARKET_OPEN_MINUTES = 9 * 60 + 15
MARKET_CLOSE_MINUTES = 15 * 60 + 30

PRIORITY_STOCKS = [
    {"name": "SENSEX", "symbol": "SENSEX", "aliases": ["SENSEX", "BSE SENSEX"]},
    {"name": "NIFTY 50", "symbol": "NIFTY 50", "aliases": ["NIFTY 50", "NIFTY", "NSE NIFTY"]},
    {"name": "NIFTY BANK", "symbol": "Nifty Bank", "aliases": ["NIFTY BANK", "BANKNIFTY", "NIFTY BANK INDEX"]},
    {"name": "INDIA VIX", "symbol": "India VIX", "aliases": ["INDIA VIX", "VIX"]},
    {"name": "Reliance Industries", "symbol": "RELIANCE", "aliases": ["RELIANCE", "RELIANCE INDUSTRIES"]},
    {"name": "Tata Consultancy Services", "symbol": "TCS", "aliases": ["TCS", "TATA CONSULTANCY SERVICES"]},
    {"name": "Infosys", "symbol": "INFY", "aliases": ["INFY", "INFOSYS"]},
    {"name": "HDFC Bank", "symbol": "HDFCBANK", "aliases": ["HDFC BANK", "HDFCBANK"]},
    {"name": "ICICI Bank", "symbol": "ICICIBANK", "aliases": ["ICICI BANK", "ICICIBANK"]},
    {"name": "State Bank of India", "symbol": "SBIN", "aliases": ["SBI", "SBIN", "STATE BANK OF INDIA"]},
]

SECTOR_BY_SYMBOL = {
    "RELIANCE": "Energy / Consumer",
    "TCS": "IT Services",
    "INFY": "IT Services",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "SBIN": "Banking",
}

# Current Upstox index keys used as a direct, deterministic fallback.
INDEX_INSTRUMENTS = {
    "SENSEX": {"instrument_key": "BSE_INDEX|SENSEX", "exchange": "BSE", "segment": "BSE_INDEX", "name": "SENSEX", "short_name": "SENSEX", "trading_symbol": "SENSEX"},
    "NIFTY 50": {"instrument_key": "NSE_INDEX|Nifty 50", "exchange": "NSE", "segment": "NSE_INDEX", "name": "NIFTY 50", "short_name": "NIFTY 50", "trading_symbol": "Nifty 50"},
    "NIFTY BANK": {"instrument_key": "NSE_INDEX|Nifty Bank", "exchange": "NSE", "segment": "NSE_INDEX", "name": "NIFTY BANK", "short_name": "NIFTY BANK", "trading_symbol": "Nifty Bank"},
    "INDIA VIX": {"instrument_key": "NSE_INDEX|India VIX", "exchange": "NSE", "segment": "NSE_INDEX", "name": "INDIA VIX", "short_name": "INDIA VIX", "trading_symbol": "India VIX"},
}

@dataclass(frozen=True)
class Settings:
    app_name: str
    base_url: str
    upstox_client_id: str
    upstox_client_secret: str
    upstox_redirect_uri: str
    upstox_access_token: str
    news_api_key: str
    news_provider: str
    news_daily_quota: int
    news_min_interval_seconds: float
    news_window_seconds: int
    prediction_refresh_seconds: int
    live_poll_seconds: int
    watch_refresh_seconds: int
    train_days_short: int
    train_days_long: int
    min_train_rows: int
    model_retrain_minutes: int
    chart_bars: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "Radhe Stock Market Predictor"),
            base_url=os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/"),
            upstox_client_id=os.getenv("UPSTOX_CLIENT_ID", ""),
            upstox_client_secret=os.getenv("UPSTOX_CLIENT_SECRET", ""),
            upstox_redirect_uri=os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/auth/upstox/callback"),
            upstox_access_token=os.getenv("UPSTOX_ACCESS_TOKEN", ""),
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            news_provider=os.getenv("NEWS_PROVIDER", "freenewsapi"),
            news_daily_quota=int(os.getenv("NEWS_DAILY_QUOTA", "5000")),
            news_min_interval_seconds=float(os.getenv("NEWS_MIN_INTERVAL_SECONDS", "4.5")),
            news_window_seconds=int(os.getenv("NEWS_WINDOW_SECONDS", "24")),
            prediction_refresh_seconds=int(os.getenv("PREDICTION_REFRESH_SECONDS", "30")),
            live_poll_seconds=int(os.getenv("LIVE_POLL_SECONDS", "1")),
            watch_refresh_seconds=int(os.getenv("WATCH_REFRESH_SECONDS", "15")),
            train_days_short=int(os.getenv("TRAIN_DAYS_SHORT", "30")),
            train_days_long=int(os.getenv("TRAIN_DAYS_LONG", "90")),
            min_train_rows=int(os.getenv("MIN_TRAIN_ROWS", "300")),
            model_retrain_minutes=int(os.getenv("MODEL_RETRAIN_MINUTES", "30")),
            chart_bars=int(os.getenv("CHART_BARS", "180")),
        )

settings = Settings.from_env()
