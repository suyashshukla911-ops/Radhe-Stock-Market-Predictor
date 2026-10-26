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

def _resolve_cache_dir() -> Path:
    candidates = [
        os.getenv("CACHE_DIR"),
        os.getenv("DATA_DIR"),
        str(ROOT_DIR / "data_cache"),
    ]
    for c in candidates:
        if not c:
            continue
        try:
            p = Path(c).resolve()
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            test_file.touch(exist_ok=True)
            test_file.unlink(missing_ok=True)
            return p
        except (OSError, PermissionError):
            continue
    fallback = ROOT_DIR / "data_cache"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

CACHE_DIR = _resolve_cache_dir()

# NSE cash-market weekday holidays, 2026 (verify yearly against the NSE circular).
NSE_HOLIDAYS_SET = {
    "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26", "2026-09-14",
    "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25",
}

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
    {"name": "Bharti Airtel", "symbol": "BHARTIARTL", "aliases": ["BHARTI AIRTEL", "BHARTIARTL"]},
    {"name": "ITC", "symbol": "ITC", "aliases": ["ITC"]},
    {"name": "Larsen & Toubro", "symbol": "LT", "aliases": ["LT", "LARSEN"]},
    {"name": "Axis Bank", "symbol": "AXISBANK", "aliases": ["AXIS BANK", "AXISBANK"]},
    {"name": "Tata Motors Passenger Vehicles", "symbol": "TMPV", "aliases": ["TMPV", "TATA MOTORS PASSENGER VEHICLES"]},
    {"name": "Tata Motors (Commercial Vehicles)", "symbol": "TMCV", "aliases": ["TMCV"]},
    {"name": "Adani Enterprises", "symbol": "ADANIENT", "aliases": ["ADANI ENTERPRISES", "ADANIENT"]},
    {"name": "Sun Pharma", "symbol": "SUNPHARMA", "aliases": ["SUN PHARMA", "SUNPHARMA"]},
    {"name": "Wipro", "symbol": "WIPRO", "aliases": ["WIPRO"]},
]

SECTOR_BY_SYMBOL = {
    "RELIANCE": "Energy / Telecom", "TCS": "IT", "INFY": "IT", "HDFCBANK": "Banking",
    "ICICIBANK": "Banking", "SBIN": "Banking", "BHARTIARTL": "Telecom", "ITC": "FMCG",
    "LT": "Infrastructure", "AXISBANK": "Banking", "TATAMOTORS": "Automotive",
    "TMPV": "Automotive", "TMCV": "Automotive",
    "ADANIENT": "Conglomerate", "SUNPHARMA": "Pharma", "WIPRO": "IT",
}

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
    upstox_access_token: str
    upstox_client_id: str
    upstox_client_secret: str
    upstox_redirect_uri: str

    news_api_key: str
    news_provider: str
    news_daily_quota: int
    news_min_interval_seconds: float
    news_refresh_minutes: int
    toi_refresh_minutes: int
    x_refresh_minutes: int
    x_bearer_token: str
    moneycontrol_refresh_minutes: int

    live_poll_seconds: int
    prediction_refresh_seconds: int
    watch_refresh_seconds: int
    model_retrain_minutes: int

    train_days_short: int
    train_days_long: int
    min_train_rows: int
    chart_bars: int
    history_cache_minutes: int
    live_rest_cache_seconds: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "Suyash Stock Market Predictor"),
            base_url=os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/"),
            upstox_access_token=os.getenv("UPSTOX_ACCESS_TOKEN", "").strip(),
            upstox_client_id=os.getenv("UPSTOX_CLIENT_ID", "").strip(),
            upstox_client_secret=os.getenv("UPSTOX_CLIENT_SECRET", "").strip(),
            upstox_redirect_uri=os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/auth/upstox/callback").strip(),
            news_api_key=os.getenv("NEWS_API_KEY", "").strip(),
            news_provider=os.getenv("NEWS_PROVIDER", "freenewsapi"),
            news_daily_quota=int(os.getenv("NEWS_DAILY_QUOTA", "5000")),
            news_min_interval_seconds=float(os.getenv("NEWS_MIN_INTERVAL_SECONDS", "4.5")),
            news_refresh_minutes=int(os.getenv("NEWS_REFRESH_MINUTES", "30")),
            toi_refresh_minutes=int(os.getenv("TOI_REFRESH_MINUTES", "10")),
            x_refresh_minutes=int(os.getenv("X_REFRESH_MINUTES", "30")),
            x_bearer_token=os.getenv("X_BEARER_TOKEN", "").strip(),
            moneycontrol_refresh_minutes=int(os.getenv("MONEYCONTROL_REFRESH_MINUTES", "30")),
            live_poll_seconds=int(os.getenv("LIVE_POLL_SECONDS", "5")),
            prediction_refresh_seconds=int(os.getenv("PREDICTION_REFRESH_SECONDS", "20")),
            watch_refresh_seconds=int(os.getenv("WATCH_REFRESH_SECONDS", "300")),
            model_retrain_minutes=int(os.getenv("MODEL_RETRAIN_MINUTES", "30")),
            train_days_short=int(os.getenv("TRAIN_DAYS_SHORT", "30")),
            train_days_long=int(os.getenv("TRAIN_DAYS_LONG", "90")),
            min_train_rows=int(os.getenv("MIN_TRAIN_ROWS", "300")),
            chart_bars=int(os.getenv("CHART_BARS", "180")),
            history_cache_minutes=int(os.getenv("HISTORY_CACHE_MINUTES", "20")),
            live_rest_cache_seconds=int(os.getenv("LIVE_REST_CACHE_SECONDS", "60")),
        )

settings = Settings.from_env()
