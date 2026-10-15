# Radhe Stock Market Predictor

## Stability hotfix (3.1)

- `frontend/styles.css` is included and served at `/static/styles.css`.
- `/api/live` returns a structured dashboard response instead of surfacing temporary provider failures as HTTP 503.
- After market close, the live-state path falls back from WebSocket/current-day intraday data to recent historical candles, so the last real session remains visible.
- Last-price fallback uses the latest market quote when candle history is temporarily unavailable.
- Browser news refresh follows the configured 30-minute cadence instead of a fixed 5-minute loop.


A stability-first Indian market analytics application centered on Upstox V3 live data, intraday candles, a calibrated ML model, Times of India headlines, FreeNewsApi, and optional X/Twitter context.

## Operating policy

- Browser/live dashboard refresh: **60 seconds**, not every second.
- Prediction/inference refresh: **5 minutes**.
- Model retraining: **30 minutes**.
- FreeNewsApi / X refresh: **30 minutes**.
- Times of India category scraper: **10 minutes**.
- Upstox WebSocket: server-side continuous stream; browser polling is deliberately decoupled.
- When the cash market is closed, the application shows the **last real market price and the final completed intraday candle from the most recent real cash session** (today when today's session data is available). It never invents an after-hours candle.
- The `1D` selector is a dedicated **previous-session history view**; it does not act as an intraday prediction mode.
- Next-candle close/high/low and prediction probability are shown **only while the cash market is open**. After close, the UI shows the actual last-candle close/high/low instead.
- BUY/SELL is never forced. The prediction and validation state remain visible during market hours even when the statistical gate says `REVALIDATE`.

## Data architecture

1. Upstox instrument master keeps the complete NSE/BSE equity directory searchable. Securities remain visible even when live history/quotes are unavailable.
2. Upstox V3 1-minute historical candles are the base training source. Requested 5/10/15/30/60-minute bars are deterministically aggregated locally from 1-minute source data, anchored at 09:15 IST.
3. Upstox V3 MarketDataStreamerV3 is the live source. Trade timestamps (`ltt`) are used; heartbeat/current timestamps are never converted into fake candles.
4. Training excludes incomplete sessions. Index instruments often have zero volume, so feature engineering uses a price-based VWAP fallback rather than dropping every row.
5. The model uses technical structure plus NIFTY 50 / NIFTY BANK / INDIA VIX context.
6. News consists of:
   - Times of India category pages using the supplied scraper logic.
   - FreeNewsApi (if `NEWS_API_KEY` is configured).
   - Optional X/Twitter recent-search context (`X_BEARER_TOKEN`).
7. News is ranked for relevance to the selected instrument and Indian market before being shown to the user or applied as a small model adjustment.

## Why this is stable

The earlier version coupled a 1-second browser loop to prediction and data calls. That created repeated REST traffic and made model failures look like live-market failures. The final version keeps the WebSocket server-side, while the browser only asks for state once per minute. The model and external news sources run on materially slower cadences.

## Prediction interpretation

For a selected timeframe:

- `NEXT CLOSE/HIGH/LOW` = the model's forecast for the next bar at that selected timeframe.
- `END-OF-DAY` = the model's same-session EOD scenario.
- Switch the selector between 5m and 10m to inspect the corresponding next-bar forecast.
- The validation gate is based on walk-forward validation and a Wilson lower bound, not just a raw point estimate.

A model cannot be guaranteed to make money or perfectly predict price. The application is intentionally designed to expose a forecast while refusing to label an empirically weak forecast as a trusted BUY/SELL.

## Render environment

Set:
- `UPSTOX_ACCESS_TOKEN`
- `NEWS_API_KEY`
- `APP_BASE_URL`

Optional:
- `UPSTOX_CLIENT_ID`, `UPSTOX_CLIENT_SECRET`, `UPSTOX_REDIRECT_URI`
- `X_BEARER_TOKEN`

Never commit credentials to Git.

## Relevant upstream API constraints

Upstox V3 provides minute intervals and has documented request-rate limits; this project therefore avoids second-by-second historical/REST calls and uses the WebSocket for live state. See the linked developer documentation in the project notes.

## Local run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

## Security

Rotate any API credential that has been exposed in chat, screenshots, source control, or logs.
