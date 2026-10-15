# Radhe Stock Market Predictor

A live India-market analytics application built around Upstox V3 market data and a server-side WebSocket stream. It is intentionally focused on the trading workflow: live price, current candle, intraday prediction, selective-signal validation, and news context.

## What changed

- Yahoo/stale daily-only inference is removed from the live path.
- Upstox Analytics Token is supported through `UPSTOX_ACCESS_TOKEN`.
- Upstox V3 WebSocket drives the live quote and rolling 1-minute bars; V3 intraday candles are the REST recovery path. The UI polls the server state every second.
- Supported prediction windows: 1m, 5m, 10m, 15m, 30m, 60m.
- Prediction inference is recalculated every 30 seconds while the screen itself refreshes every second.
- The current candle is labeled with exact start/end, timeframe, and seconds remaining; 5m/10m/15m/30m bars are session-anchored to 09:15 IST.
- Historical training uses completed sessions only, then live inference uses the current forming candle.
- Index context includes NIFTY 50, NIFTY BANK and INDIA VIX, with a small adaptive live-regime blend measured on the same validation window as the signal gate.
- SENSEX is treated as a first-class live instrument (`BSE_INDEX|SENSEX`).
- The 90% gate is now an actual measured selective-precision gate. When it is not empirically satisfied, the app says `REVALIDATE` rather than pretending a BUY/SELL is reliable.
- FreeNewsApi.io is used via `x-api-key`. The app enforces a daily 5,000-call budget and a 2 req/s-safe client pacing rule.
- TrustGraph, Repository Health and Signal Journal are removed from this trading-focused build.

## Important behavior

A one-second screen refresh does **not** mean retraining the model every second. Retraining every second would add latency and amplify noise. The screen follows every live tick; the ML inference is recalculated on a slower cadence, and the model itself is retrained only on the configured model-retrain schedule or when you press **RETRAIN MODEL**.

The system does not guarantee profits or a perfect prediction. The gate is deliberately allowed to return `REVALIDATE` when the current evidence is not strong enough.

## Local setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Put the Upstox Analytics Token in `UPSTOX_ACCESS_TOKEN` and the FreeNewsApi key in `NEWS_API_KEY`.

## Render

1. Push this folder to GitHub.
2. Create a Render Web Service or use `render.yaml`.
3. Set `UPSTOX_ACCESS_TOKEN` and `NEWS_API_KEY` as Render environment variables.
4. Do not commit `.env` or hard-code credentials into frontend code.

## Upstox data design

Upstox V3 supports intraday minute intervals from 1 through 300 and the market WebSocket provides live updates. The app uses the WebSocket for the live quote and local current-candle construction, while V3 historical/intraday candle APIs supply training and recovery data.

## News budget

The 5,000-call quota is treated as a **hard daily ceiling**, not a target to burn. Requests are cached and paced at 0.51s minimum spacing so the published 2 req/s provider ceiling is respected. The current news layer is cached and paced to stay within the provider's published 2 requests/second limit, with a 5,000-request daily ceiling. The dashboard shows today's used and remaining budget.

## Sources

- Upstox Analytics Token: https://upstox.com/developer/api-documentation/analytics-token/
- Upstox Market Data Feed V3: https://upstox.com/developer/api-documentation/v3/get-market-data-feed/
- Upstox Intraday Candle V3: https://upstox.com/developer/api-documentation/v3/get-intra-day-candle-data/
- Upstox Historical Candle V3: https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/
- FreeNewsApi.io: https://www.freenewsapi.io/docs/

## Disclaimer

This application is for analytical/paper-trading use. It does not place trades and does not guarantee future returns.
