# Radhe Stock Market Predictor — Production Architecture

## 1. Design goal

The application is deliberately split into four cadences so a live trading screen does not hammer model, historical-data, or news endpoints:

| Layer | Cadence | Purpose |
|---|---:|---|
| Upstox WebSocket | continuous server-side | live trade price and current 1-minute state |
| Browser `/api/live` | 60 s | stable UI refresh |
| Prediction | 5 min | refresh the current forecast without retraining |
| Model retrain | 30 min | rebuild ML models from completed-session data |
| TOI scraper | 10 min | refresh top Indian news headlines |
| FreeNewsApi / X | 30 min | slower external-news context |
| Market pulse | 5 min | refresh selected liquid/priority quotes |

## 2. Market-data flow

`Upstox V3 WebSocket -> LiveMarketHub -> 1m bars -> local 5/10/15/30/60m aggregation -> FastAPI state`

The WebSocket uses the trade timestamp (`ltt`). A websocket `currentTs` heartbeat is never treated as a trade. Packets outside the 09:15–15:30 IST cash session do not manufacture a new candle.

When the WebSocket is unavailable, the application uses V3 intraday candle data as a recovery path, cached for one minute.

After 15:30 IST, intraday views show the final actual cash-session candle for the latest session with data (today when today's session has been published by Upstox). The `1D` view remains a previous-session history view. The application never creates a 15:30–16:30 or 16:00–16:15 fake candle.

## 3. Historical training

The V3 1-minute historical API is the canonical source for intraday training. For 5/10/15/30/60-minute models, the application retrieves 1-minute history and aggregates locally to the requested timeframe. This keeps candle boundaries consistent and avoids differences between provider aggregation and the application's session boundaries.

Training excludes the current incomplete session. Once 15:30 IST passes, that session becomes eligible for the next retrain.

Index feeds can have zero volume. Feature generation therefore has a price-based session VWAP fallback rather than discarding every row.

## 4. Model lifecycle

A model is cached per `(instrument, timeframe)`.

1. Fetch completed-session history.
2. Build features and next-bar/EOD targets.
3. Split by time/session, not random shuffle.
4. Calibrate directional probabilities.
5. Determine a selective threshold.
6. Evaluate precision, coverage, signal count, Brier score and Wilson lower bound.
7. Fit final models on all completed training rows.
8. Cache the model for 30 minutes.

A forecast is returned even when the gate fails. BUY/SELL is not returned unless the validation sample supports the configured gate.

## 5. News pipeline

### Times of India

The supplied TOI scraper logic is preserved:
- category pages under Business, Markets, Economy, Technology and World;
- article extraction by TOI `/articleshow/` URLs;
- URL de-duplication;
- robots.txt check;
- section tracking.

The integrated service refreshes this source every ten minutes and stores the latest in-memory/disk snapshot.

### FreeNewsApi

FreeNewsApi is deliberately slower than UI refreshes:
- broad India-market query every 30 minutes;
- selected-instrument query at most once per 30 minutes;
- 5,000 requests/day is a hard ceiling, not a target.

### X/Twitter

X recent-search support is optional and activates only when `X_BEARER_TOKEN` is configured. X data is treated as secondary context, not as authoritative market truth.

## 6. Stock universe

The Upstox NSE/BSE equity instrument master is downloaded and cached. Search is performed locally, so a symbol remains discoverable even when its quote/history endpoint is unavailable.

This separates:
- **instrument availability** — whether the security exists in the directory;
- **live quote availability** — whether a current price is available;
- **historical-data availability** — whether training candles exist.

## 7. Browser behavior

Instrument/timeframe changes clear stale state immediately.

On load:
1. health
2. priority directory
3. current/last candle
4. model analysis
5. current news
6. market pulse
7. universe directory

After load:
- live state every 60 seconds;
- model analysis every 5 minutes;
- news panel every 5 minutes (served from its slower cache);
- market pulse every 5 minutes.

## 8. Failure behavior

- Upstox WebSocket down: use cached/rest recovery when possible.
- Historical data unavailable: chart can still show a current/last-session state, but ML may return `REVALIDATE`.
- Insufficient training rows: forecast is withheld, rather than using a misleading model trained on too little data.
- News API failure: cached headlines remain visible.
- TOI scrape failure: previous TOI snapshot remains visible.
- X not configured: panel continues without X.
- Market closed: last real session remains visible.

## 9. Security

All credentials are environment variables. Do not commit `.env`, tokens, or API keys. Rotate credentials that have been exposed in chat or source control.
