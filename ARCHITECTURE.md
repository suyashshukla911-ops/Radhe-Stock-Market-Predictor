# Suyash Stock Market Predictor — Live Trading Architecture

## 1. Actual market dashboard
The first dashboard is the market truth layer.

- Primary realtime source: Upstox Market Data Feed V3 WebSocket.
- 1-minute live OHLC is used to keep the current minute visible without waiting for the REST candle endpoint to close the minute.
- REST intraday data is used as a history/backfill layer and only refreshed on a slow cache cadence.
- 5m / 10m / 15m / 30m / 60m candles are local aggregations of the real 1-minute session data.
- When the market is closed, the last completed cash-market session remains visible.
- 1D is a dedicated previous-session history view.

Upstox V3 documents real-time WebSocket updates and current/intraday OHLC endpoints; the application uses the WebSocket for live state and V3 candle/history endpoints for backfill. See Upstox documentation for the supported feed and candle endpoints.

## 2. Prediction dashboard
The second dashboard is model-only.

- It forecasts the next two candles of the selected intraday timeframe.
- At 09:47 with 1m selected, the first observed dashboard should show the live 09:47 candle while the prediction dashboard forecasts the next two 1-minute candles.
- At 09:51, the same rule applies to the current observed minute and the next two modelled minutes.
- The prediction refreshes every 20 seconds by default; retraining happens every 30 minutes.
- The browser refreshes the current market state every 5 seconds, but this does not mean Upstox REST is called every 5 seconds. The WebSocket is the realtime layer.

## 3. Fixed end-of-day outlook
The EOD forecast is created once for each instrument at the first valid morning model run and persisted for that session.

Later intraday price movement, retraining, or news refreshes do not rewrite the day's EOD forecast. This prevents the application from moving the EOD target whenever the market moves.

The morning outlook is calculated from real market state plus the available news context at the time the daily outlook is established.

## 4. News layers
User-facing news is Times of India business/markets/economy news.

Model-only context can include:
- Times of India
- FreeNewsApi
- Moneycontrol market headlines
- Optional X/Twitter context when configured

Moneycontrol is fetched on a slow 30-minute cadence from the requested market page and is used as additional model context. It is deliberately not exposed as a backend/source label in the normal user UI.

## 5. Stock workflow
Regular intraday trading is limited to SENSEX and NIFTY 50.

For listed equities, the application provides a separate next-session forecast:
- real daily Upstox candles
- daily technical features
- current news impact
- next-session close/high/low estimates
- up probability and neutral/up/down classification

## 6. No synthetic production market data
The production application does not generate substitute market prices when provider data is unavailable. If real market history is insufficient, the prediction is withheld.

Synthetic/random candles exist only inside automated unit tests.
