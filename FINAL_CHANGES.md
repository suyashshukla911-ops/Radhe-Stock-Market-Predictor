# Final Changes — 16 Oct 2026

1. Split the experience into two explicit dashboards:
   - Dashboard 1: actual market candles.
   - Dashboard 2: next two modelled candles.
2. Live current-minute candle now combines the server-side Upstox WebSocket state with REST backfill, so the display is not intentionally two minutes behind.
3. Browser state refresh is 5 seconds; Upstox REST backfill is cache-controlled. The WebSocket remains the high-frequency market source.
4. EOD forecast is persisted once per instrument/session and is not rewritten by later price movement or manual retraining.
5. Added next-session equity forecasting for listed stocks using real daily Upstox history and news context. Intraday model is restricted to SENSEX and NIFTY 50.
6. Added Moneycontrol as a backend-only model-context source on a 30-minute refresh cadence.
7. User-facing news remains Times of India business/markets/economy only, with Excel export.
8. Added explicit live/current vs forecast labeling and mobile-friendly two-chart layout.
9. No synthetic market data is used in production. Provider failures withhold predictions rather than fabricate prices.
10. Test suite: 8 passed.
