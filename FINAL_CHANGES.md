# Radhe-Radhe Final Changes

This build incorporates the latest user-facing refinements:

- Closed-market predictions are frozen per instrument/timeframe until the next trading session, unless the user explicitly retrains.
- Intraday 1m/5m/10m/15m/30m/60m views use only the latest completed cash session when the market is closed, preventing mixed-session charts and unstable after-hours probabilities.
- 1D remains a daily previous-session/history view while the market is closed and can use the live day while the market is open.
- Top market news is displayed directly from Times of India, prioritising Business, India Business, International Business, Economy, Markets and Technology sections; the model can still consume the full news context.
- Added a manual Refresh News action and Excel download for the TOI dataset.
- User-facing UI no longer exposes backend provider implementation details such as FreeNewsApi, X/Twitter or Upstox V3 internals.
- Market pulse is explicitly presented as live priority stocks and indices.
- Added plain-language “How to read” guidance so new traders can understand price, next-candle estimates, probability and REVALIDATE status.
- Plotly charts now preserve zoom/pan state with `uirevision`, use pan mode, and remain responsive on phones.
- Search and selection feedback remains visible and market-pulse cards continue to load the selected instrument.

## Validation

- Python compile check: passed
- JavaScript syntax check: passed
- Test suite: 7 passed
- FastAPI smoke test: `/` 200, `/static/styles.css` 200, `/api/config` 200
