# Radhe Stock Market Predictor

Live Indian market dashboard focused on SENSEX and NIFTY 50, with a separate next-session stock forecast workflow.

## Production data sources
- Upstox V3 WebSocket + V3 candle/history endpoints for market data.
- Times of India business/markets/economy scraper for user-facing headlines.
- FreeNewsApi and Moneycontrol as additional model context.
- X/Twitter is optional and disabled unless an X bearer token is configured.

## Dashboard behavior
- Dashboard 1: actual market candle, including the current forming minute while the market is open.
- Dashboard 2: next two forecast candles for the selected intraday timeframe.
- EOD outlook: fixed once per session from the morning model run.
- 1D: previous completed session/history.
- Listed equities: separate next-session forecast only.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Render
Use the supplied `render.yaml`. Required secrets are set in Render environment variables, never committed to Git.

The default production cadence is deliberately conservative:
- browser live refresh: 5 seconds
- prediction refresh: 20 seconds
- model retrain: 30 minutes
- TOI refresh: 10 minutes
- Moneycontrol refresh: 30 minutes
- FreeNews refresh: 30 minutes

## Safety
The application is analysis-only. It does not place broker orders. If real market history is insufficient, prediction is withheld rather than replaced with synthetic data.
