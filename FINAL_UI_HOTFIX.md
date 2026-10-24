# Suyash UI Hotfix

Changes in this build:

- Replaced the intraday instrument free-text control with exactly two trading choices: SENSEX and NIFTY 50.
- Added request cancellation and view-generation guards so stale NIFTY results cannot overwrite a newly selected SENSEX view (and vice versa).
- Added clearer actual-candle OHLC cards and explicit Plotly OHLC hover values.
- Improved candle readability with 2-decimal price axes, stronger bodies/wicks, and a clearer chart layout.
- Current/live candle is merged into the actual chart so the latest candle remains visible at the right edge.
- Added live stock/company search suggestions in the next-session equity forecast section.
- Added a clearer stock forecast result card with next-session date, expected close/high/low, probability and validation information.
- Fixed the missing `HistGradientBoostingRegressor` import in the stock forecasting service.
- Added next-trading-session date calculation so weekend forecasts are labeled with the next actual trading session.
- Added UI cache-busting from `final4` to `final5` for the frontend assets.

Validation:

- Python compilation: OK
- JavaScript syntax: OK
- Pytest: 8 passed
