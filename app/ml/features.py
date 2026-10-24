from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
    "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "atr_pct", "rsi_14", "ema_9_gap", "ema_20_gap", "ema_spread",
    "volatility_20", "volatility_60", "volume_z", "vwap_gap",
    "day_position", "day_sin", "day_cos", "dow_sin", "dow_cos",
    "session_gap", "session_return", "session_high_gap", "session_low_gap",
    "nifty_return", "banknifty_return", "vix_return", "price_vs_session_vwap", "range_expansion",
]

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50.0)

def add_context(base: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = base.copy().sort_values("timestamp")
    tolerance = pd.to_timedelta(60, unit="m")
    for name, ctx in context.items():
        if ctx is None or ctx.empty:
            out[f"{name}_return"] = 0.0
            continue
        temp = ctx[["timestamp", "close"]].sort_values("timestamp").copy()
        temp[f"{name}_return"] = temp["close"].pct_change().fillna(0.0)
        temp = temp[["timestamp", f"{name}_return"]]
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            temp,
            on="timestamp",
            direction="backward",
            tolerance=tolerance,
        )
        out[f"{name}_return"] = out[f"{name}_return"].fillna(0.0)
    return out

def build_features(df: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if data.empty:
        return data
    for c in ["open","high","low","close","volume"]:
        data[c] = pd.to_numeric(data[c], errors="coerce").astype(float)

    # Index feeds frequently expose zero volume. Do not let that erase every
    # feature row; use a session typical-price VWAP fallback in that case.
    data["volume"] = data["volume"].fillna(0.0)
    prev_close = data["close"].shift(1)
    for n in [1,2,3,5,10,20]:
        data[f"return_{n}"] = data["close"].pct_change(n)
    data["range_pct"] = (data["high"]-data["low"]) / data["close"].replace(0,np.nan)
    data["body_pct"] = (data["close"]-data["open"]) / data["close"].replace(0,np.nan)
    top = data[["open","close"]].max(axis=1)
    bot = data[["open","close"]].min(axis=1)
    data["upper_wick_pct"] = (data["high"]-top) / data["close"].replace(0,np.nan)
    data["lower_wick_pct"] = (bot-data["low"]) / data["close"].replace(0,np.nan)

    tr = pd.concat([(data["high"]-data["low"]), (data["high"]-prev_close).abs(), (data["low"]-prev_close).abs()], axis=1).max(axis=1)
    data["atr_pct"] = (tr.rolling(14, min_periods=5).mean() / data["close"].replace(0,np.nan)).ffill().fillna(0.0)
    data["rsi_14"] = _rsi(data["close"])
    ema9 = data["close"].ewm(span=9, adjust=False).mean()
    ema20 = data["close"].ewm(span=20, adjust=False).mean()
    data["ema_9_gap"] = data["close"]/ema9-1
    data["ema_20_gap"] = data["close"]/ema20-1
    data["ema_spread"] = ema9/ema20-1

    ret = data["return_1"]
    data["volatility_20"] = ret.rolling(20, min_periods=5).std().ffill().fillna(0.0)
    data["volatility_60"] = ret.rolling(60, min_periods=10).std().ffill().fillna(0.0)
    vm = data["volume"].rolling(30, min_periods=5).mean()
    vs = data["volume"].rolling(30, min_periods=5).std().replace(0,np.nan)
    data["volume_z"] = ((data["volume"]-vm)/vs).replace([np.inf,-np.inf],np.nan).fillna(0.0)

    session = data["timestamp"].dt.date
    typical = (data["high"]+data["low"]+data["close"])/3
    pv = typical * data["volume"]
    cumulative_v = data["volume"].groupby(session).cumsum()
    weighted_vwap = (pv.groupby(session).cumsum() / cumulative_v.replace(0,np.nan))
    count_vwap = typical.groupby(session).cumsum() / typical.groupby(session).cumcount().add(1)
    session_vwap = weighted_vwap.fillna(count_vwap)
    data["vwap_gap"] = data["close"]/session_vwap-1
    data["price_vs_session_vwap"] = data["close"]/session_vwap-1

    prior_range = (data["high"]-data["low"]).rolling(20, min_periods=5).median()
    denom = (prior_range/data["close"]).replace(0,np.nan)
    data["range_expansion"] = (((data["high"]-data["low"])/data["close"]) / denom).replace([np.inf,-np.inf],np.nan).fillna(1.0)

    market_open = 9*60+15
    minutes = data["timestamp"].dt.hour*60 + data["timestamp"].dt.minute
    data["day_position"] = (minutes-market_open).clip(lower=0,upper=375)/375.0
    data["day_sin"] = np.sin(2*np.pi*data["day_position"])
    data["day_cos"] = np.cos(2*np.pi*data["day_position"])
    dow = data["timestamp"].dt.dayofweek
    data["dow_sin"] = np.sin(2*np.pi*dow/5.0)
    data["dow_cos"] = np.cos(2*np.pi*dow/5.0)

    first_open = data.groupby(session)["open"].transform("first")
    daily_last = data.groupby(session)["close"].last()
    prior_close_by_session = daily_last.shift(1)
    data["session_gap"] = (first_open / pd.Series(session, index=data.index).map(prior_close_by_session) - 1).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    session_open = first_open.replace(0,np.nan)
    data["session_return"] = (data["close"]/session_open-1).fillna(0.0)
    sh = data.groupby(session)["high"].cummax()
    sl = data.groupby(session)["low"].cummin()
    data["session_high_gap"] = data["close"]/sh-1
    data["session_low_gap"] = data["close"]/sl-1

    if context:
        data = add_context(data, context)
    else:
        for c in ["nifty_return","banknifty_return","vix_return"]:
            data[c] = 0.0
    for c in ["nifty_return","banknifty_return","vix_return"]:
        if c not in data.columns:
            data[c] = 0.0
    return data.replace([np.inf,-np.inf],np.nan)

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    session = data["timestamp"].dt.date
    # Target T+1
    data["next_return"] = data["close"].shift(-1)/data["close"]-1
    data["next_high_return"] = data["high"].shift(-1)/data["close"]-1
    data["next_low_return"] = data["low"].shift(-1)/data["close"]-1
    data["direction"] = (data["next_return"]>0).astype(int)

    # Target T+2 (direct multi-step target)
    # shift inside each session so the 2nd-to-last bar never targets tomorrow's open
    by_session = data.groupby(session)
    data["next_2_return"] = by_session["close"].shift(-2)/data["close"]-1
    data["next_2_high_return"] = by_session["high"].shift(-2)/data["close"]-1
    data["next_2_low_return"] = by_session["low"].shift(-2)/data["close"]-1

    session_close = data.groupby(session)["close"].transform("last")
    data["eod_return"] = session_close/data["close"]-1
    max_ts = data.groupby(session)["timestamp"].transform("max")
    data["is_session_last"] = data["timestamp"].eq(max_ts)
    return data

def prepare_training_frame(df: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    data = add_targets(build_features(df, context=context))
    data = data.loc[~data["is_session_last"]].copy()
    data["next_2_return"] = data["next_2_return"].fillna(data["next_return"])
    data["next_2_high_return"] = data["next_2_high_return"].fillna(data["next_high_return"])
    data["next_2_low_return"] = data["next_2_low_return"].fillna(data["next_low_return"])
    return data.dropna(subset=FEATURES+["next_return","next_high_return","next_low_return","direction","eod_return"])
