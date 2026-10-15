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
    return (100 - 100/(1+rs)).fillna(50)


def add_context(base: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for name, ctx in context.items():
        if ctx is None or ctx.empty:
            out[f"{name}_return"] = 0.0
            continue
        temp = ctx[["timestamp", "close"]].sort_values("timestamp").copy()
        temp[f"{name}_return"] = temp["close"].pct_change()
        temp = temp[["timestamp", f"{name}_return"]]
        out = pd.merge_asof(out.sort_values("timestamp"), temp, on="timestamp", direction="backward", tolerance=pd.Timedelta("30min"))
        out[f"{name}_return"] = out[f"{name}_return"].fillna(0.0)
    return out


def build_features(df: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if data.empty:
        return data
    for c in ["open","high","low","close","volume"]:
        data[c] = pd.to_numeric(data[c], errors="coerce").astype(float)

    prev_close = data["close"].shift(1)
    for n in [1,2,3,5,10,20]:
        data[f"return_{n}"] = data["close"].pct_change(n)
    data["range_pct"] = (data["high"]-data["low"]) / data["close"].replace(0,np.nan)
    data["body_pct"] = (data["close"]-data["open"]) / data["close"].replace(0,np.nan)
    top = data[["open","close"]].max(axis=1)
    bot = data[["open","close"]].min(axis=1)
    data["upper_wick_pct"] = (data["high"]-top) / data["close"].replace(0,np.nan)
    data["lower_wick_pct"] = (bot-data["low"]) / data["close"].replace(0,np.nan)

    tr = pd.concat([(data["high"]-data["low"]),(data["high"]-prev_close).abs(),(data["low"]-prev_close).abs()],axis=1).max(axis=1)
    data["atr_pct"] = tr.rolling(14).mean() / data["close"].replace(0,np.nan)
    data["rsi_14"] = _rsi(data["close"])
    ema9 = data["close"].ewm(span=9, adjust=False).mean()
    ema20 = data["close"].ewm(span=20, adjust=False).mean()
    data["ema_9_gap"] = data["close"]/ema9-1
    data["ema_20_gap"] = data["close"]/ema20-1
    data["ema_spread"] = ema9/ema20-1

    ret = data["return_1"]
    data["volatility_20"] = ret.rolling(20).std()
    data["volatility_60"] = ret.rolling(60).std()
    vm = data["volume"].rolling(30).mean()
    vs = data["volume"].rolling(30).std().replace(0,np.nan)
    data["volume_z"] = ((data["volume"]-vm)/vs).fillna(0.0)

    session = data["timestamp"].dt.date
    typical = (data["high"]+data["low"]+data["close"])/3
    pv = typical*data["volume"].fillna(0)
    cumulative_v = data["volume"].groupby(session).cumsum().replace(0,np.nan)
    session_vwap = pv.groupby(session).cumsum()/cumulative_v
    data["vwap_gap"] = data["close"]/session_vwap-1
    data["price_vs_session_vwap"] = data["close"]/session_vwap-1
    prior_range = (data["high"]-data["low"]).rolling(20).median()
    data["range_expansion"] = ((data["high"]-data["low"])/data["close"]) / (prior_range/data["close"]).replace(0,np.nan)

    market_open = 9*60+15
    minutes = data["timestamp"].dt.hour*60+data["timestamp"].dt.minute
    data["day_position"] = (minutes-market_open).clip(lower=0,upper=375)/375.0
    data["day_sin"] = np.sin(2*np.pi*data["day_position"])
    data["day_cos"] = np.cos(2*np.pi*data["day_position"])
    dow = data["timestamp"].dt.dayofweek
    data["dow_sin"] = np.sin(2*np.pi*dow/5.0)
    data["dow_cos"] = np.cos(2*np.pi*dow/5.0)

    first_open = data.groupby(session)["open"].transform("first")
    daily_last = data.groupby(session)["close"].last()
    prev_session_close = pd.Series(session).map(daily_last.shift(1)).to_numpy()
    session_gap = pd.Series(first_open.to_numpy()/prev_session_close-1, index=data.index)
    data["session_gap"] = session_gap.fillna(0.0).to_numpy()
    session_open = first_open.replace(0,np.nan)
    data["session_return"] = data["close"]/session_open-1
    sh = data.groupby(session)["high"].cummax()
    sl = data.groupby(session)["low"].cummin()
    data["session_high_gap"] = data["close"]/sh-1
    data["session_low_gap"] = data["close"]/sl-1

    if context:
        data = add_context(data, context)
    else:
        for c in ["nifty_return","banknifty_return","vix_return"]:
            data[c] = 0.0
    return data.replace([np.inf,-np.inf],np.nan)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    session = data["timestamp"].dt.date
    data["next_return"] = data["close"].shift(-1)/data["close"]-1
    data["next_high_return"] = data["high"].shift(-1)/data["close"]-1
    data["next_low_return"] = data["low"].shift(-1)/data["close"]-1
    data["direction"] = (data["next_return"]>0).astype(int)
    session_close = data.groupby(session)["close"].transform("last")
    data["eod_return"] = session_close/data["close"]-1
    data["is_session_last"] = data["timestamp"].eq(data.groupby(session)["timestamp"].transform("max"))
    return data


def prepare_training_frame(df: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    data = add_targets(build_features(df, context=context))
    data = data.loc[~data["is_session_last"]].copy()
    return data.dropna(subset=FEATURES+["next_return","next_high_return","next_low_return","direction","eod_return"])
