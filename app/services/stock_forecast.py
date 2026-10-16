from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from app.config import IST, settings
from app.data.news import NewsService
from app.data.upstox import Instrument, UpstoxClient

FEATURES = ["ret1","ret5","ret20","range_pct","atr_pct","ema9_gap","ema20_gap","volatility20","volume_z"]

@dataclass
class StockForecast:
    symbol: str
    company: str
    current_price: float
    next_day_close: float
    next_day_high: float
    next_day_low: float
    up_probability: float
    signal: str
    news_impact: float
    training_rows: int
    generated_at: str

class StockForecastService:
    def __init__(self, upstox: UpstoxClient, news: NewsService):
        self.upstox = upstox
        self.news = news
        self.cache: dict[str, tuple[float, StockForecast]] = {}

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        d=df.copy().sort_values("timestamp").reset_index(drop=True)
        d["close"]=pd.to_numeric(d["close"],errors="coerce")
        d["open"]=pd.to_numeric(d["open"],errors="coerce")
        d["high"]=pd.to_numeric(d["high"],errors="coerce")
        d["low"]=pd.to_numeric(d["low"],errors="coerce")
        d["volume"]=pd.to_numeric(d["volume"],errors="coerce").fillna(0.0)
        d["ret1"]=d.close.pct_change()
        d["ret5"]=d.close.pct_change(5)
        d["ret20"]=d.close.pct_change(20)
        d["range_pct"]=(d.high-d.low)/d.close.replace(0,np.nan)
        tr=pd.concat([(d.high-d.low),(d.high-d.close.shift(1)).abs(),(d.low-d.close.shift(1)).abs()],axis=1).max(axis=1)
        d["atr_pct"]=tr.rolling(14,min_periods=5).mean()/d.close.replace(0,np.nan)
        d["ema9_gap"]=d.close/d.close.ewm(span=9,adjust=False).mean()-1
        d["ema20_gap"]=d.close/d.close.ewm(span=20,adjust=False).mean()-1
        d["volatility20"]=d.ret1.rolling(20,min_periods=5).std()
        vm=d.volume.rolling(20,min_periods=5).mean(); vs=d.volume.rolling(20,min_periods=5).std().replace(0,np.nan)
        d["volume_z"]=((d.volume-vm)/vs).fillna(0.0)
        return d.replace([np.inf,-np.inf],np.nan)

    def forecast(self, instrument: Instrument, force: bool=False) -> dict:
        key=instrument.instrument_key
        cached=self.cache.get(key)
        if cached and not force and datetime.now(ZoneInfo(IST)).timestamp()-cached[0] < 1800:
            return cached[1].__dict__
        daily=self.upstox.get_daily_history(key,max(400,settings.train_days_long*4))
        if daily.empty or len(daily)<80:
            raise ValueError("Not enough real daily Upstox data for next-session stock forecast.")
        d=self._features(daily)
        d["next_close_ret"]=d.close.shift(-1)/d.close-1
        d["next_high_ret"]=d.high.shift(-1)/d.close-1
        d["next_low_ret"]=d.low.shift(-1)/d.close-1
        d["direction"]=(d.next_close_ret>0).astype(int)
        train=d.iloc[:-1].dropna(subset=FEATURES+['next_close_ret','next_high_ret','next_low_ret','direction']).copy()
        if len(train)<60 or train.direction.nunique()<2:
            raise ValueError("Real daily stock history is not sufficient for next-session training.")
        split=max(40,int(len(train)*0.85))
        X=train[FEATURES]; y=train.direction
        clf=HistGradientBoostingClassifier(learning_rate=0.04,max_iter=120,max_leaf_nodes=16,min_samples_leaf=10,l2_regularization=0.2,random_state=42)
        clf.fit(X.iloc[:split],y.iloc[:split])
        val_p=clf.predict(X.iloc[split:])
        accuracy=float(accuracy_score(y.iloc[split:],val_p)) if split<len(train) else 0.5
        clf.fit(X,y)
        reg_args=dict(learning_rate=0.04,max_iter=140,max_leaf_nodes=16,min_samples_leaf=10,l2_regularization=0.2,loss="absolute_error",random_state=42)
        ret=HistGradientBoostingRegressor(**reg_args).fit(X,train.next_close_ret)
        hi=HistGradientBoostingRegressor(**reg_args).fit(X,train.next_high_ret)
        lo=HistGradientBoostingRegressor(**reg_args).fit(X,train.next_low_ret)
        extra=ExtraTreesRegressor(n_estimators=80,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1).fit(X,train.next_close_ret)
        latest=d.dropna(subset=FEATURES).iloc[[-1]]
        price=float(latest.close.iloc[0]); raw_p=float(clf.predict_proba(latest[FEATURES])[0,1])
        news=self.news.snapshot(instrument.trading_symbol,instrument.name or instrument.trading_symbol)
        impact=float(news["summary"].get("impact",0.0))
        prob=float(np.clip(0.9*raw_p+0.1*(0.5+0.5*impact),0.01,0.99))
        r=0.65*float(ret.predict(latest[FEATURES])[0])+0.35*float(extra.predict(latest[FEATURES])[0])
        h=float(hi.predict(latest[FEATURES])[0]); l=float(lo.predict(latest[FEATURES])[0])
        forecast=StockForecast(
            symbol=instrument.trading_symbol, company=instrument.name or instrument.trading_symbol,
            current_price=price, next_day_close=price*(1+r), next_day_high=price*(1+max(h,r,0)), next_day_low=price*(1+min(l,r,0)),
            up_probability=prob, signal="UP" if prob>=0.6 else "DOWN" if prob<=0.4 else "NEUTRAL",
            news_impact=impact, training_rows=len(train), generated_at=datetime.now(ZoneInfo(IST)).isoformat()
        )
        self.cache[key]=(datetime.now(ZoneInfo(IST)).timestamp(),forecast)
        return forecast.__dict__ | {"validation_accuracy":round(accuracy*100,2)}
