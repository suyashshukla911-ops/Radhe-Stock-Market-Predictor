from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import INDEX_INSTRUMENTS, SECTOR_BY_SYMBOL, IST, settings
from app.data.news import NewsService
from app.data.upstox import Instrument, LiveMarketHub, UpstoxClient, UpstoxError
from app.ml.features import FEATURES, build_features, prepare_training_frame
from app.ml.model import CandleModel

CONTEXT_KEYS = {
    "nifty_return": "NIFTY 50",
    "banknifty_return": "NIFTY BANK",
    "vix_return": "INDIA VIX",
}


@dataclass
class ModelCache:
    model: CandleModel
    fitted_at: float
    history: pd.DataFrame
    context_history: dict[str,pd.DataFrame]


class AnalysisEngine:
    def __init__(self, upstox: UpstoxClient, live: LiveMarketHub, news: NewsService):
        self.upstox=upstox
        self.live=live
        self.news=news
        self.models: dict[tuple[str,int], ModelCache]={}
        self.last_results: dict[tuple[str,int],dict]={}
        self._lock=threading.RLock()
        self._intraday_cache: dict[tuple[str,int],tuple[float,pd.DataFrame]]={}
        self._context_cache: dict[tuple[str,int],tuple[float,dict[str,pd.DataFrame]]]={}

    def _resolve_context(self) -> dict[str,Instrument]:
        out={}
        for label, name in [("nifty","NIFTY 50"),("banknifty","NIFTY BANK"),("vix","INDIA VIX")]:
            try: out[label]=self.upstox.resolve_symbol(name)
            except Exception: pass
        return out

    def _history_window(self, interval:int) -> int:
        return settings.train_days_short if interval<=15 else settings.train_days_long

    @staticmethod
    def _completed_session_cutoff() -> pd.Timestamp:
        now = datetime.now(ZoneInfo(IST))
        minute = now.hour * 60 + now.minute
        # During the live session, never train on today's incomplete session.
        # Once 15:30 IST has passed, today's session is complete and becomes
        # part of the next model fit.
        return pd.Timestamp(now.date() if minute >= 15 * 60 + 30 else (now - pd.Timedelta(days=1)).date())

    def _get_training_cache(self, instrument:Instrument, interval:int, force:bool=False) -> ModelCache:
        key=(instrument.instrument_key,interval)
        now=time.time()
        cached=self.models.get(key)
        if cached and not force and now-cached.fitted_at < settings.model_retrain_minutes*60:
            return cached
        days=self._history_window(interval)
        history=self.upstox.get_recent_history(instrument.instrument_key,interval,days)
        if history.empty:
            raise UpstoxError("No historical candles returned from Upstox for this instrument/timeframe.")
        cutoff = self._completed_session_cutoff().date()
        history_train=history.loc[history["timestamp"].dt.date <= cutoff].copy()
        context_hist={}
        for label, ctx in self._resolve_context().items():
            try:
                h=self.upstox.get_recent_history(ctx.instrument_key,interval,days)
                context_hist[label]=h.loc[h["timestamp"].dt.date <= cutoff].copy()
            except Exception:
                context_hist[label]=pd.DataFrame()
        train=prepare_training_frame(history_train,context=context_hist)
        if len(train) < settings.min_train_rows:
            raise ValueError(f"Not enough completed-session data for {instrument.trading_symbol} at {interval}m. Raw candles={len(history_train)}, clean feature rows={len(train)}, need={settings.min_train_rows}. Upstox history/resampling must provide more intraday history.")
        model=CandleModel(min_train_rows=settings.min_train_rows)
        model.fit(train,FEATURES)
        cached=ModelCache(model=model,fitted_at=now,history=history_train,context_history=context_hist)
        self.models[key]=cached
        return cached

    def _intraday_snapshot(self, instrument_key:str, interval:int, ttl:float=5.0) -> pd.DataFrame:
        key=(instrument_key,interval); now=time.time()
        cached=self._intraday_cache.get(key)
        if cached and now-cached[0] < ttl:
            return cached[1].copy()
        try:
            df=self.upstox.get_intraday(instrument_key,interval)
        except Exception:
            df=pd.DataFrame()
        self._intraday_cache[key]=(now,df.copy())
        return df

    def _merge_live_rest(self, instrument_key:str, interval:int) -> pd.DataFrame:
        self.live.subscribe([instrument_key])
        bars=self.live.live_bars(instrument_key,interval)
        intraday=self._intraday_snapshot(instrument_key,interval,ttl=5.0)
        if bars.empty:return intraday
        if intraday.empty:return bars
        return pd.concat([intraday,bars],ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)

    def _live_or_intraday(self, instrument:Instrument, interval:int) -> pd.DataFrame:
        return self._merge_live_rest(instrument.instrument_key,interval)

    def _live_context(self, interval:int) -> dict[str,pd.DataFrame]:
        out={}; contexts=self._resolve_context()
        for label,ins in contexts.items():
            out[label]=self._merge_live_rest(ins.instrument_key,interval)
        return out

    def analyze(self, instrument:Instrument, interval:int, force_retrain:bool=False) -> dict:
        if interval not in {1,5,10,15,30,60}:
            raise ValueError("Supported live prediction intervals: 1, 5, 10, 15, 30, 60 minutes.")
        self.live.subscribe([instrument.instrument_key])
        cache=self._get_training_cache(instrument,interval,force=force_retrain)
        live=self._live_or_intraday(instrument,interval)
        if live.empty:
            raise UpstoxError("No live candles are available yet. Make sure the Upstox market feed is connected.")
        combined=pd.concat([cache.history,live],ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)
        context=self._live_context(interval)
        features=build_features(combined,context=context)
        usable=features.dropna(subset=FEATURES).copy()
        if usable.empty:
            raise ValueError("Not enough candles to build current live features.")
        latest=usable.iloc[[-1]]
        current_price=float(latest["close"].iloc[0])

        sector=SECTOR_BY_SYMBOL.get(instrument.trading_symbol.upper(),"Market Index" if instrument.segment.endswith("INDEX") else "Unknown")
        articles=self.news.search(instrument.name or instrument.trading_symbol,instrument.trading_symbol)
        summary=self.news.aggregate(articles)
        prediction=cache.model.predict(latest,current_price,news_impact=summary["impact"])
        last_ts=latest["timestamp"].iloc[0]

        # Use only the most recent chart window, but keep the current day forming candle visible.
        chart=live.tail(settings.chart_bars)
        rows=[]
        for r in chart.itertuples():
            rows.append({"time":r.timestamp.isoformat(),"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close),"volume":float(r.volume) if pd.notna(r.volume) else 0.0})

        result={
            "instrument":{"name":instrument.name,"short_name":instrument.short_name,"symbol":instrument.trading_symbol,"exchange":instrument.exchange,"segment":instrument.segment,"instrument_key":instrument.instrument_key,"sector":sector},
            "interval":interval,
            "last_timestamp":last_ts.isoformat(),
            "current_price":current_price,
            "candles":rows,
            "prediction":prediction,
            "news":{"summary":summary,"articles":articles[:12],"diagnostics":self.news.diagnostics()},
            "market_state":{"market_data_source":"Upstox WebSocket + V3 candles","stream_connected":self.live.connected,"stream_error":self.live.last_error,"history_rows":len(cache.history),"training_rows":prediction["validation"]["train_rows"],"live_rows":len(live),"model_cache_age_seconds":round(time.time()-cache.fitted_at,1),"prediction_refresh_seconds":settings.prediction_refresh_seconds},
        }
        self.last_results[(instrument.instrument_key,interval)]=result
        return result

    def live_state(self,instrument:Instrument,interval:int) -> dict:
        self.live.subscribe([instrument.instrument_key])
        tick=self.live.latest(instrument.instrument_key)
        bars=self.live.live_bars(instrument.instrument_key,interval)
        # REST snapshot is a recovery path, never the primary 1-second UI source.
        if bars.empty:
            bars=self._intraday_snapshot(instrument.instrument_key,interval,ttl=5.0)
        now=datetime.now(ZoneInfo(IST))
        minute=now.hour*60+now.minute
        market_open = minute >= 9*60+15 and minute < 15*60+30 and now.weekday() < 5
        market_status = "OPEN" if market_open else ("WEEKEND" if now.weekday() >= 5 else "CLOSED")
        # Never manufacture a new candle after the cash market closes.
        if not market_open:
            tick = tick if tick else None
        if tick is None and not bars.empty:
            last=bars.iloc[-1]
            tick={"price":float(last.close),"trade_time":pd.Timestamp(last.timestamp).isoformat(),"updated_at":now.isoformat(),"instrument_key":instrument.instrument_key}
        current_candle=None; seconds_left=None
        if not bars.empty:
            row=bars.iloc[-1]; start=pd.Timestamp(row["timestamp"]).tz_convert(IST)
            end=min(start+pd.Timedelta(minutes=interval), start.normalize()+pd.Timedelta(minutes=15*60+30))
            current_candle={"start":start.isoformat(),"end":end.isoformat(),"open":float(row.open),"high":float(row.high),"low":float(row.low),"close":float(row.close),"volume":float(row.volume),"forming":bool(market_open and start.date()==now.date() and start <= pd.Timestamp(now))}
            if market_open and current_candle["forming"]:
                seconds_left=max(0,int((end.to_pydatetime()-now).total_seconds()))
            else:
                seconds_left=0
        cached=self.last_results.get((instrument.instrument_key,interval))
        age=None
        if tick and tick.get("updated_at"):
            try: age=round(max(0.0,(now-datetime.fromisoformat(tick["updated_at"])).total_seconds()),2)
            except Exception: age=None
        return {"now":now.isoformat(),"instrument_key":instrument.instrument_key,"symbol":instrument.trading_symbol,"tick":tick,"current_candle":current_candle,"seconds_to_candle_close":seconds_left,"prediction":cached.get("prediction") if cached else None,"market_data":{"stream_connected":self.live.connected,"stream_error":self.live.last_error,"last_update_age_seconds":age,"source":"websocket" if self.live.connected and tick else "rest-recovery" if tick else "unavailable","market_status":market_status}}
