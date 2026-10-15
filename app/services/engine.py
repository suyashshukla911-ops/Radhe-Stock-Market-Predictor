from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import INDEX_INSTRUMENTS, IST, SECTOR_BY_SYMBOL, settings
from app.data.news import NewsService
from app.data.upstox import Instrument, LiveMarketHub, UpstoxClient, UpstoxError
from app.ml.features import FEATURES, build_features, prepare_training_frame
from app.ml.model import CandleModel

@dataclass
class ModelCache:
    model: CandleModel
    fitted_at: float
    history: pd.DataFrame
    context_history: dict[str,pd.DataFrame]

class AnalysisEngine:
    def __init__(self, upstox: UpstoxClient, live: LiveMarketHub, news: NewsService):
        self.upstox = upstox
        self.live = live
        self.news = news
        self.models: dict[tuple[str,int],ModelCache] = {}
        self.last_results: dict[tuple[str,int],dict] = {}
        self._history_cache: dict[tuple[str,int],tuple[float,pd.DataFrame]] = {}
        self._context_cache: dict[tuple[str,int],tuple[float,dict[str,pd.DataFrame]]] = {}
        self._analysis_lock = threading.RLock()

    @staticmethod
    def _completed_cutoff():
        now=datetime.now(ZoneInfo(IST))
        mins=now.hour*60+now.minute
        return now.date() if mins>=15*60+30 and now.weekday()<5 else (now-pd.Timedelta(days=1)).date()

    @staticmethod
    def _market_open() -> bool:
        now=datetime.now(ZoneInfo(IST))
        if now.weekday()>=5: return False
        m=now.hour*60+now.minute
        return 9*60+15 <= m < 15*60+30

    def _history_window(self, interval):
        return settings.train_days_short if interval<=15 else settings.train_days_long

    def _resolve_context(self):
        out={}
        for label,name in [("nifty","NIFTY 50"),("banknifty","NIFTY BANK"),("vix","INDIA VIX")]:
            try: out[label]=self.upstox.resolve_symbol(name)
            except Exception: pass
        return out

    def _get_history(self, instrument:Instrument, interval:int, calendar_days:int, force=False):
        key=(instrument.instrument_key,interval)
        cached=self._history_cache.get(key)
        if cached and not force and time.time()-cached[0] < settings.history_cache_minutes*60:
            return cached[1].copy()
        hist=self.upstox.get_recent_history(instrument.instrument_key,interval,calendar_days)
        self._history_cache[key]=(time.time(),hist.copy())
        return hist

    def _get_context_history(self, interval:int, days:int, force=False):
        key=(interval,days)
        cached=self._context_cache.get(key)
        if cached and not force and time.time()-cached[0] < settings.history_cache_minutes*60:
            return {k:v.copy() for k,v in cached[1].items()}
        out={}
        for label,ins in self._resolve_context().items():
            try: out[label]=self.upstox.get_recent_history(ins.instrument_key,interval,days)
            except Exception: out[label]=pd.DataFrame()
        self._context_cache[key]=(time.time(),out)
        return {k:v.copy() for k,v in out.items()}

    def _get_training_cache(self,instrument,interval,force=False):
        key=(instrument.instrument_key,interval)
        now=time.time()
        cached=self.models.get(key)
        if cached and not force and now-cached.fitted_at < settings.model_retrain_minutes*60:
            return cached

        days=self._history_window(interval)
        history=self._get_history(instrument,interval,days,force=force)
        if history.empty:
            raise UpstoxError("Upstox returned no intraday history for this instrument.")
        cutoff=self._completed_cutoff()
        history_train=history.loc[history["timestamp"].dt.date <= cutoff].copy()
        context=self._get_context_history(interval,days,force=force)
        context_train={k:v.loc[v["timestamp"].dt.date <= cutoff].copy() for k,v in context.items()}
        train=prepare_training_frame(history_train,context=context_train)
        if len(train) < settings.min_train_rows:
            raise ValueError(
                f"Not enough clean training rows for {instrument.trading_symbol} at {interval}m: "
                f"raw={len(history_train)}, clean={len(train)}, need={settings.min_train_rows}. "
                "The live data remains available; model prediction is withheld until the training sample is adequate."
            )
        model=CandleModel(settings.min_train_rows)
        model.fit(train,FEATURES)
        cache=ModelCache(model,time.time(),history_train,context_train)
        self.models[key]=cache
        return cache

    def _live_snapshot(self,instrument,interval,ttl_seconds=None):
        ttl=settings.live_rest_cache_seconds if ttl_seconds is None else ttl_seconds
        self.live.subscribe([instrument.instrument_key])
        cache_key=(instrument.instrument_key,interval)

        # 1) Prefer the server-side WebSocket bars while the market is live.
        try:
            live_1m=self.live.live_bars(instrument.instrument_key,1)
            if not live_1m.empty:
                aggregated=self.upstox.aggregate_minutes(live_1m,interval)
                if not aggregated.empty:
                    self._live_view_cache[cache_key]=(time.time(),aggregated.copy())
                    return aggregated
        except Exception:
            pass

        # 2) Reuse a recent REST snapshot before making another request.
        cached=self._live_view_cache.get(cache_key)
        if cached and time.time()-cached[0] < ttl:
            return cached[1].copy()

        # 3) Current-day intraday endpoint. This can legitimately be empty after
        # the cash market closes, so it is only the first REST fallback.
        intraday=pd.DataFrame()
        try:
            intraday=self.upstox.get_intraday(instrument.instrument_key,interval)
        except Exception:
            pass

        # 4) IMPORTANT: after market close (or before the first live tick), use
        # completed historical sessions so the UI keeps the last real candle
        # instead of becoming blank or inventing an after-hours candle.
        if intraday.empty:
            try:
                days=settings.train_days_short if interval <= 15 else min(settings.train_days_long,30)
                historical=self.upstox.get_recent_history(instrument.instrument_key,interval,max(7,days))
                if not historical.empty:
                    intraday=historical
            except Exception:
                pass

        self._live_view_cache[cache_key]=(time.time(),intraday.copy())
        return intraday

    def _merge_training_and_live(self,instrument,interval):
        history_cache=self._get_history(instrument,interval,self._history_window(interval))
        live=self._live_snapshot(instrument,interval)
        if live.empty: return history_cache,live
        combined=pd.concat([history_cache,live],ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)
        return combined,live

    def analyze(self,instrument:Instrument,interval:int,force_retrain:bool=False):
        if interval not in {1,5,10,15,30,60}:
            raise ValueError("Supported timeframes: 1m, 5m, 10m, 15m, 30m, 60m.")
        key=(instrument.instrument_key,interval)
        with self._analysis_lock:
            # Reuse a recent result instead of refitting/re-hitting external services.
            cached=self.last_results.get(key)
            if cached and not force_retrain:
                age=time.time()-cached["_created_at"]
                if (not self._market_open()) or age < settings.prediction_refresh_seconds:
                    return {k:v for k,v in cached.items() if k!="_created_at"}

            model_cache=self._get_training_cache(instrument,interval,force=force_retrain)
            combined,live=self._merge_training_and_live(instrument,interval)
            if live.empty:
                # A model result can be absent while the chart still works in /api/live.
                raise UpstoxError("No current-session candle snapshot is available yet.")
            context_live={}
            for label,ins in self._resolve_context().items():
                try:
                    context_live[label]=self._live_snapshot(ins,interval)
                except Exception: context_live[label]=pd.DataFrame()

            features=build_features(combined,context=context_live)
            usable=features.dropna(subset=FEATURES).copy()
            if usable.empty:
                raise ValueError("Current market data does not contain enough feature history yet.")
            latest=usable.iloc[[-1]]
            current_price=float(latest["close"].iloc[0])

            company=instrument.name or instrument.trading_symbol
            news=self.news.snapshot(instrument.trading_symbol,company)
            prediction=model_cache.model.predict(latest,current_price,news_impact=news["summary"]["impact"])

            rows=[]
            for r in live.tail(settings.chart_bars).itertuples():
                rows.append({"time":r.timestamp.isoformat(),"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close),"volume":float(r.volume or 0.0)})

            created=time.time()
            result={
                "instrument":{"name":instrument.name,"short_name":instrument.short_name,"symbol":instrument.trading_symbol,"exchange":instrument.exchange,"segment":instrument.segment,"instrument_key":instrument.instrument_key,"sector":SECTOR_BY_SYMBOL.get(instrument.trading_symbol.upper(),"Market Index" if instrument.segment.endswith("INDEX") else "Unknown")},
                "interval":interval,
                "last_timestamp":latest["timestamp"].iloc[0].isoformat(),
                "current_price":current_price,
                "candles":rows,
                "prediction":prediction,
                "news":news,
                "market_state":{
                    "market_data_source":"Upstox V3 WebSocket + V3 intraday candles",
                    "stream_connected":self.live.connected,
                    "stream_error":self.live.last_error,
                    "history_rows":len(model_cache.history),
                    "training_rows":prediction["validation"]["train_rows"],
                    "live_rows":len(live),
                    "model_cache_age_seconds":round(time.time()-model_cache.fitted_at,1),
                    "prediction_refresh_seconds":settings.prediction_refresh_seconds,
                    "model_retrain_minutes":settings.model_retrain_minutes,
                    "market_status":"OPEN" if self._market_open() else "CLOSED",
                },
                "refresh_policy":{
                    "browser_seconds":settings.live_poll_seconds,
                    "prediction_seconds":settings.prediction_refresh_seconds,
                    "model_retrain_minutes":settings.model_retrain_minutes,
                    "toi_minutes":settings.toi_refresh_minutes,
                    "news_minutes":settings.news_refresh_minutes,
                },
                "_created_at":created,
            }
            self.last_results[key]=result
            return {k:v for k,v in result.items() if k!="_created_at"}

    def live_state(self,instrument:Instrument,interval:int):
        self.live.subscribe([instrument.instrument_key])
        now=datetime.now(ZoneInfo(IST))
        open_now=self._market_open()
        tick=self.live.latest(instrument.instrument_key)
        bars=self._live_snapshot(instrument,interval,ttl_seconds=settings.live_rest_cache_seconds)

        if bars.empty:
            cache=self.last_results.get((instrument.instrument_key,interval))
            if cache:
                rows=cache.get("candles",[])
            else:
                rows=[]
        else:
            rows=[]
            for r in bars.tail(settings.chart_bars).itertuples():
                rows.append({"time":r.timestamp.isoformat(),"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close),"volume":float(r.volume or 0.0)})

        if tick is None and not bars.empty:
            last=bars.iloc[-1]
            last_ts=pd.Timestamp(last.timestamp)
            if last_ts.tzinfo is None:
                last_ts=last_ts.tz_localize(IST)
            else:
                last_ts=last_ts.tz_convert(IST)
            tick={
                "price":float(last.close),
                "trade_time":last_ts.isoformat(),
                # Keep the real candle timestamp so the UI shows a meaningful
                # stale age after the market closes.
                "updated_at":last_ts.isoformat(),
                "instrument_key":instrument.instrument_key,
                "session_trade":False,
            }

        # If the candle-history provider is temporarily empty, still show the
        # latest quoted market price instead of a blank dashboard.
        if tick is None:
            try:
                quotes=self.upstox.get_quote_batch([instrument.instrument_key])
                q=quotes.get(instrument.instrument_key) or next(iter(quotes.values()), {})
                last_price=q.get("last_price") or (q.get("ohlc") or {}).get("close")
                prev_close=q.get("prev_close_price")
                if last_price is not None:
                    tick={
                        "price":float(last_price),
                        "previous_close":float(prev_close) if prev_close is not None else None,
                        "trade_time":None,
                        "updated_at":now.isoformat(),
                        "instrument_key":instrument.instrument_key,
                        "session_trade":False,
                    }
            except Exception:
                pass

        current_candle=None
        seconds_left=0
        if not bars.empty:
            row=bars.iloc[-1]
            start=pd.Timestamp(row.timestamp).tz_convert(IST)
            end=min(start+pd.Timedelta(minutes=interval),start.normalize()+pd.Timedelta(minutes=15*60+30))
            forming=bool(open_now and start.date()==now.date() and start <= pd.Timestamp(now))
            current_candle={"start":start.isoformat(),"end":end.isoformat(),"open":float(row.open),"high":float(row.high),"low":float(row.low),"close":float(row.close),"volume":float(row.volume or 0.0),"forming":forming}
            if forming: seconds_left=max(0,int((end.to_pydatetime()-now).total_seconds()))

        last_result=self.last_results.get((instrument.instrument_key,interval))
        status="OPEN" if open_now else ("WEEKEND" if now.weekday()>=5 else "CLOSED")
        age=None
        if tick and tick.get("updated_at"):
            try: age=round(max(0,(now-datetime.fromisoformat(tick["updated_at"])).total_seconds()),1)
            except Exception: pass

        return {
            "now":now.isoformat(),
            "instrument_key":instrument.instrument_key,
            "symbol":instrument.trading_symbol,
            "tick":tick,
            "candles":rows,
            "current_candle":current_candle,
            "seconds_to_candle_close":seconds_left,
            "prediction":last_result.get("prediction") if last_result else None,
            "market_data":{
                "stream_connected":self.live.connected,
                "stream_error":self.live.last_error,
                "last_update_age_seconds":age,
                "source":"websocket" if self.live.connected and tick and open_now else "rest-last-session" if bars.size else "unavailable",
                "market_status":status,
                "last_market_price":float(bars.iloc[-1]["close"]) if not bars.empty else None,
                "last_market_candle":bars.iloc[-1]["timestamp"].isoformat() if not bars.empty else None,
            }
        }
