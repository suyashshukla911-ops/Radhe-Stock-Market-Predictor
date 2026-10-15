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
        self._live_view_cache: dict[tuple[str,object],tuple[float,pd.DataFrame]] = {}
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

    @staticmethod
    def _last_completed_session_date():
        now=datetime.now(ZoneInfo(IST))
        candidate=now.date()
        if now.weekday()>=5 or (now.hour*60+now.minute) < 15*60+30:
            candidate=(now - pd.Timedelta(days=1)).date()
        while candidate.weekday()>=5:
            candidate=(pd.Timestamp(candidate)-pd.Timedelta(days=1)).date()
        return candidate

    @staticmethod
    def _session_dates(df: pd.DataFrame) -> list:
        if df.empty or "timestamp" not in df.columns:
            return []
        data=df.copy()
        ts=pd.to_datetime(data["timestamp"],utc=True).dt.tz_convert(IST)
        mins=ts.dt.hour*60+ts.dt.minute
        data["session_date"]=ts.dt.date
        mask=(mins>=9*60+15)&(mins<15*60+30)&(ts.dt.dayofweek<5)
        return sorted(data.loc[mask,"session_date"].dropna().unique().tolist())

    def _history_window(self, interval):
        if interval == "1d":
            return max(365, settings.train_days_long * 4)
        return settings.train_days_short if int(interval)<=15 else settings.train_days_long

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
        if interval == "1d":
            raise ValueError("Daily view is for previous-session history. Select an intraday timeframe for prediction.")
        key=(instrument.instrument_key,int(interval))
        now=time.time()
        cached=self.models.get(key)
        if cached and not force and now-cached.fitted_at < settings.model_retrain_minutes*60:
            return cached

        days=self._history_window(int(interval))
        history=self._get_history(instrument,int(interval),days,force=force)
        if history.empty:
            raise UpstoxError("Upstox returned no intraday history for this instrument.")
        cutoff=self._completed_cutoff()
        history_train=history.loc[history["timestamp"].dt.date <= cutoff].copy()
        context=self._get_context_history(int(interval),days,force=force)
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
        open_now=self._market_open()

        if open_now:
            # Keep the full current session visible. The WebSocket is the realtime
            # source, while the one-minute intraday endpoint supplies any session
            # history that arrived before this process subscribed.
            ws=pd.DataFrame()
            try:
                ws=self.live.live_bars(instrument.instrument_key,1)
            except Exception:
                ws=pd.DataFrame()
            rest=pd.DataFrame()
            cached=self._live_view_cache.get(cache_key)
            if cached and time.time()-cached[0] < ttl:
                return cached[1].copy()

            try:
                rest=self.upstox.get_intraday(instrument.instrument_key,1)
            except Exception:
                rest=pd.DataFrame()

            # Prefer provider-built intraday candles for OHLC integrity. The
            # WebSocket remains the live tick source and becomes a candle source
            # only when the REST intraday snapshot is unavailable.
            frames=[rest] if not rest.empty else ([ws] if not ws.empty else [])
            if frames:
                base=frames[0].copy()
                base["timestamp"]=pd.to_datetime(base["timestamp"],utc=True).dt.tz_convert(IST)
                # Only the current cash session is used for the live chart.
                today=datetime.now(ZoneInfo(IST)).date()
                mins=base["timestamp"].dt.hour*60+base["timestamp"].dt.minute
                base=base.loc[
                    (base["timestamp"].dt.date==today)
                    &(mins>=9*60+15)&(mins<15*60+30)
                ].copy()
                if not base.empty:
                    base=base.sort_values("timestamp").drop_duplicates("timestamp",keep="last")
                    aggregated=self.upstox.aggregate_minutes(base,int(interval))
                    if not aggregated.empty:
                        self._live_view_cache[cache_key]=(time.time(),aggregated.copy())
                        return aggregated.copy()

            cached=self._live_view_cache.get(cache_key)
            if cached:
                return cached[1].copy()
            return pd.DataFrame()

        # CLOSED/WEEKEND: prefer today's already-completed session if Upstox has
        # published it. This is the key distinction from the 1D history view:
        # intraday 1/5/10/15/30/60m should end with today's final cash-market bar.
        cached=self._live_view_cache.get(cache_key)
        if cached and time.time()-cached[0] < ttl:
            return cached[1].copy()

        today=datetime.now(ZoneInfo(IST)).date()
        base=pd.DataFrame()
        try:
            base=self.upstox.get_intraday(instrument.instrument_key,1)
        except Exception:
            base=pd.DataFrame()

        if base.empty:
            try:
                base=self.upstox.get_recent_history(instrument.instrument_key,1,7)
            except Exception:
                base=pd.DataFrame()

        if not base.empty:
            base["timestamp"]=pd.to_datetime(base["timestamp"],utc=True).dt.tz_convert(IST)
            mins=base["timestamp"].dt.hour*60+base["timestamp"].dt.minute
            base=base.loc[
                (mins>=9*60+15)&(mins<15*60+30)&(base["timestamp"].dt.dayofweek<5)
            ].copy()
            if not base.empty:
                session_dates=self._session_dates(base)
                # Prefer today's session when present; otherwise use the latest
                # actual session returned by the provider.
                session_date=today if today in session_dates else max(session_dates)
                base=base.loc[base["timestamp"].dt.date==session_date].copy()
                if interval == "1d":
                    # This branch is intentionally kept for completeness; the
                    # dedicated daily_view() always supplies the previous
                    # completed daily session.
                    daily=self.upstox.get_daily_history(
                        instrument.instrument_key,max(30,settings.train_days_long)
                    )
                    if not daily.empty:
                        daily["timestamp"]=pd.to_datetime(daily["timestamp"],utc=True).dt.tz_convert(IST)
                        daily=daily[daily["timestamp"].dt.date < today].tail(settings.chart_bars)
                        if not daily.empty:
                            self._live_view_cache[cache_key]=(time.time(),daily.copy())
                            return daily.copy()
                aggregated=self.upstox.aggregate_minutes(base,int(interval))
                if not aggregated.empty:
                    self._live_view_cache[cache_key]=(time.time(),aggregated.copy())
                    return aggregated.copy()

        return cached[1].copy() if cached else pd.DataFrame()

    def _merge_training_and_live(self,instrument,interval):
        history_cache=self._get_history(instrument,interval,self._history_window(interval))
        live=self._live_snapshot(instrument,interval)
        if live.empty: return history_cache,live
        combined=pd.concat([history_cache,live],ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp",keep="last").reset_index(drop=True)
        return combined,live

    def analyze(self,instrument:Instrument,interval:int,force_retrain:bool=False):
        if not self._market_open():
            raise ValueError("Prediction is available only while the Indian cash market is open (09:15–15:30 IST).")
        if int(interval) not in {1,5,10,15,30,60}:
            raise ValueError("Prediction is available for 1m, 5m, 10m, 15m, 30m and 60m. The 1D option is a history view.")
        interval=int(interval)
        key=(instrument.instrument_key,interval)
        with self._analysis_lock:
            # Reuse a recent result instead of refitting/re-hitting external services.
            # Once the cash session is closed, freeze the last result for that
            # instrument/timeframe until the next trading session. This keeps
            # the displayed probability from drifting after the market closes.
            cached=self.last_results.get(key)
            today=datetime.now(ZoneInfo(IST)).date()
            if cached and not force_retrain:
                age=time.time()-cached["_created_at"]
                cached_session=cached.get("session_date")
                if not self._market_open():
                    return {k:v for k,v in cached.items() if k!="_created_at"}
                if cached_session != str(today):
                    cached=None
                elif age < settings.prediction_refresh_seconds:
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
                "session_date":str(datetime.now(ZoneInfo(IST)).date()),
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

    def daily_view(self, instrument: Instrument):
        now=datetime.now(ZoneInfo(IST))
        cache_key=(instrument.instrument_key,"1d")
        bars=pd.DataFrame()
        cached=self._live_view_cache.get(cache_key)
        if cached and time.time()-cached[0] < settings.live_rest_cache_seconds:
            bars=cached[1].copy()
        else:
            try:
                daily=self.upstox.get_daily_history(
                    instrument.instrument_key,max(30,settings.train_days_long)
                )
                if not daily.empty:
                    daily["timestamp"]=pd.to_datetime(daily["timestamp"],utc=True).dt.tz_convert(IST)
                    # 1D is explicitly the previous-session/history view:
                    # never substitute today's intraday session for this mode.
                    daily=daily[daily["timestamp"].dt.date < now.date()].tail(settings.chart_bars)
                    bars=daily.copy()
                    self._live_view_cache[cache_key]=(time.time(),bars.copy())
            except Exception:
                bars=pd.DataFrame()
        rows=[]
        if not bars.empty:
            for r in bars.tail(settings.chart_bars).itertuples():
                rows.append({"time":r.timestamp.isoformat(),"open":float(r.open),"high":float(r.high),"low":float(r.low),"close":float(r.close),"volume":float(r.volume or 0.0)})
            last=bars.iloc[-1]
            lts=pd.Timestamp(last.timestamp).tz_convert(IST)
            tick={"price":float(last.close),"trade_time":lts.isoformat(),"updated_at":lts.isoformat(),"instrument_key":instrument.instrument_key,"session_trade":False}
        else:
            tick=self.live.latest(instrument.instrument_key)
            if tick is None:
                try:
                    q=next(iter(self.upstox.get_quote_batch([instrument.instrument_key]).values()),{})
                    lp=q.get("last_price") or (q.get("ohlc") or {}).get("close")
                    tick={"price":float(lp),"trade_time":None,"updated_at":now.isoformat(),"instrument_key":instrument.instrument_key,"session_trade":False} if lp is not None else None
                except Exception:
                    tick=None
        status="OPEN" if self._market_open() else ("WEEKEND" if now.weekday()>=5 else "CLOSED")
        last_candle=None
        if not bars.empty:
            row=bars.iloc[-1]; ts=pd.Timestamp(row.timestamp).tz_convert(IST)
            last_candle={"start":ts.isoformat(),"end":(ts+pd.Timedelta(days=1)).isoformat(),"open":float(row.open),"high":float(row.high),"low":float(row.low),"close":float(row.close),"volume":float(row.volume or 0.0),"forming":False}
        age=None
        if tick and tick.get("updated_at"):
            try: age=round(max(0,(now-datetime.fromisoformat(tick["updated_at"])).total_seconds()),1)
            except Exception: pass
        return {"now":now.isoformat(),"instrument_key":instrument.instrument_key,"symbol":instrument.trading_symbol,"tick":tick,"candles":rows,"current_candle":last_candle,"seconds_to_candle_close":0,"prediction":None,"view_only":True,"market_data":{"stream_connected":self.live.connected,"stream_error":self.live.last_error,"last_update_age_seconds":age,"source":"daily-history","market_status":status,"last_market_price":float(bars.iloc[-1]["close"]) if not bars.empty else (float(tick["price"]) if tick else None),"last_market_candle":bars.iloc[-1]["timestamp"].isoformat() if not bars.empty else None}}

    def live_state(self,instrument:Instrument,interval):
        if interval == "1d":
            return self.daily_view(instrument)
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
                "source":"websocket" if self.live.connected and tick and open_now else "last-completed-session" if bars.size else "unavailable",
                "market_status":status,
                "last_market_price":float(bars.iloc[-1]["close"]) if not bars.empty else None,
                "last_market_candle":bars.iloc[-1]["timestamp"].isoformat() if not bars.empty else None,
            }
        }
