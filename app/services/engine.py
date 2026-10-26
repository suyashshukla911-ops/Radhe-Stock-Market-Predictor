from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import CACHE_DIR, IST, SECTOR_BY_SYMBOL, settings
from app.data.news import NewsService
from app.data.upstox import Instrument, LiveMarketHub, UpstoxClient, UpstoxError
from app.ml.features import FEATURES, build_features, prepare_training_frame
from app.ml.model import CandleModel

@dataclass
class ModelCache:
    model: CandleModel
    fitted_at: float
    history: pd.DataFrame
    context_history: dict[str, pd.DataFrame]

class AnalysisEngine:
    def __init__(self, upstox: UpstoxClient, live: LiveMarketHub, news: NewsService):
        self.upstox = upstox
        self.live = live
        self.news = news
        self.models = {}
        self.last_results = {}
        self._history_cache = {}
        self._context_cache = {}
        self._live_view_cache = {}
        self._rest_live_cache = {}
        self._training = set()
        self._train_guard = threading.Lock()
        self._train_error = {}
        self._eod_cache = {}
        self._morning_cache = {}
        self._analysis_lock = threading.RLock()
        cache_dir = Path(os.getenv("CACHE_DIR") or os.getenv("DATA_DIR") or CACHE_DIR)
        self._session_state_file = cache_dir / 'eod_outlook.json'
        self._session_state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_session_state()

    def _load_session_state(self):
        try:
            cutoff = (datetime.now(ZoneInfo(IST)).date() - timedelta(days=7)).isoformat()
            obj = json.loads(self._session_state_file.read_text(encoding='utf-8'))
            self._eod_cache.update({k: v for k, v in obj.get('eod', {}).items() if k.split('|')[-1] >= cutoff})
            self._morning_cache.update({k: v for k, v in obj.get('morning', {}).items() if k.split('|')[-1] >= cutoff})
        except Exception:
            pass
    _load_state = _load_session_state

    def _save_session_state(self):
        try:
            cutoff = (datetime.now(ZoneInfo(IST)).date() - timedelta(days=7)).isoformat()
            eod_pruned = {k: v for k, v in self._eod_cache.items() if k.split('|')[-1] >= cutoff}
            morning_pruned = {k: v for k, v in self._morning_cache.items() if k.split('|')[-1] >= cutoff}
            self._session_state_file.write_text(json.dumps({'eod': eod_pruned, 'morning': morning_pruned}, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass
    _save_state = _save_session_state

    @staticmethod
    def _market_open():
        return UpstoxClient.is_market_open()

    @staticmethod
    def _today():
        return str(datetime.now(ZoneInfo(IST)).date())

    def _history_window(self, interval):
        return max(365, settings.train_days_long * 4) if int(interval) > 15 else settings.train_days_short

    def _resolve_context(self):
        out = {}
        for label, name in [('nifty', 'NIFTY 50'), ('banknifty', 'NIFTY BANK'), ('vix', 'INDIA VIX')]:
            try:
                out[label] = self.upstox.resolve_symbol(name)
            except Exception:
                pass
        return out

    def _get_history(self, instrument, interval, calendar_days, force=False):
        key = (instrument.instrument_key, int(interval))
        cached = self._history_cache.get(key)
        if cached and not force and time.time() - cached[0] < settings.history_cache_minutes * 60:
            return cached[1].copy()
        hist = self.upstox.get_recent_history(instrument.instrument_key, int(interval), calendar_days)
        self._history_cache[key] = (time.time(), hist.copy())
        return hist

    def _get_context_history(self, interval, days, force=False):
        key = (int(interval), days)
        cached = self._context_cache.get(key)
        if cached and not force and time.time() - cached[0] < settings.history_cache_minutes * 60:
            return {k: v.copy() for k, v in cached[1].items()}
        out = {}
        for label, ins in self._resolve_context().items():
            try:
                out[label] = self.upstox.get_recent_history(ins.instrument_key, int(interval), days)
            except Exception:
                out[label] = pd.DataFrame()
        self._context_cache[key] = (time.time(), out)
        return {k: v.copy() for k, v in out.items()}

    def _get_training_cache(self, instrument, interval, force=False):
        key = (instrument.instrument_key, int(interval))
        now = time.time()
        cached = self.models.get(key)
        if cached and not force and now - cached.fitted_at < settings.model_retrain_minutes * 60:
            return cached
        history = self._get_history(instrument, int(interval), self._history_window(int(interval)), force=force)
        if history.empty:
            raise UpstoxError('Upstox returned no intraday history for this instrument.')
        cutoff = datetime.now(ZoneInfo(IST)).date()
        history_train = history.loc[history['timestamp'].dt.date < cutoff].copy()
        context = self._get_context_history(int(interval), self._history_window(int(interval)), force=force)
        context_train = {}
        for k in ('nifty', 'banknifty', 'vix'):
            v = context.get(k)
            context_train[k] = (v.loc[v['timestamp'].dt.date < cutoff].copy()
                              if v is not None and not v.empty and 'timestamp' in v.columns else pd.DataFrame())
        train = prepare_training_frame(history_train, context=context_train)
        if len(train) < settings.min_train_rows:
            raise ValueError(f'Not enough clean training rows for {instrument.trading_symbol} at {interval}m: raw={len(history_train)}, clean={len(train)}, need={settings.min_train_rows}.')
        model = CandleModel(settings.min_train_rows)
        model.fit(train, FEATURES)
        cache = ModelCache(model, time.time(), history_train, context_train)
        self.models[key] = cache
        return cache

    def _retrain_async(self, instrument, interval):
        key = (instrument.instrument_key, int(interval))
        with self._train_guard:
            if key in self._training:
                return
            self._training.add(key)
        def run():
            try:
                self._get_training_cache(instrument, int(interval), force=True)
                self._train_error.pop(key, None)
            except Exception as exc:
                self._train_error[key] = str(exc)
            finally:
                with self._train_guard:
                    self._training.discard(key)
        threading.Thread(target=run, name=f"train-{key[1]}m", daemon=True).start()

    def _model_for_request(self, instrument, interval, force=False):
        """Never train inside an HTTP request: serve the cached model and refresh it in the background."""
        key = (instrument.instrument_key, int(interval))
        cached = self.models.get(key)
        fresh = bool(cached) and time.time() - cached.fitted_at < settings.model_retrain_minutes * 60
        if cached and fresh and not force:
            return cached
        self._retrain_async(instrument, interval)
        if cached:
            return cached
        err = self._train_error.pop(key, None)
        if err:
            raise ValueError(err)
        raise ValueError(f'Model for {instrument.trading_symbol} {int(interval)}m is training in the background (about a minute). It will appear automatically.')

    def _live_snapshot(self, instrument, interval, ttl_seconds=None):
        ttl = settings.live_rest_cache_seconds if ttl_seconds is None else ttl_seconds
        key = (instrument.instrument_key, int(interval))
        open_now = self._market_open()
        self.live.subscribe([instrument.instrument_key])
        ws = pd.DataFrame()
        try:
            ws = self.live.live_bars(instrument.instrument_key, 1)
        except Exception:
            pass

        rest_cached = self._rest_live_cache.get(instrument.instrument_key)
        rest_age = (time.time() - rest_cached[0]) if rest_cached else 999999
        rest = pd.DataFrame()
        if (not rest_cached) or rest_age >= ttl:
            try:
                rest = self.upstox.get_intraday(instrument.instrument_key, 1)
                if not rest.empty:
                    self._rest_live_cache[instrument.instrument_key] = (time.time(), rest.copy())
                elif not rest_cached:
                    self._rest_live_cache[instrument.instrument_key] = (time.time(), pd.DataFrame())
            except Exception:
                if rest_cached:
                    rest = rest_cached[1].copy()
                else:
                    self._rest_live_cache[instrument.instrument_key] = (time.time(), pd.DataFrame())
        elif rest_cached:
            rest = rest_cached[1].copy()

        parts = [x for x in [rest, ws] if not x.empty]
        cached = self._live_view_cache.get(key)
        if not parts and cached:
            return cached[1].copy()
        if not parts:
            if open_now:
                return pd.DataFrame()
            try:
                hist = self.upstox.get_recent_history(instrument.instrument_key, 1, 7)
                hist['timestamp'] = pd.to_datetime(hist['timestamp'], utc=True).dt.tz_convert(IST)
                mins = hist.timestamp.dt.hour * 60 + hist.timestamp.dt.minute
                hist = hist.loc[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30) & (hist.timestamp.dt.dayofweek < 5)].copy()
                if hist.empty:
                    return pd.DataFrame()
                session = max(hist.timestamp.dt.date.unique())
                base = hist.loc[hist.timestamp.dt.date == session].copy()
            except Exception:
                return pd.DataFrame()
        else:
            base = pd.concat(parts, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp', keep='last')
            base['timestamp'] = pd.to_datetime(base['timestamp'], utc=True).dt.tz_convert(IST)
            today = datetime.now(ZoneInfo(IST)).date()
            mins = base.timestamp.dt.hour * 60 + base.timestamp.dt.minute
            if open_now:
                base = base.loc[(base.timestamp.dt.date == today) & (mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)].copy()
            else:
                base = base.loc[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30) & (base.timestamp.dt.dayofweek < 5)].copy()
                if not base.empty:
                    sessions = sorted(base.timestamp.dt.date.unique())
                    session = today if today in sessions else sessions[-1]
                    base = base.loc[base.timestamp.dt.date == session].copy()
        out = self.upstox.aggregate_minutes(base, int(interval))
        if not out.empty:
            self._live_view_cache[key] = (time.time(), out.copy())
        return out.copy()

    def _merge_training_and_live(self, instrument, interval):
        h = self._get_history(instrument, interval, self._history_window(interval))
        live = self._live_snapshot(instrument, interval)
        if live.empty:
            return h, live
        return pd.concat([h, live], ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True), live

    def _forecast_path(self, model_cache, combined, context_live, interval, current_price, news_impact, steps=2):
        feat = build_features(combined, context=context_live).dropna(subset=FEATURES)
        if feat.empty:
            return []
        latest = feat.iloc[[-1]]
        pred = model_cache.model.predict(latest, current_price, news_impact=news_impact)
        start_time = pd.Timestamp(combined.timestamp.iloc[-1]).tz_convert(IST)

        t1_time = start_time + pd.to_timedelta(int(interval), unit="m")
        t1_close = pred['predicted_close']
        f1 = {'time': t1_time.isoformat(), 'open': round(current_price, 4), 'high': pred['predicted_high'], 'low': pred['predicted_low'], 'close': t1_close, 'step': 1}

        t2_time = t1_time + pd.to_timedelta(int(interval), unit="m")
        t2_close = pred.get('predicted_t2_close', round(t1_close * (1 + pred['raw_return'] * 0.95), 4))
        t2_high = pred.get('predicted_t2_high', round(max(t1_close, t2_close), 4))
        t2_low = pred.get('predicted_t2_low', round(min(t1_close, t2_close), 4))
        f2 = {'time': t2_time.isoformat(), 'open': round(t1_close, 4), 'high': t2_high, 'low': t2_low, 'close': t2_close, 'step': 2}
        return [f1, f2]

    def analyze(self, instrument: Instrument, interval: int, force_retrain=False):
        if not self._market_open():
            raise ValueError('Prediction is available only while the Indian cash market is open (09:15–15:30 IST).')
        if int(interval) not in {1, 5, 10, 15, 30, 60}:
            raise ValueError('Prediction timeframes: 1m, 5m, 10m, 15m, 30m and 60m.')
        key = (instrument.instrument_key, int(interval))
        today = self._today()
        with self._analysis_lock:
            cached = self.last_results.get(key)
            if cached and not force_retrain and cached.get('session_date') == today and time.time() - cached['_created_at'] < settings.prediction_refresh_seconds:
                return {k: v for k, v in cached.items() if k != '_created_at'}
            mc = self._model_for_request(instrument, interval, force=force_retrain)
            combined, live = self._merge_training_and_live(instrument, interval)
            if live.empty:
                raise UpstoxError('Current-session candle snapshot is unavailable. Waiting for the live market feed.')
            context_live = {}
            for label, ins in self._resolve_context().items():
                try:
                    context_live[label] = self._live_snapshot(ins, interval)
                except Exception:
                    context_live[label] = pd.DataFrame()
            features = build_features(combined, context=context_live).dropna(subset=FEATURES)
            if features.empty:
                raise ValueError('Current market data does not contain enough feature history yet.')
            latest = features.iloc[[-1]]
            tick = self.live.latest(instrument.instrument_key)
            current_price = float(tick['price']) if tick and tick.get('price') is not None else float(latest['close'].iloc[0])
            news = self.news.snapshot(instrument.trading_symbol, instrument.name or instrument.trading_symbol)
            impact = float(news['summary'].get('impact', 0.0))
            live_pred = mc.model.predict(latest, current_price, news_impact=impact)
            path = self._forecast_path(mc, combined, context_live, interval, current_price, impact, steps=2)
            if path:
                live_pred['predicted_close'] = path[0]['close']
                live_pred['predicted_high'] = path[0]['high']
                live_pred['predicted_low'] = path[0]['low']
            eod_key = f'{instrument.instrument_key}|{today}'
            if eod_key not in self._eod_cache:
                today_date = datetime.now(ZoneInfo(IST)).date()
                today_bars = live.loc[live.timestamp.dt.date == today_date] if not live.empty else pd.DataFrame()
                opening_ref = float(today_bars.iloc[0].open) if not today_bars.empty else current_price
                now_ist = datetime.now(ZoneInfo(IST))
                is_late = (now_ist.hour * 60 + now_ist.minute) > 11 * 60
                self._eod_cache[eod_key] = {
                    'predicted_eod_close': float(live_pred['predicted_eod_close']),
                    'eod_return': float(live_pred['eod_return']),
                    'opening_reference_price': opening_ref,
                    'price_at_lock': float(current_price),
                    'news_impact': impact,
                    'created_at': now_ist.isoformat(),
                    'late': is_late,
                    'locked_at_time': now_ist.strftime("%H:%M"),
                }
                self._morning_cache[eod_key] = {
                    'news_headline_count': news['summary'].get('headline_count', 0),
                    'news_impact': impact,
                    'created_at': now_ist.isoformat(),
                }
                self._save_session_state()
            eod = self._eod_cache[eod_key]
            live_pred['predicted_eod_close'] = eod['predicted_eod_close']
            live_pred['eod_return'] = eod['eod_return']
            rows = [{'time': r.timestamp.isoformat(), 'open': float(r.open), 'high': float(r.high), 'low': float(r.low), 'close': float(r.close), 'volume': float(r.volume or 0)} for r in live.tail(settings.chart_bars).itertuples()]
            ref_price = eod.get('price_at_lock') or eod.get('opening_reference_price') or current_price
            change_pct = round((eod['predicted_eod_close'] / ref_price - 1) * 100, 2)
            result = {
                'symbol': instrument.trading_symbol,
                'instrument': {'name': instrument.name, 'short_name': instrument.short_name, 'symbol': instrument.trading_symbol, 'exchange': instrument.exchange, 'segment': instrument.segment, 'instrument_key': instrument.instrument_key, 'sector': SECTOR_BY_SYMBOL.get(instrument.trading_symbol.upper(), 'Market Index' if instrument.segment.endswith('INDEX') else 'Unknown')},
                'interval': int(interval), 'session_date': today, 'last_timestamp': latest.timestamp.iloc[0].isoformat(), 'current_price': current_price,
                'actual_candles': rows, 'forecast_candles': path, 'candles': rows, 'prediction': live_pred, 'news': news,
                'morning_outlook': {'eod_close': eod['predicted_eod_close'], 'change_pct': change_pct, 'news_impact': eod['news_impact'], 'fixed_since': eod['created_at'], 'late': eod.get('late', False), 'locked_at_time': eod.get('locked_at_time', '')},
                'market_state': {'stream_connected': self.live.connected, 'stream_error': self.live.last_error, 'history_rows': len(mc.history), 'training_rows': live_pred['validation']['train_rows'], 'live_rows': len(live), 'model_cache_age_seconds': round(time.time() - mc.fitted_at, 1), 'prediction_refresh_seconds': settings.prediction_refresh_seconds, 'model_retrain_minutes': settings.model_retrain_minutes, 'market_status': 'OPEN'},
                'refresh_policy': {'browser_seconds': settings.live_poll_seconds, 'prediction_seconds': settings.prediction_refresh_seconds, 'model_retrain_minutes': settings.model_retrain_minutes, 'toi_refresh_minutes': settings.toi_refresh_minutes, 'news_minutes': settings.news_refresh_minutes, 'moneycontrol_minutes': settings.moneycontrol_refresh_minutes},
                '_created_at': time.time()
            }
            self.last_results[key] = result
            return {k: v for k, v in result.items() if k != '_created_at'}

    def daily_view(self, instrument: Instrument):
        now = datetime.now(ZoneInfo(IST))
        clock = UpstoxClient.market_clock_status(now)
        bars = pd.DataFrame()
        try:
            daily = self.upstox.get_daily_history(instrument.instrument_key, max(60, settings.train_days_long * 2))
            if not daily.empty:
                daily['timestamp'] = pd.to_datetime(daily['timestamp'], utc=True).dt.tz_convert(IST)
                bars = daily[daily.timestamp.dt.date < now.date()].tail(settings.chart_bars).copy()
        except Exception:
            pass
        rows = [{'time': r.timestamp.isoformat(), 'open': float(r.open), 'high': float(r.high), 'low': float(r.low), 'close': float(r.close), 'volume': float(r.volume or 0)} for r in bars.itertuples()]
        tick = self.live.latest(instrument.instrument_key)
        if bars.empty:
            try:
                q = next(iter(self.upstox.get_quote_batch([instrument.instrument_key]).values()), {})
                lp = q.get('last_price') or (q.get('ohlc') or {}).get('close')
                tick = {'price': float(lp), 'trade_time': None, 'updated_at': now.isoformat(), 'instrument_key': instrument.instrument_key, 'session_trade': False} if lp is not None else tick
            except Exception:
                pass
        if not bars.empty:
            last = bars.iloc[-1]
            ts = pd.Timestamp(last.timestamp).tz_convert(IST)
            tick = {'price': float(last.close), 'trade_time': ts.isoformat(), 'updated_at': ts.isoformat(), 'instrument_key': instrument.instrument_key, 'session_trade': False}
            candle = {'start': ts.isoformat(), 'end': (ts + pd.to_timedelta(1, unit="D")).isoformat(), 'open': float(last.open), 'high': float(last.high), 'low': float(last.low), 'close': float(last.close), 'volume': float(last.volume or 0), 'forming': False}
        else:
            candle = None
        age = None
        if tick and tick.get('updated_at'):
            try:
                age = round(max(0, (now - datetime.fromisoformat(tick['updated_at'])).total_seconds()), 1)
            except Exception:
                pass
        data_status = "OK" if not bars.empty or self.upstox.connected else "NO_TOKEN"
        return {
            'now': now.isoformat(), 'instrument_key': instrument.instrument_key, 'symbol': instrument.trading_symbol,
            'tick': tick, 'candles': rows, 'current_candle': candle, 'seconds_to_candle_close': 0, 'prediction': None, 'view_only': True,
            'market_data': {
                'stream_connected': self.live.connected, 'stream_error': self.live.last_error, 'last_update_age_seconds': age,
                'source': 'previous-session-history', 'market_status': clock["market_status"], 'market_clock_label': clock["label"],
                'data_status': data_status, 'data_message': 'Previous session daily history',
                'last_market_price': float(bars.iloc[-1]['close']) if not bars.empty else (float(tick['price']) if tick else None),
                'last_market_candle': bars.iloc[-1]['timestamp'].isoformat() if not bars.empty else None
            }
        }

    def live_state(self, instrument: Instrument, interval):
        if interval == '1d':
            return self.daily_view(instrument)
        now = datetime.now(ZoneInfo(IST))
        clock = UpstoxClient.market_clock_status(now)
        open_now = clock["is_open"]
        self.live.subscribe([instrument.instrument_key])
        bars = self._live_snapshot(instrument, int(interval), ttl_seconds=settings.live_rest_cache_seconds)
        tick = self.live.latest(instrument.instrument_key)
        rows = [{'time': r.timestamp.isoformat(), 'open': float(r.open), 'high': float(r.high), 'low': float(r.low), 'close': float(r.close), 'volume': float(r.volume or 0)} for r in bars.tail(settings.chart_bars).itertuples()] if not bars.empty else []
        if tick is None and not bars.empty:
            last = bars.iloc[-1]
            ts = pd.Timestamp(last.timestamp).tz_convert(IST)
            tick = {'price': float(last.close), 'trade_time': ts.isoformat(), 'updated_at': ts.isoformat(), 'instrument_key': instrument.instrument_key, 'session_trade': False}
        current = None
        secs = 0
        if not bars.empty:
            row = bars.iloc[-1]
            start = pd.Timestamp(row.timestamp).tz_convert(IST)
            end = min(start + pd.to_timedelta(int(interval), unit="m"), start.normalize() + pd.to_timedelta(15 * 60 + 30, unit="m"))
            forming = bool(open_now and start.date() == now.date() and start <= pd.Timestamp(now) < end)
            current = {'start': start.isoformat(), 'end': end.isoformat(), 'open': float(row.open), 'high': float(row.high), 'low': float(row.low), 'close': float(row.close), 'volume': float(row.volume or 0), 'forming': forming}
            if forming:
                secs = max(0, int((end.to_pydatetime() - now).total_seconds()))
        age = None
        if tick and tick.get('updated_at'):
            try:
                age = round(max(0, (now - datetime.fromisoformat(tick['updated_at'])).total_seconds()), 1)
            except Exception:
                pass

        if not self.upstox.access_token:
            data_status = "NO_TOKEN"
            data_msg = "Upstox access token missing. Please connect Upstox."
        elif self.live.last_error and any(x in self.live.last_error.lower() for x in ("401", "403", "unauthorized", "token", "expired")):
            data_status = "TOKEN_EXPIRED"
            data_msg = "Upstox session expired. Please re-login."
        elif open_now and not self.live.connected and bars.empty:
            data_status = "FEED_DOWN"
            data_msg = "Live feed recovering. Waiting for market data."
        else:
            data_status = "OK"
            data_msg = "Feed operational"

        last_result = self.last_results.get((instrument.instrument_key, int(interval)))
        last_price = (float(tick['price']) if tick and tick.get('price') is not None else (float(bars.iloc[-1]['close']) if not bars.empty else None))
        return {
            'now': now.isoformat(), 'instrument_key': instrument.instrument_key, 'symbol': instrument.trading_symbol,
            'tick': tick, 'candles': rows, 'current_candle': current, 'seconds_to_candle_close': secs,
            'prediction': last_result.get('prediction') if last_result else None,
            'forecast_candles': last_result.get('forecast_candles', []) if last_result else [],
            'market_data': {
                'stream_connected': self.live.connected, 'stream_error': self.live.last_error, 'last_update_age_seconds': age,
                'source': 'upstox-websocket' if self.live.connected and tick and open_now else 'last-session-fallback',
                'market_status': clock["market_status"], 'market_clock_label': clock["label"],
                'data_status': data_status, 'data_message': data_msg,
                'last_market_price': last_price, 'last_market_candle': bars.iloc[-1]['timestamp'].isoformat() if not bars.empty else None
            }
        }
