from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class ModelReport:
    direction_accuracy: float
    calibrated_brier: float
    signal_precision: float
    signal_coverage: float
    signal_count: int
    precision_lower_bound: float
    next_return_mae: float
    threshold: float
    gate_90_passed: bool
    train_rows: int
    validation_rows: int
    fit_timestamp: str

class CandleModel:
    def __init__(self, min_train_rows: int = 300):
        self.min_train_rows = min_train_rows
        self.direction_model = None
        self.return_model = None
        self.high_model = None
        self.low_model = None
        self.eod_model = None
        self.features: list[str] = []
        self.report: ModelReport | None = None

    @staticmethod
    def _adaptive_probability(base_p: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        x = frame
        atr = np.maximum(np.abs(x.get("atr_pct", 0.001).to_numpy(dtype=float)), 0.00025)
        r3 = x.get("return_3", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        r10 = x.get("return_10", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        ema = x.get("ema_spread", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        nifty = x.get("nifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        bank = x.get("banknifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        vix = x.get("vix_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)

        regime = (
            0.38*np.tanh(r3/(2.0*atr))
            + 0.24*np.tanh(r10/(4.0*atr))
            + 0.17*np.tanh(ema/(2.4*atr))
            + 0.11*np.tanh(nifty/(2.7*atr))
            + 0.10*np.tanh(bank/(2.7*atr))
            - 0.08*np.tanh(vix/(3.0*atr))
        )
        regime_prob = 0.5 + 0.5*np.clip(regime, -1.0, 1.0)
        # Keep the trained model primary, but let strong current market direction
        # move the probability enough to avoid stale-regime decisions.
        return np.clip(0.70*np.asarray(base_p, dtype=float) + 0.30*regime_prob, 0.01, 0.99)

    @staticmethod
    def _trend_return(frame: pd.DataFrame) -> np.ndarray:
        x = frame
        atr = np.maximum(np.abs(x.get("atr_pct", 0.001).to_numpy(dtype=float)), 0.00025)
        r3 = x.get("return_3", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        r10 = x.get("return_10", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        ema = x.get("ema_spread", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        context = 0.5*x.get("nifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float) + 0.5*x.get("banknifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        score = 0.60*np.tanh(r3/(2.0*atr)) + 0.25*np.tanh(r10/(4.0*atr)) + 0.15*np.tanh((ema+context)/(3.0*atr))
        return np.clip(score*0.60*atr, -0.008, 0.008)

    @staticmethod
    def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
        if total <= 0:
            return 0.0
        p = successes / total
        denom = 1 + z*z/total
        centre = p + z*z/(2*total)
        adj = z*math.sqrt((p*(1-p)+z*z/(4*total))/total)
        return (centre-adj)/denom

    @classmethod
    def _threshold(cls, y: np.ndarray, p: np.ndarray):
        best = (0.70, 0.0, 0.0, 0, 0.0)
        n = len(y)
        minimum = max(30, int(n*0.03))
        for t in np.arange(0.55, 0.96, 0.01):
            long_mask = p >= t
            short_mask = p <= 1-t
            mask = long_mask | short_mask
            signals = int(mask.sum())
            if signals < minimum:
                continue
            preds = np.where(long_mask[mask], 1, 0)
            truth = y[mask]
            wins = int((preds == truth).sum())
            precision = float(precision_score(truth, preds, zero_division=0))
            coverage = signals/n if n else 0.0
            lower = cls._wilson_lower(wins, signals)
            # Prioritize statistical credibility over a threshold that only
            # looks good because it selected a tiny validation subset.
            score = lower + 0.05*math.sqrt(coverage)
            best_score = best[4] + 0.05*math.sqrt(best[2]) if best[3] else -1
            if score > best_score:
                best = (float(t), precision, coverage, signals, lower)
        return best

    @staticmethod
    def _fit_classifier(X, y):
        if len(np.unique(y)) < 2:
            raise ValueError("Training window contains one direction class.")
        base = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=140, max_leaf_nodes=24,
            l2_regularization=0.20, min_samples_leaf=15, random_state=42,
        )
        return CalibratedClassifierCV(base, cv=TimeSeriesSplit(n_splits=3), method="sigmoid")

    @staticmethod
    def _fit_regressor(X, y):
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=140, max_leaf_nodes=24,
            l2_regularization=0.15, loss="absolute_error", random_state=42,
        )
        et = ExtraTreesRegressor(
            n_estimators=90, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42
        )
        class Ensemble:
            def __init__(self, a, b): self.a, self.b = a, b
            def fit(self, Xf, yf): self.a.fit(Xf, yf); self.b.fit(Xf, yf); return self
            def predict(self, Xf): return 0.65*self.a.predict(Xf) + 0.35*self.b.predict(Xf)
        return Ensemble(hgb, et)

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> ModelReport:
        if len(train_df) < self.min_train_rows:
            raise ValueError(f"Need at least {self.min_train_rows} clean intraday rows; got {len(train_df)}.")
        self.features = list(features)
        data = train_df.sort_values("timestamp").reset_index(drop=True)
        # Walk forward by date. Validation is kept strictly later than the train split.
        sessions = data["timestamp"].dt.date.drop_duplicates().tolist()
        split_idx = max(1, int(len(sessions)*0.80))
        split_day = sessions[min(split_idx, len(sessions)-1)]
        train = data.loc[data["timestamp"].dt.date < split_day].copy()
        val = data.loc[data["timestamp"].dt.date >= split_day].copy()
        if len(val) < max(60, int(len(data)*0.12)) or train.empty:
            cut = int(len(data)*0.78)
            train, val = data.iloc[:cut].copy(), data.iloc[cut:].copy()
        if len(np.unique(train["direction"].astype(int))) < 2:
            raise ValueError("Training window contains one direction class; widen the history.")
        if len(np.unique(val["direction"].astype(int))) < 2:
            # Validation is allowed to be one-class; precision still remains inspectable.
            pass

        X_train, X_val = train[self.features], val[self.features]
        y_train, y_val = train["direction"].astype(int), val["direction"].astype(int)
        direction = self._fit_classifier(X_train, y_train)
        direction.fit(X_train, y_train)
        raw_p = direction.predict_proba(X_val)[:,1]
        p_val = self._adaptive_probability(raw_p, val)
        threshold, precision, coverage, signals, lower = self._threshold(y_val.to_numpy(), p_val)
        acc = accuracy_score(y_val, (p_val >= 0.5).astype(int))
        brier = brier_score_loss(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.25

        def fit_reg(target):
            reg = self._fit_regressor(X_train, train[target])
            reg.fit(X_train, train[target])
            mae = mean_absolute_error(val[target], reg.predict(X_val))
            return reg, float(mae)

        ret_model, ret_mae = fit_reg("next_return")
        hi_model, _ = fit_reg("next_high_return")
        lo_model, _ = fit_reg("next_low_return")
        eod_model, _ = fit_reg("eod_return")

        self.direction_model = self._fit_classifier(data[self.features], data["direction"].astype(int)).fit(data[self.features], data["direction"].astype(int))
        self.return_model = self._fit_regressor(data[self.features], data["next_return"]).fit(data[self.features], data["next_return"])
        self.high_model = self._fit_regressor(data[self.features], data["next_high_return"]).fit(data[self.features], data["next_high_return"])
        self.low_model = self._fit_regressor(data[self.features], data["next_low_return"]).fit(data[self.features], data["next_low_return"])
        self.eod_model = self._fit_regressor(data[self.features], data["eod_return"]).fit(data[self.features], data["eod_return"])

        self.report = ModelReport(
            direction_accuracy=float(acc),
            calibrated_brier=float(brier),
            signal_precision=float(precision),
            signal_coverage=float(coverage),
            signal_count=int(signals),
            precision_lower_bound=float(lower),
            next_return_mae=float(ret_mae),
            threshold=float(threshold),
            # Predictions are visible regardless; BUY/SELL requires a statistically
            # defensible validation sample.
            gate_90_passed=bool(precision >= 0.90 and lower >= 0.90 and signals >= 30 and coverage >= 0.02),
            train_rows=len(train),
            validation_rows=len(val),
            fit_timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        return self.report

    def predict(self, latest_row: pd.DataFrame, current_price: float, news_impact: float = 0.0) -> dict:
        if self.report is None or self.direction_model is None:
            raise RuntimeError("Model is not trained.")
        X = latest_row[self.features]
        raw_p = float(self.direction_model.predict_proba(X)[0,1])
        p_up = float(self._adaptive_probability(np.array([raw_p]), latest_row)[0])
        news_prob = 0.5 + 0.5*float(np.clip(news_impact, -1, 1))
        p_up = float(np.clip(0.92*p_up + 0.08*news_prob, 0.01, 0.99))

        raw_return = float(self.return_model.predict(X)[0])
        high_return = float(self.high_model.predict(X)[0])
        low_return = float(self.low_model.predict(X)[0])
        eod_return = float(self.eod_model.predict(X)[0])
        trend_overlay = float(self._trend_return(latest_row)[0])

        # News is deliberately a small adjustment. Price/market structure remains primary.
        news_overlay = float(np.clip(news_impact, -1, 1) * max(abs(raw_return), 0.0008) * 0.15)
        adjusted_return = 0.82*raw_return + 0.18*trend_overlay + news_overlay
        predicted_close = current_price*(1+adjusted_return)
        predicted_high = current_price*(1+max(0.0, high_return, adjusted_return))
        predicted_low = current_price*(1+min(0.0, low_return, adjusted_return))
        predicted_eod = current_price*(1+eod_return+news_overlay*0.25)

        t = self.report.threshold
        buy = p_up >= t
        sell = p_up <= 1-t
        if self.report.gate_90_passed and buy and news_impact > -0.55:
            signal = "BUY"
        elif self.report.gate_90_passed and sell and news_impact < 0.55:
            signal = "SELL"
        else:
            signal = "REVALIDATE"

        return {
            "signal": signal,
            "probability_up": round(p_up, 4),
            "probability_down": round(1-p_up, 4),
            "raw_return": round(raw_return, 6),
            "trend_adjustment": round(trend_overlay, 6),
            "news_adjusted_return": round(adjusted_return, 6),
            "predicted_close": round(predicted_close, 4),
            "predicted_high": round(max(predicted_high, predicted_close), 4),
            "predicted_low": round(min(predicted_low, predicted_close), 4),
            "eod_return": round(eod_return, 6),
            "predicted_eod_close": round(predicted_eod, 4),
            "validation": {
                "direction_accuracy": round(self.report.direction_accuracy*100, 2),
                "calibrated_brier": round(self.report.calibrated_brier, 5),
                "signal_precision": round(self.report.signal_precision*100, 2),
                "signal_coverage": round(self.report.signal_coverage*100, 2),
                "signal_count": self.report.signal_count,
                "precision_lower_bound": round(self.report.precision_lower_bound*100, 2),
                "next_return_mae": round(self.report.next_return_mae, 6),
                "threshold": round(self.report.threshold, 2),
                "gate_90_passed": self.report.gate_90_passed,
                "train_rows": self.report.train_rows,
                "validation_rows": self.report.validation_rows,
                "fit_timestamp": self.report.fit_timestamp,
            },
        }
