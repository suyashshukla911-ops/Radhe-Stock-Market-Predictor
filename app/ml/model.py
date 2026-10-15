from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ModelReport:
    direction_accuracy: float
    calibrated_brier: float
    signal_precision: float
    signal_coverage: float
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
        """Blend calibrated model probability with a small live-regime component.

        The same transform is used during validation and live inference, so the 90%
        selective-precision gate measures the probability actually used for decisions.
        """
        x = frame
        atr = np.maximum(np.abs(x.get("atr_pct", 0.001).to_numpy(dtype=float)), 0.0002)
        r3 = x.get("return_3", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        r10 = x.get("return_10", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        ema = x.get("ema_spread", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        nifty = x.get("nifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        bank = x.get("banknifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        vix = x.get("vix_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        regime = (
            0.42*np.tanh(r3/(2.2*atr))
            + 0.24*np.tanh(r10/(4.5*atr))
            + 0.16*np.tanh(ema/(2.5*atr))
            + 0.10*np.tanh(nifty/(3.0*atr))
            + 0.08*np.tanh(bank/(3.0*atr))
            - 0.06*np.tanh(vix/(3.0*atr))
        )
        regime_prob = 0.5 + 0.5*np.clip(regime, -1.0, 1.0)
        # Model remains primary; regime is a stabilizing live adjustment, not a hard override.
        return np.clip(0.80*np.asarray(base_p, dtype=float) + 0.20*regime_prob, 0.01, 0.99)

    @staticmethod
    def _trend_return(frame: pd.DataFrame) -> np.ndarray:
        x = frame
        atr = np.maximum(np.abs(x.get("atr_pct", 0.001).to_numpy(dtype=float)), 0.0002)
        r3 = x.get("return_3", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        r10 = x.get("return_10", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        ema = x.get("ema_spread", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        context = 0.5*x.get("nifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float) + 0.5*x.get("banknifty_return", pd.Series(0.0, index=x.index)).to_numpy(dtype=float)
        score = 0.60*np.tanh(r3/(2.0*atr)) + 0.25*np.tanh(r10/(4.0*atr)) + 0.15*np.tanh((ema+context)/(3.0*atr))
        return np.clip(score*0.60*atr, -0.006, 0.006)

    @staticmethod
    def _threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
        best = (0.70, 0.0, 0.0)
        n = len(y)
        minimum = max(12, int(n*0.02))
        for t in np.arange(0.55, 0.96, 0.01):
            long_mask = p >= t
            short_mask = p <= 1-t
            mask = long_mask | short_mask
            signals = int(mask.sum())
            if signals < minimum:
                continue
            preds = np.where(long_mask[mask], 1, 0)
            truth = y[mask]
            precision = precision_score(truth, preds, zero_division=0)
            coverage = signals/n if n else 0.0
            score = precision + 0.07*math.sqrt(coverage)
            best_score = best[1] + 0.07*math.sqrt(best[2]) if best[2] else -1
            if score > best_score:
                best=(float(t), float(precision), float(coverage))
        return best

    @staticmethod
    def _fit_classifier(X, y):
        if len(np.unique(y)) < 2:
            raise ValueError("The training window contains only one market direction class; widen the training window before using the predictor.")
        base = HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=220, max_leaf_nodes=31,
            l2_regularization=0.15, min_samples_leaf=12, random_state=42,
        )
        cv = TimeSeriesSplit(n_splits=3)
        return CalibratedClassifierCV(base, cv=cv, method="sigmoid")

    @staticmethod
    def _fit_regressor(X, y):
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.045, max_iter=220, max_leaf_nodes=31,
            l2_regularization=0.1, loss="absolute_error", random_state=42,
        )
        et = ExtraTreesRegressor(n_estimators=140, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=42)
        class Ensemble:
            def __init__(self, a, b): self.a, self.b = a, b
            def fit(self, Xf, yf): self.a.fit(Xf,yf); self.b.fit(Xf,yf); return self
            def predict(self, Xf): return 0.65*self.a.predict(Xf)+0.35*self.b.predict(Xf)
        return Ensemble(hgb, et)

    def fit(self, train_df: pd.DataFrame, features: list[str]) -> ModelReport:
        if len(train_df) < self.min_train_rows:
            raise ValueError(f"Need at least {self.min_train_rows} clean intraday rows; got {len(train_df)}.")
        self.features=list(features)
        data=train_df.sort_values("timestamp").reset_index(drop=True)
        sessions=data["timestamp"].dt.date.drop_duplicates().tolist()
        split_idx=max(1,int(len(sessions)*0.78))
        split_day=sessions[min(split_idx,len(sessions)-1)]
        train=data.loc[data["timestamp"].dt.date < split_day].copy()
        val=data.loc[data["timestamp"].dt.date >= split_day].copy()
        if len(val)<max(50,int(len(data)*0.10)) or train.empty:
            cut=int(len(data)*0.80); train=data.iloc[:cut].copy(); val=data.iloc[cut:].copy()
        if len(np.unique(train["direction"].astype(int)))<2 or len(np.unique(val["direction"].astype(int)))<2:
            cut=int(len(data)*0.75); train=data.iloc[:cut].copy(); val=data.iloc[cut:].copy()
        if len(np.unique(train["direction"].astype(int)))<2:
            raise ValueError("The training window contains only one market direction class; widen the training window before using the predictor.")

        X_train=train[self.features]; X_val=val[self.features]
        y_train=train["direction"].astype(int); y_val=val["direction"].astype(int)
        direction=self._fit_classifier(X_train,y_train); direction.fit(X_train,y_train)
        raw_p=direction.predict_proba(X_val)[:,1]
        p_val=self._adaptive_probability(raw_p,val)
        threshold, precision, coverage=self._threshold(y_val.to_numpy(),p_val)
        acc=accuracy_score(y_val,(p_val>=0.5).astype(int))
        brier=brier_score_loss(y_val,p_val)

        def fit_reg(target):
            reg=self._fit_regressor(X_train,train[target]); reg.fit(X_train,train[target])
            mae=mean_absolute_error(val[target],reg.predict(X_val)); return reg,float(mae)
        ret_model,ret_mae=fit_reg("next_return")
        hi_model,_=fit_reg("next_high_return")
        lo_model,_=fit_reg("next_low_return")
        eod_model,_=fit_reg("eod_return")

        # Final live fit uses all completed-session rows; current forming candle is never used as a target.
        self.direction_model=self._fit_classifier(data[self.features],data["direction"].astype(int)).fit(data[self.features],data["direction"].astype(int))
        self.return_model=self._fit_regressor(data[self.features],data["next_return"]).fit(data[self.features],data["next_return"])
        self.high_model=self._fit_regressor(data[self.features],data["next_high_return"]).fit(data[self.features],data["next_high_return"])
        self.low_model=self._fit_regressor(data[self.features],data["next_low_return"]).fit(data[self.features],data["next_low_return"])
        self.eod_model=self._fit_regressor(data[self.features],data["eod_return"]).fit(data[self.features],data["eod_return"])
        self.report=ModelReport(
            direction_accuracy=float(acc), calibrated_brier=float(brier), signal_precision=float(precision), signal_coverage=float(coverage),
            next_return_mae=float(ret_mae), threshold=float(threshold), gate_90_passed=bool(precision>=0.90 and coverage>=0.02),
            train_rows=len(train), validation_rows=len(val), fit_timestamp=pd.Timestamp.utcnow().isoformat(),
        )
        return self.report

    def predict(self, latest_row: pd.DataFrame, current_price: float, news_impact: float = 0.0) -> dict:
        if self.report is None or self.direction_model is None:
            raise RuntimeError("Model is not trained.")
        X=latest_row[self.features]
        raw_p=float(self.direction_model.predict_proba(X)[0,1])
        p_up=float(self._adaptive_probability(np.array([raw_p]), latest_row)[0])
        raw_return=float(self.return_model.predict(X)[0])
        high_return=float(self.high_model.predict(X)[0])
        low_return=float(self.low_model.predict(X)[0])
        eod_return=float(self.eod_model.predict(X)[0])

        trend_overlay=float(self._trend_return(latest_row)[0])
        news_overlay=float(np.clip(news_impact,-1,1)*max(abs(raw_return),0.0008)*0.18)
        adjusted_return=0.82*raw_return+0.18*trend_overlay+news_overlay
        predicted_close=current_price*(1+adjusted_return)
        predicted_high=current_price*(1+max(0.0, high_return, adjusted_return))
        predicted_low=current_price*(1+min(0.0, low_return, adjusted_return))
        predicted_eod=current_price*(1+eod_return+news_overlay*0.25)

        t=self.report.threshold
        select_buy=p_up>=t; select_sell=p_up<=1-t
        gate=self.report.gate_90_passed
        if gate and select_buy and news_impact>-0.55:
            signal="BUY"
        elif gate and select_sell and news_impact<0.55:
            signal="SELL"
        else:
            signal="REVALIDATE"
        return {
            "signal":signal,
            "probability_up":round(p_up,4),
            "probability_down":round(1-p_up,4),
            "raw_return":round(raw_return,6),
            "trend_adjustment":round(trend_overlay,6),
            "news_adjusted_return":round(adjusted_return,6),
            "predicted_close":round(predicted_close,4),
            "predicted_high":round(max(predicted_high,predicted_close),4),
            "predicted_low":round(min(predicted_low,predicted_close),4),
            "eod_return":round(eod_return,6),
            "predicted_eod_close":round(predicted_eod,4),
            "validation": {
                "direction_accuracy":round(self.report.direction_accuracy*100,2),
                "calibrated_brier":round(self.report.calibrated_brier,5),
                "signal_precision":round(self.report.signal_precision*100,2),
                "signal_coverage":round(self.report.signal_coverage*100,2),
                "next_return_mae":round(self.report.next_return_mae,6),
                "threshold":round(self.report.threshold,2),
                "gate_90_passed":self.report.gate_90_passed,
                "train_rows":int(self.report.train_rows),
                "validation_rows":int(self.report.validation_rows),
                "fit_timestamp":self.report.fit_timestamp,
            },
        }
