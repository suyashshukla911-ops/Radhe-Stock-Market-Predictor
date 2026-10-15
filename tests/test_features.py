import pandas as pd
from app.ml.features import FEATURES, build_features, prepare_training_frame

def sample_df(rows=120):
    ts=pd.date_range("2026-10-01 09:15", periods=rows, freq="5min", tz="Asia/Kolkata")
    base=100+(pd.Series(range(rows))*0.1)
    return pd.DataFrame({"timestamp":ts,"open":base,"high":base+0.2,"low":base-0.2,"close":base+0.05,"volume":1000.0,"oi":0.0})

def test_features_have_expected_columns():
    out=build_features(sample_df())
    assert set(FEATURES).issubset(out.columns)

def test_training_frame_not_empty():
    out=prepare_training_frame(sample_df(240))
    assert not out.empty
