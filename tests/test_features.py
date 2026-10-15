import numpy as np
import pandas as pd
from app.ml.features import FEATURES, build_features, prepare_training_frame

def sample_df(rows=500, volume=0):
    days=[]
    base=[]
    close=[]
    cur=100.0
    for d in pd.date_range("2026-10-01","2026-10-08",freq="D"):
        if d.weekday()>=5: continue
        ts=pd.date_range(d.replace(hour=9,minute=15),periods=75,freq="5min",tz="Asia/Kolkata")
        for t in ts:
            days.append(t); cur += 0.05*np.sin(len(days)/7)
            base.append(cur); close.append(cur+0.02)
            if len(days)>=rows: break
        if len(days)>=rows: break
    df=pd.DataFrame({"timestamp":days,"open":base,"high":np.array(base)+.2,"low":np.array(base)-.2,"close":close,"volume":float(volume),"oi":0.0})
    return df

def test_features_have_expected_columns():
    out=build_features(sample_df())
    assert set(FEATURES).issubset(out.columns)

def test_training_frame_not_empty_with_zero_index_volume():
    out=prepare_training_frame(sample_df(500,volume=0))
    assert not out.empty
