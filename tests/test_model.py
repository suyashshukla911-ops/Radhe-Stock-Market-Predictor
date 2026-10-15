import numpy as np
import pandas as pd
from app.ml.features import FEATURES, prepare_training_frame
from app.ml.model import CandleModel

def mock_candles():
    rows=[]
    rng=np.random.default_rng(4)
    cur=100.0
    for d in pd.date_range("2026-08-03","2026-10-02",freq="D"):
        if d.weekday()>=5: continue
        ts=pd.date_range(d.replace(hour=9,minute=15),periods=75,freq="5min",tz="Asia/Kolkata")
        for t in ts:
            ret=float(rng.normal(0,.001))
            op=cur; cur=max(.1,cur*(1+ret)); rows.append([t,op,max(op,cur)+.1,min(op,cur)-.1,cur,0.0,0.0])
    return pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume","oi"])

def test_model_can_fit_and_predict_zero_volume():
    train=prepare_training_frame(mock_candles())
    assert len(train)>=300
    model=CandleModel(min_train_rows=300)
    report=model.fit(train,FEATURES)
    result=model.predict(train.iloc[[-1]],float(train.close.iloc[-1]),news_impact=.1)
    assert 0<=result["probability_up"]<=1
    assert result["signal"] in {"BUY","SELL","REVALIDATE"}
    assert report.precision_lower_bound>=0
