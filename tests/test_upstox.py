import pandas as pd
from app.data.upstox import UpstoxClient, LiveMarketHub

def test_local_aggregation_09_15():
    ts=pd.date_range("2026-10-15 09:15",periods=10,freq="1min",tz="Asia/Kolkata")
    df=pd.DataFrame({"timestamp":ts,"open":range(10),"high":range(1,11),"low":range(10),"close":range(1,11),"volume":[0]*10,"oi":[0]*10})
    out=UpstoxClient.aggregate_minutes(df,5)
    assert len(out)==2
    assert out.iloc[0].timestamp.hour==9 and out.iloc[0].timestamp.minute==15

def test_live_hub_empty_safe():
    hub=LiveMarketHub(UpstoxClient(""))
    assert hub.live_bars("NSE_INDEX|Nifty 50",5).empty
