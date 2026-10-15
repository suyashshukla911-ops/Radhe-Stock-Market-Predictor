import pandas as pd
from app.data.upstox import LiveMarketHub, UpstoxClient

def test_live_hub_empty_is_safe():
    hub=LiveMarketHub(UpstoxClient(""))
    out=hub.live_bars("NSE_INDEX|Nifty 50",5)
    assert isinstance(out,pd.DataFrame)
    assert out.empty
