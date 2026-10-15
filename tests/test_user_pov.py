from datetime import datetime, timezone
from types import SimpleNamespace

from app.data.news import NewsService
from app.data.toi import TOISnapshot


def test_display_news_prefers_market_sections(tmp_path):
    svc=NewsService("", state_dir=str(tmp_path))
    svc._toi.snapshot=TOISnapshot(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        headlines=[
            {"title":"Kenya election update", "section":"World", "url":"u1", "scraped_at":datetime.now(timezone.utc).isoformat()},
            {"title":"Sensex, Nifty gain as banks rally", "section":"Markets", "url":"u2", "scraped_at":datetime.now(timezone.utc).isoformat()},
            {"title":"India economy outlook improves", "section":"Economy", "url":"u3", "scraped_at":datetime.now(timezone.utc).isoformat()},
        ],
        sections_ok=18, sections_total=18, error=""
    )
    out=svc.display_snapshot("SENSEX","SENSEX")
    assert out["articles"][0]["section"] in {"Markets","Economy"}


def test_news_endpoint_is_user_facing_only(tmp_path):
    svc=NewsService("", state_dir=str(tmp_path))
    svc._toi.snapshot=TOISnapshot(
        fetched_at=datetime.now(timezone.utc).isoformat(), headlines=[], sections_ok=18, sections_total=18, error=""
    )
    out=svc.display_snapshot("SENSEX","SENSEX")
    assert out["download_url"]=="/api/news/download"

def test_display_news_excludes_technology_from_market_panel(tmp_path):
    svc=NewsService("", state_dir=str(tmp_path))
    now=datetime.now(timezone.utc).isoformat()
    svc._toi.snapshot=TOISnapshot(
        fetched_at=now,
        headlines=[
            {"title":"New phone launch", "section":"Technology", "url":"tech", "scraped_at":now},
            {"title":"Markets gain as banks rally", "section":"Markets", "url":"mkt", "scraped_at":now},
            {"title":"India economy grows", "section":"Economy", "url":"eco", "scraped_at":now},
        ],
        sections_ok=18, sections_total=18, error=""
    )
    out=svc.display_snapshot("SENSEX","SENSEX")
    assert [x["url"] for x in out["articles"]][:2] == ["mkt","eco"]
