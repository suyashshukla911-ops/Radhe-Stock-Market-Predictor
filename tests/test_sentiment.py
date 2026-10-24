from app.data.news import lexical_sentiment

def test_rbi_rate_cut_sensex_surges():
    score = lexical_sentiment("RBI rate cut, Sensex surges")
    assert score > 0.5, f"Expected positive score for rate cut surge, got {score}"

def test_debt_reduction_plan_boosts_shares():
    score = lexical_sentiment("Debt reduction plan boosts shares")
    assert score > 0.5, f"Expected positive score for debt reduction boost, got {score}"

def test_sensex_falls_despite_strong_earnings():
    score = lexical_sentiment("Sensex falls despite strong earnings")
    assert score < -0.2, f"Expected negative score for market fall despite earnings, got {score}"

def test_bagged_mega_order():
    score = lexical_sentiment("Company bags mega order for infrastructure project")
    assert score > 0.3, f"Expected positive score for bagged order, got {score}"

def test_slapped_with_fine():
    score = lexical_sentiment("Regulator slapped heavy fine on bank for compliance violation")
    assert score < -0.3, f"Expected negative score for regulatory fine, got {score}"

def test_job_cuts_negative():
    score = lexical_sentiment("Company announces 5000 job cuts amid slowdown")
    assert score < -0.3, f"Expected negative score for job cuts, got {score}"
