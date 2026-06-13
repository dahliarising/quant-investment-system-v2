# tests/test_prediction_wiring.py
import json
from corvin_jarvis.prediction import run_prediction_digest as orch


def test_gather_inputs_reads_portfolio_and_universe(tmp_path):
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": [{"symbol": "META", "price": 100,
                                            "shares": 7}]}))
    uni = tmp_path / "monitored_universe.json"
    uni.write_text(json.dumps({"tickers": [{"symbol": "005930", "market": "KR"}]}))
    inp = orch.gather_inputs(portfolio_path=pf, universe_path=uni,
                             db_path=tmp_path / "daily.db", geo_fetch=lambda: None)
    assert any(h["symbol"] == "META" for h in inp["holdings"])
    assert "005930" in inp["universe"]
