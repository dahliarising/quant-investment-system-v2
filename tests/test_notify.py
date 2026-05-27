import json

from corvin_jarvis import notify


def test_actionability_flags_held_symbol():
    held = {"NVDA", "META", "MSFT"}
    a = {"metric": "pnl_NVDA", "message": "NVDA 수익 +16%", "value": 16.0, "severity": "medium"}
    b = {"metric": "universe_005930_confirmed", "message": "삼성전자 급등", "value": 7.0, "severity": "medium"}
    assert notify._actionability(a, held) == 1
    assert notify._actionability(b, held) == 0


def test_rank_orders_by_severity_then_action_then_magnitude():
    held = {"NVDA"}
    alerts = [
        {"metric": "x", "message": "m", "value": 3.0, "severity": "medium"},
        {"metric": "rs_NVDA_confirmed", "message": "NVDA RS", "value": 5.0, "severity": "medium"},
        {"metric": "y", "message": "y", "value": 1.0, "severity": "critical"},
    ]
    ranked = notify._rank_alerts(alerts, held)
    assert ranked[0]["severity"] == "critical"          # severity 최우선
    assert ranked[1]["metric"] == "rs_NVDA_confirmed"    # 같은 medium 중 보유종목 우선


def test_held_symbols_reads_portfolio(tmp_path, monkeypatch):
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": [{"symbol": "NVDA"}, {"symbol": "META"}]}))
    monkeypatch.setattr(notify, "PORTFOLIO_FILE", pf)
    assert notify._held_symbols() == {"NVDA", "META"}
