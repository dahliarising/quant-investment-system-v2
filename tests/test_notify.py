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


def _write_state(tmp_path, monkeypatch, alerts):
    af = tmp_path / "alerts.json"
    af.write_text(json.dumps({"alerts": alerts}))
    cf = tmp_path / "config.json"
    cf.write_text(json.dumps({"notification": {
        "urgent_min_severity": "high", "digest_min_severity": "medium",
        "digest_max_items": 8, "max_per_push": 8,
        "discord_webhook_url": None, "imessage_recipient": None,
    }}))
    pf = tmp_path / "portfolio.json"
    pf.write_text(json.dumps({"holdings": []}))
    monkeypatch.setattr(notify, "ALERTS_FILE", af)
    monkeypatch.setattr(notify, "CONFIG_FILE", cf)
    monkeypatch.setattr(notify, "PORTFOLIO_FILE", pf)
    monkeypatch.setattr(notify, "DEDUP_FILE", tmp_path / "push_dedup.json")
    monkeypatch.setattr(notify, "PENDING_FILE", tmp_path / "pending.json")


def test_urgent_mode_drops_medium(tmp_path, monkeypatch):
    _write_state(tmp_path, monkeypatch, [
        {"category": "x", "metric": "m1", "severity": "medium", "message": "med", "value": 3.0},
        {"category": "x", "metric": "m2", "severity": "high", "message": "hi", "value": 9.0},
    ])
    res = notify.notify(mode="urgent")
    assert res.skipped_severity == 1     # medium은 high 게이트에서 제외


def test_digest_mode_includes_medium_no_dedup(tmp_path, monkeypatch):
    _write_state(tmp_path, monkeypatch, [
        {"category": "x", "metric": "m1", "severity": "medium", "message": "med", "value": 3.0},
        {"category": "x", "metric": "m2", "severity": "high", "message": "hi", "value": 9.0},
    ])
    res = notify.notify(mode="digest")
    assert res.skipped_severity == 0     # medium 포함
    assert res.skipped_dedup == 0        # digest는 dedup 안 함


def test_format_message_respects_title_and_limit():
    alerts = [{"severity": "high", "message": f"alert {i}", "value": float(i)} for i in range(10)]
    msg = notify._format_message(alerts, compact=True, title="📋 다이제스트", limit=3)
    assert msg.startswith("📋 다이제스트")
    assert "…외 7건" in msg
