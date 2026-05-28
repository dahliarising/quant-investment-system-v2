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


def test_notify_never_sends_real_imessage_during_tests(tmp_path, monkeypatch):
    # 회귀 방지(2026-05-28 사고): notify()가 테스트 중 실제 osascript를 호출하면 안 됨.
    import subprocess

    def _boom(*_a, **_k):
        raise AssertionError("테스트가 실제 iMessage(osascript)를 전송함 — 격리 깨짐")

    monkeypatch.setattr(subprocess, "run", _boom)
    _write_state(tmp_path, monkeypatch, [
        {"category": "x", "metric": "m2", "severity": "high", "message": "hi", "value": 9.0},
    ])
    res = notify.notify(mode="digest")
    assert "imessage" not in res.channels_delivered


def test_format_message_respects_title_and_limit():
    alerts = [{"severity": "high", "message": f"alert {i}", "value": float(i)} for i in range(10)]
    msg = notify._format_message(alerts, compact=True, title="📋 다이제스트", limit=3)
    assert msg.startswith("📋 다이제스트")
    assert "…외 7건" in msg


def test_interpret_universe():
    a = {"category": "universe", "metric": "universe_000660_confirmed", "value": 9.31,
         "message": "✅ SK하이닉스(000660) 급등 +9.31%", "severity": "high"}
    txt = notify._interpret(a)
    assert "SK하이닉스" in txt and "급등" in txt   # 코드(000660) 대신 종목명


def test_interpret_narrative_foreign():
    a = {"category": "narrative", "metric": "foreign_net_buy", "value": 2.38,
         "message": "KR foreign_net_buy spike Z=+2.38", "severity": "high"}
    txt = notify._interpret(a)
    assert "외국인" in txt


def test_interpret_leading_rs_positive_is_leader():
    a = {"category": "leading_rs", "metric": "rs_000660_confirmed", "value": 7.06,
         "message": "✅ SK하이닉스 상대강도 강세", "severity": "medium"}
    txt = notify._interpret(a)
    assert "주도주" in txt


def test_interpret_portfolio_mentions_symbol():
    a = {"category": "portfolio", "metric": "pnl_NVDA", "value": 16.49,
         "message": "NVDA 수익 +16.49%", "severity": "medium"}
    txt = notify._interpret(a)
    assert "NVDA" in txt


def test_format_message_appends_interpretation():
    alerts = [{"category": "universe", "metric": "universe_000660_confirmed", "value": 9.31,
               "message": "✅ SK하이닉스(000660) 급등 +9.31%", "severity": "high"}]
    msg = notify._format_message(alerts, compact=False, title="t", limit=8)
    assert "📖 해석" in msg
    assert "000660" in msg


def test_format_message_appends_verdicts():
    alerts = [{"category": "universe", "metric": "universe_000660_confirmed", "value": 9.0,
               "message": "✅ SK하이닉스 급등", "severity": "high"}]
    verdicts = {
        "NVDA": {"action": "홀딩", "confidence": "상", "rationale": "보유 논리 유효"},
        "000660": {"action": "관망", "confidence": "중", "rationale": "추격 위험"},
    }
    msg = notify._format_message(alerts, compact=False, title="t", limit=8, verdicts=verdicts)
    assert "🎯 행동 판정" in msg
    assert "NVDA" in msg and "홀딩" in msg
    assert "관망" in msg


def test_format_message_no_verdict_section_when_empty():
    alerts = [{"category": "universe", "metric": "u", "value": 9.0, "message": "x", "severity": "high"}]
    msg = notify._format_message(alerts, compact=False, title="t", limit=8, verdicts={})
    assert "🎯 행동 판정" not in msg


def test_format_message_compact_verdicts_section():
    alerts = [{"category": "universe", "metric": "u", "value": 9.0, "message": "x", "severity": "high"}]
    verdicts = {"NVDA": {"action": "홀딩", "confidence": "상", "rationale": "r", "name": "NVIDIA"}}
    msg = notify._format_message(alerts, compact=True, title="t", limit=8, verdicts=verdicts)
    assert "🎯 판정" in msg
    assert "NVIDIA 홀딩" in msg


def test_format_message_verdict_missing_keys_no_crash():
    alerts = [{"category": "universe", "metric": "u", "value": 9.0, "message": "x", "severity": "high"}]
    verdicts = {"XYZ": {}}   # 손상된 항목 — 크래시 없이 처리돼야
    msg = notify._format_message(alerts, compact=False, title="t", limit=8, verdicts=verdicts)
    assert "XYZ" in msg   # 관망/하 아님(키 없음→필터 통과) → 표시되되 크래시 없음


def test_format_message_hides_low_conf_watch_verdicts():
    alerts = [{"category": "universe", "metric": "u", "value": 9.0, "message": "x", "severity": "high"}]
    verdicts = {
        "NVDA": {"action": "홀딩", "confidence": "상", "rationale": "보유 논리 유효"},
        "005490": {"action": "관망", "confidence": "하", "rationale": "뚜렷한 신호 없음"},
    }
    msg = notify._format_message(alerts, compact=False, title="t", limit=8, verdicts=verdicts)
    assert "NVDA" in msg          # 홀딩은 표시
    assert "005490" not in msg    # 관망+신뢰도 하 = 노이즈라 숨김
