"""Tests for corvin_jarvis.leading_orchestrator."""
from __future__ import annotations

import pytest

from corvin_jarvis import leading_orchestrator as orch
from corvin_jarvis.signals.leading_signal import LeadingSignal


def _sig(sym, conf, advisory=False, msg="x", direction="bull"):
    return LeadingSignal(
        pillar="event", symbol=sym, direction=direction,
        confidence=conf, score=None, horizon="days",
        advisory=advisory, message=msg, evidence={},
    )


@pytest.mark.unit
def test_format_brief_groups_passed_signals():
    sigs = [_sig("012450", 80.0, msg="📅 012450 실적 D-3"),
            _sig("META", 40.0, msg="무음대상")]  # 40 → 게이트 탈락
    brief = orch.format_brief(sigs, threshold=60.0)
    assert "012450" in brief
    assert "무음대상" not in brief


@pytest.mark.unit
def test_format_brief_advisory_tagged():
    sigs = [_sig("012450", 90.0, advisory=True, msg="LMT 야간 강세")]
    brief = orch.format_brief(sigs, threshold=60.0)
    assert "참고용" in brief
    assert "LMT 야간 강세" in brief


@pytest.mark.unit
def test_format_brief_all_muted_returns_quiet_marker():
    sigs = [_sig("META", 30.0), _sig("MSFT", 50.0)]
    brief = orch.format_brief(sigs, threshold=60.0)
    assert brief == ""  # 전부 무음 → 빈 문자열(전송 안 함 신호)


@pytest.mark.unit
def test_collect_flattens_providers():
    p1 = lambda: [_sig("012450", 80.0)]
    p2 = lambda: [_sig("META", 70.0), _sig("MSFT", 65.0)]
    sigs = orch.collect([p1, p2])
    assert len(sigs) == 3


@pytest.mark.unit
def test_collect_skips_failing_provider():
    def boom():
        raise RuntimeError("provider down")
    p_ok = lambda: [_sig("012450", 80.0)]
    sigs = orch.collect([boom, p_ok])
    assert len(sigs) == 1     # 실패 provider 스킵, 나머지 진행


@pytest.mark.unit
def test_dispatch_brief_sends_nonempty():
    sent = {}
    def fake_sender(body: str) -> bool:
        sent["body"] = body
        return True
    ok = orch.dispatch_brief("내용 있음", sender=fake_sender)
    assert ok is True
    assert sent["body"] == "내용 있음"


@pytest.mark.unit
def test_dispatch_brief_skips_empty():
    called = {"n": 0}
    def fake_sender(body: str) -> bool:
        called["n"] += 1
        return True
    ok = orch.dispatch_brief("", sender=fake_sender)
    assert ok is False
    assert called["n"] == 0     # 빈 브리프 → sender 미호출


@pytest.mark.unit
def test_end_to_end_four_pillars():
    from datetime import date

    from corvin_jarvis.signals import (
        cross_market,
        ensemble,
        event_calendar,
        fundamental,
    )

    providers = [
        lambda: event_calendar.build_event_signals(
            as_of=date(2026, 6, 1),
            earnings_rows=[{"symbol": "012450", "earnings_date": date(2026, 6, 4)}],
            macro_horizon_days=30,
        ),
        lambda: [fundamental.build_fundamental_signal(
            "012450", financial=85.0, qualitative=80.0, news=70.0)],
        lambda: [ensemble.build_ensemble_signal(
            "012450", minervini=90.0, rs=80.0, volume=70.0)],
        lambda: [cross_market.build_cross_market_signal(
            "012450", proxy_name="LMT 야간", proxy_pct=9.0)],  # 페널티 후 ≥60 게이트 통과
    ]
    signals = orch.collect(providers)
    brief = orch.format_brief(signals, threshold=60.0)

    sent = {}
    ok = orch.dispatch_brief(brief, sender=lambda b: sent.update(body=b) or True)
    assert ok is True
    assert "펀더멘털" in sent["body"]
    assert "앙상블" in sent["body"]
    assert "참고용" in sent["body"]
