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
