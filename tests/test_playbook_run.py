"""Tests for corvin_jarvis.playbook.run_playbook."""
from __future__ import annotations

from pathlib import Path

import pytest

from corvin_jarvis.playbook import run_playbook
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb():
    tech = Technicals(symbol="BWXT", market="US", price=188.0, ma20=204.0,
                      ma50=212.0, rsi=26.0, hi_52w=238.0)
    z = Zone("buy", "Z3 딥밸류", 30, 198.0, 202.0, "딥")
    return Playbook(symbol="BWXT", name="BWX", stance="ENTER", tech=tech,
                    zones=(z, z, z), status="BUY_NOW", badge="🟢",
                    pnl_pct=None, active_zone=z)


@pytest.mark.unit
def test_run_sends_push_and_writes_html(tmp_path: Path) -> None:
    sent: list[str] = []
    html_path = tmp_path / "playbook.html"
    result = run_playbook.run(
        playbooks=[_pb()],
        date_label="6/2",
        sender=lambda body: sent.append(body) or True,
        html_path=html_path,
    )
    assert result is True
    assert sent and "BWXT" in sent[0]
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_path.read_text()


@pytest.mark.unit
def test_run_returns_false_when_no_playbooks(tmp_path: Path) -> None:
    result = run_playbook.run(
        playbooks=[], date_label="6/2",
        sender=lambda body: True, html_path=tmp_path / "p.html",
    )
    assert result is False
