"""Tests for corvin_jarvis.playbook.render_text."""
from __future__ import annotations

import pytest

from corvin_jarvis.playbook import render_text
from corvin_jarvis.playbook.models import Playbook, Technicals, Zone


def _pb(symbol, status, badge, stance, pnl=None) -> Playbook:
    tech = Technicals(symbol=symbol, market="US", price=100.0, ma20=105.0,
                      ma50=110.0, rsi=50.0, hi_52w=130.0)
    z = Zone("buy", "Z2 핵심지지", 40, 99.0, 101.0, "MA50")
    return Playbook(symbol=symbol, name=symbol, stance=stance, tech=tech,
                    zones=(z, z, z), status=status, badge=badge,
                    pnl_pct=pnl, active_zone=z)


@pytest.mark.unit
def test_push_lists_triggers_first() -> None:
    pbs = [
        _pb("AAA", "WAIT", "⏳", "ENTER"),
        _pb("BWXT", "BUY_NOW", "🟢", "ENTER"),
    ]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "오늘 액션" in out
    assert "BWXT" in out
    # trigger appears before the wait-only summary section
    assert out.index("BWXT") < out.index("관찰")


@pytest.mark.unit
def test_push_counts_watch_and_triggers() -> None:
    pbs = [_pb("AAA", "WAIT", "⏳", "ENTER") for _ in range(5)]
    pbs.append(_pb("BBB", "BUY_NOW", "🟢", "ENTER"))
    out = render_text.render_push(pbs, date_label="6/2")
    assert "트리거1" in out or "트리거 1" in out


@pytest.mark.unit
def test_push_shows_holdings_line() -> None:
    pbs = [_pb("MSFT", "TRIM_NOW", "✂️", "HARVEST", pnl=20.2)]
    out = render_text.render_push(pbs, date_label="6/2")
    assert "보유" in out
    assert "MSFT" in out


@pytest.mark.unit
def test_push_holdings_include_price_and_pnl() -> None:
    pbs = [_pb("MSFT", "WAIT", "✅", "ACCUMULATE", pnl=20.2)]
    out = render_text.render_push(pbs, date_label="6/2")
    # 보유 종목은 현재가 + 손익률을 노출해야 한다 (단순 칩 X)
    assert "$100" in out
    assert "+20.2%" in out


@pytest.mark.unit
def test_kr_price_no_scientific_notation() -> None:
    # KR 고가 종목은 콤마 정수, 지수표기(1.07e+06) 금지
    assert render_text.fmt_price("KR", 1070000.0) == "₩1,070,000"
    assert render_text.fmt_price("US", 441.31) == "$441.31"


@pytest.mark.unit
def test_push_watch_triggers_show_price() -> None:
    watch = [_pb(f"W{i}", "WAIT", "⏳", "ENTER") for i in range(3)]
    watch.append(_pb("BWXT", "BUY_NOW", "🟢", "ENTER"))
    out = render_text.render_push(watch, date_label="6/2")
    # 관찰 트리거는 액션 섹션에서 현재가가 보여야 한다
    assert "$100" in out
