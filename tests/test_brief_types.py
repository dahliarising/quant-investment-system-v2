import dataclasses
from corvin_jarvis.brief.types import (
    PositionLine, EvidenceStack, Framing, PsychGuard, Brief,
)


def test_position_line_is_frozen():
    pl = PositionLine(symbol="META", pnl_pct=-4.9, action_icon="👀",
                      label="관망존", fresh_label="종가")
    assert pl.symbol == "META"
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        pl.symbol = "X"  # type: ignore[misc]


def test_brief_composes_all_parts():
    brief = Brief(
        headline="테스트", positions=[], evidence={}, framing=None,
        psych=None, as_of="2026-06-12", market_state="장마감",
    )
    assert brief.market_state == "장마감"
    assert brief.positions == []
