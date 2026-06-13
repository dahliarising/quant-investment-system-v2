"""Tests for brief CLI entrypoint — compose_from_sources (pure, injected loaders)."""
from __future__ import annotations

from corvin_jarvis.brief.__main__ import compose_from_sources


def test_compose_from_sources_injected(tmp_path, monkeypatch):
    # 데이터 로더를 주입해 외부 IO 없이 검증
    positions = [{"symbol": "MSFT", "pnl_pct": 12.9, "bucket": "trade",
                  "currency": "USD", "current_price": 390.3, "error": None}]
    text = compose_from_sources(
        load_positions=lambda: positions,
        load_actions=lambda: [{"symbol": "BWXT", "action": "매수후보",
                               "urgency": 55, "rationale": "원자력",
                               "sources": ["leading"]}],
        load_calibration=lambda: {},
        load_held=lambda: {"MSFT"},
        market_state="장마감", fresh_label="종가",
        as_of="2026-06-12 05:45 KST",
    )
    assert "MSFT" in text
    assert "━━━" in text
    assert len(text) <= 1900
