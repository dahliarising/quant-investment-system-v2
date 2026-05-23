"""Tests for corvin_jarvis.counterfactual — Tier 4.3 calibration."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from corvin_jarvis import counterfactual, timeseries


@pytest.mark.unit
def test_extract_predictions_buy_bullish() -> None:
    text = "META 분할매수 권고. NVDA 보유 유지."
    preds = counterfactual.extract_predictions(text)
    by_sym = {p["symbol"]: p for p in preds}
    assert "META" in by_sym
    assert by_sym["META"]["direction"] == "bullish"
    assert by_sym["NVDA"]["direction"] == "neutral"


@pytest.mark.unit
def test_extract_predictions_sell_bearish() -> None:
    text = "MSFT 분할매도 검토. TSLA 손절 권고."
    preds = counterfactual.extract_predictions(text)
    by_sym = {p["symbol"]: p for p in preds}
    assert by_sym["MSFT"]["direction"] == "bearish"
    assert by_sym["TSLA"]["direction"] == "bearish"


@pytest.mark.unit
def test_extract_predictions_skip_unmentioned() -> None:
    text = "오늘 시장 안정. 특별 신호 없음."
    preds = counterfactual.extract_predictions(text)
    assert preds == []


@pytest.mark.unit
def test_score_prediction_correct_bullish() -> None:
    """매수 권고 + 실제 +2% → correct."""
    verdict = counterfactual.score_prediction(
        direction="bullish", actual_pct=2.0,
    )
    assert verdict == "correct"


@pytest.mark.unit
def test_score_prediction_incorrect_bullish() -> None:
    """매수 권고 + 실제 -3% → incorrect."""
    verdict = counterfactual.score_prediction(
        direction="bullish", actual_pct=-3.0,
    )
    assert verdict == "incorrect"


@pytest.mark.unit
def test_score_prediction_inconclusive_when_flat() -> None:
    """direction 있어도 |move| < 0.5% → inconclusive."""
    verdict = counterfactual.score_prediction(
        direction="bullish", actual_pct=0.2,
    )
    assert verdict == "inconclusive"


@pytest.mark.unit
def test_score_prediction_neutral_always_inconclusive() -> None:
    assert counterfactual.score_prediction("neutral", actual_pct=5.0) == "inconclusive"
    assert counterfactual.score_prediction("neutral", actual_pct=-5.0) == "inconclusive"


@pytest.mark.unit
def test_score_prediction_handles_none_move() -> None:
    assert counterfactual.score_prediction("bullish", actual_pct=None) == "inconclusive"


@pytest.mark.unit
def test_monthly_review_aggregates(tmp_db_path: Path, tmp_path: Path) -> None:
    wiki = tmp_path / "corvin-sessions"
    wiki.mkdir()
    (wiki / "2026-05-01-w1.md").write_text("META 분할매수 권고.\n")
    (wiki / "2026-05-08-w2.md").write_text("MSFT 분할매도 검토.\n")

    # seed prices: META +5%, MSFT -2% (both correct directionally)
    timeseries.init_db(tmp_db_path)
    for sym, prices in [("META", [600.0, 630.0]), ("MSFT", [400.0, 392.0])]:
        for i, p in enumerate(prices):
            snap = {
                "timestamp_utc": f"2026-05-{1+i*14:02d}T00:00:00+00:00",
                "timestamp_kst": f"2026-05-{1+i*14:02d}T09:00:00+09:00",
                "indices": {}, "commodities": {}, "fx": {}, "watchlist": [],
                "portfolio": [{
                    "symbol": sym, "shares": 1, "avg_price": p, "currency": "USD",
                    "current_price": p, "pnl_pct": 0.0, "market_value": p, "error": None,
                }],
            }
            timeseries.write_snapshot(tmp_db_path, snap)

    report = counterfactual.monthly_review(
        wiki, tmp_db_path, year=2026, month=5,
    )
    assert report["year"] == 2026
    assert report["month"] == 5
    assert report["total_predictions"] >= 2
    # 둘 다 correct → accuracy > 0
    assert report["accuracy_pct"] > 0


@pytest.mark.unit
def test_monthly_review_empty_when_no_sessions(tmp_db_path: Path, tmp_path: Path) -> None:
    wiki = tmp_path / "empty"
    timeseries.init_db(tmp_db_path)
    report = counterfactual.monthly_review(wiki, tmp_db_path, year=2026, month=5)
    assert report["total_predictions"] == 0
    assert report["accuracy_pct"] is None
