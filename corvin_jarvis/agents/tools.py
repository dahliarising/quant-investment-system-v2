"""기존 quant 함수들을 LangGraph 툴로 래핑."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from corvin_jarvis import qa, predict, regime as regime_mod  # noqa: E402
from corvin_jarvis.debate import gather_bull_context, gather_bear_context  # noqa: E402

DB_PATH = BASE_DIR / "state" / "timeseries.db"


def get_bull_context(symbol: str, days: int = 7) -> dict[str, Any]:
    return gather_bull_context(DB_PATH, symbol=symbol, days=days)


def get_bear_context(symbol: str, days: int = 7) -> dict[str, Any]:
    return gather_bear_context(DB_PATH, symbol=symbol, days=days)


def get_history(symbol: str, days: int = 14) -> dict[str, Any]:
    return qa.recent_history_summary(DB_PATH, symbol=symbol, days=days)


def get_relative_strength(symbol: str, benchmark: str = "sp500", days: int = 7) -> dict[str, Any]:
    return qa.relative_strength(DB_PATH, symbol=symbol, benchmark=benchmark, days=days)


def get_downside_probability(symbol: str, threshold_pct: float = -0.05) -> float:
    """현재가 대비 threshold_pct 이하로 떨어질 로그-정규 확률 (30일 horizon)."""
    hist = get_history(symbol, days=30)
    if not hist["end_price"] or not hist["pct_change"]:
        return 0.5
    current = hist["end_price"]
    target = current * (1 + threshold_pct)
    returns = [hist["pct_change"] / 100] if hist["pct_change"] else [0.0]
    import math
    mu = sum(returns) / len(returns)
    sigma = max(abs(r - mu) for r in returns) or 0.02
    return predict.probability_below(current, target, mu=mu, sigma=sigma, horizon=30)


def get_regime(latest: dict[str, Any]) -> dict[str, Any]:
    """VIX + FX 기반 레짐 라벨."""
    from corvin_jarvis.regime import Snapshot as RegimeSnapshot
    try:
        snap = RegimeSnapshot(
            vix=latest.get("vix", {}).get("price", 20.0),
            usd_krw_pct=latest.get("fx", {}).get("usd_krw_1d_pct", 0.0),
            correlation=latest.get("correlation", 0.0),
        )
        return regime_mod.detect_regime_from_snapshot(snap)
    except Exception:
        return {"label": "neutral", "score": 0.5}
