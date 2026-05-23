"""Corvin Jarvis — Operation Sungsu 2027 Bridge (Tier 3.4)

가족 전략 시나리오 ↔ 자산배분 연동. strategies/family-plan*.html의 시나리오 점수와
asset_implications를 읽어 자산 권고 생성.

⚠️ HTML 파싱은 별도 도구 권장. 이 모듈은 사용자가 추출한 scenarios.json 받는 인터페이스.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("corvin.sungsu")


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """scenarios.json → list[scenario dict]."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        items = data.get("scenarios", [])
        return list(items) if isinstance(items, list) else []
    except (json.JSONDecodeError, OSError) as e:
        log.warning("scenarios load failed: %s", e)
        return []


def best_scenario(scenarios: list[dict[str, Any]]) -> dict[str, Any] | None:
    """score 가장 높은 시나리오."""
    if not scenarios:
        return None
    return max(scenarios, key=lambda s: float(s.get("score", 0)))


def asset_allocation_recommendation(
    scenario: dict[str, Any],
    current_net_worth_krw: int,
) -> dict[str, Any]:
    """시나리오 → 자산배분 권고 사항 (advisory only, LLM 합성용 raw)."""
    sid = scenario.get("id")
    annual_cost = int(scenario.get("annual_cost_krw", 0) or 0)
    impl = scenario.get("asset_implications", {})

    recommendations: list[str] = []

    if annual_cost > 0:
        years_runway = current_net_worth_krw / annual_cost if annual_cost else float("inf")
        recommendations.append(
            f"연간 비용 {annual_cost:,} KRW (현 자산 기준 {years_runway:.1f}년 runway)"
        )

    if impl.get("sell_real_estate"):
        recommendations.append("부동산 매각 검토 — 시장 타이밍 + tax-loss 영향 분석 필요")
    if impl.get("keep_liquid"):
        liquid = int(impl["keep_liquid"])
        recommendations.append(f"유동성 확보 {liquid:,} KRW (현금성 자산 유지)")
    if impl.get("increase_us_stocks"):
        recommendations.append("미국 주식 비중 증가 — 환노출 hedge 검토")
    if impl.get("reduce_kr_stocks"):
        recommendations.append("한국 주식 비중 축소")

    return {
        "scenario_id": sid,
        "scenario_name": scenario.get("name"),
        "annual_cost_krw": annual_cost,
        "current_net_worth_krw": current_net_worth_krw,
        "recommendations": recommendations,
    }
