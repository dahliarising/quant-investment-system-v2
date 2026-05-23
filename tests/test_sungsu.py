"""Tests for corvin_jarvis.sungsu — Operation Sungsu 2027 bridge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corvin_jarvis import sungsu


@pytest.mark.unit
def test_load_scenarios_returns_empty_when_missing(tmp_path: Path) -> None:
    out = sungsu.load_scenarios(tmp_path / "nope.json")
    assert out == []


@pytest.mark.unit
def test_load_scenarios_parses_valid(tmp_path: Path) -> None:
    f = tmp_path / "scenarios.json"
    f.write_text(json.dumps({
        "scenarios": [
            {"id": "s1", "name": "통근 + 화성/현 학교 유지",
             "score": 88, "annual_cost_krw": 0,
             "asset_implications": {"keep_liquid": 30000000}},
            {"id": "s2", "name": "이사 + 강남 통근",
             "score": 65, "annual_cost_krw": 40000000,
             "asset_implications": {"sell_real_estate": True}},
        ]
    }))
    out = sungsu.load_scenarios(f)
    assert len(out) == 2
    assert out[0]["score"] == 88


@pytest.mark.unit
def test_best_scenario_returns_highest_score() -> None:
    data = [
        {"id": "a", "name": "A", "score": 70},
        {"id": "b", "name": "B", "score": 88},
        {"id": "c", "name": "C", "score": 75},
    ]
    out = sungsu.best_scenario(data)
    assert out["id"] == "b"


@pytest.mark.unit
def test_best_scenario_handles_empty() -> None:
    assert sungsu.best_scenario([]) is None


@pytest.mark.unit
def test_asset_allocation_recommendation_for_scenario() -> None:
    scenario = {
        "id": "s2",
        "name": "이사",
        "annual_cost_krw": 40_000_000,
        "asset_implications": {"sell_real_estate": True},
    }
    rec = sungsu.asset_allocation_recommendation(
        scenario, current_net_worth_krw=1_000_000_000,
    )
    assert "scenario_id" in rec
    assert rec["scenario_id"] == "s2"
    assert rec["annual_cost_krw"] == 40_000_000
    assert "recommendations" in rec
    assert any("real_estate" in r or "부동산" in r for r in rec["recommendations"])
