"""Tests for corvin_jarvis.qualitative (mock client, no real API)."""
from __future__ import annotations

import pytest

from corvin_jarvis import qualitative as ql


@pytest.mark.unit
def test_verdict_to_score():
    assert ql.verdict_to_score("BULL") == 80.0
    assert ql.verdict_to_score("neutral") == 50.0
    assert ql.verdict_to_score("Bear") == 20.0
    assert ql.verdict_to_score("garbage") is None


@pytest.mark.unit
def test_extract_verdict_clean():
    assert ql.extract_verdict("BULL") == "bull"
    assert ql.extract_verdict("BEAR") == "bear"


@pytest.mark.unit
def test_extract_verdict_in_noise():
    # CLI 출력에 잡음이 섞여도 첫 verdict 추출
    assert ql.extract_verdict("제 판단은 NEUTRAL 입니다") == "neutral"
    assert ql.extract_verdict("답: bull\n(hook noise)") == "bull"


@pytest.mark.unit
def test_extract_verdict_none():
    assert ql.extract_verdict("모르겠음") is None
    assert ql.extract_verdict("") is None


@pytest.mark.unit
def test_qualitative_score_via_cli_with_runner():
    # runner 주입(실제 CLI 미호출) → verdict 매핑
    sigs = ql.qualitative_score_via_cli(
        "NVDA", "AI GPU 절대강자", runner=lambda prompt: "BULL"
    )
    assert sigs == 80.0


@pytest.mark.unit
def test_qualitative_score_via_cli_runner_none_output():
    assert ql.qualitative_score_via_cli(
        "NVDA", "x", runner=lambda prompt: None
    ) is None
    assert ql.qualitative_score_via_cli(
        "NVDA", "x", runner=lambda prompt: "횡설수설"
    ) is None
