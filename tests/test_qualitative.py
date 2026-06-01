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


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("B", (), {"text": text})()]


class _FakeClient:
    def __init__(self, text):
        self._text = text
        self.messages = type(
            "M", (), {"create": lambda _self, **kw: _FakeMsg(self._text)}
        )()


@pytest.mark.unit
def test_qualitative_score_with_client():
    client = _FakeClient("BULL")
    score = ql.qualitative_score("NVDA", "AI GPU 절대강자", client=client)
    assert score == 80.0


@pytest.mark.unit
def test_qualitative_score_no_client_returns_none():
    assert ql.qualitative_score("NVDA", "x", client=None) is None
