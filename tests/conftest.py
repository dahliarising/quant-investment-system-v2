"""Pytest fixtures for Corvin tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture(autouse=True)
def _no_real_imessage(tmp_path_factory, monkeypatch):
    """안전망: 테스트가 실수로 실제 iMessage(osascript)를 폐하 폰으로 보내지 않게 한다.

    notify/dca/watchdog 모두 channels.send_imessage를 거치고, channels는 자체 CONFIG_FILE로
    실제 config.json을 읽는다. 테스트가 이를 patch하지 않으면 실제 전송이 나간다(2026-05-28 사고).
    모든 테스트에서 channels.CONFIG_FILE을 'log_only' 임시 config로 고정 → send_imessage 항상 False.
    실제 전송 경로를 검증하는 테스트(test_channels)는 본문에서 CONFIG_FILE/subprocess를 재패치해 override.
    """
    try:
        from corvin_jarvis import channels
    except Exception:
        return
    cfg = tmp_path_factory.mktemp("ch_safe") / "config.json"
    cfg.write_text(json.dumps({"notification": {"channels": ["log_only"]}}))
    monkeypatch.setattr(channels, "CONFIG_FILE", cfg)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """tmp_path 안에 timeseries.db 경로 반환 (파일은 미생성)."""
    return tmp_path / "timeseries.db"
