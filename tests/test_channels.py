"""Tests for corvin_jarvis.channels — notification channel routing."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from corvin_jarvis import channels


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notification: dict[str, Any]) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"notification": notification}))
    monkeypatch.setattr(channels, "CONFIG_FILE", cfg)


def test_is_enabled_reflects_config_list(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["imessage", "log_only"]})
    assert channels.is_enabled("imessage") is True
    assert channels.is_enabled("discord") is False


def test_defaults_to_imessage_only_when_channels_missing(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"imessage_recipient": "+10000000000"})
    assert channels.is_enabled("imessage") is True
    assert channels.is_enabled("discord") is False


def test_defaults_when_config_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "CONFIG_FILE", tmp_path / "missing.json")
    assert channels.enabled_channels() == set(channels.DEFAULT_CHANNELS)


def test_send_imessage_noop_when_channel_disabled(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["discord"], "imessage_recipient": "+10000000000"})

    def _boom(*_a, **_k):  # subprocess must not be called
        raise AssertionError("osascript should not run when imessage disabled")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert channels.send_imessage("hi") is False


def test_send_imessage_false_when_no_recipient(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["imessage"], "imessage_recipient": ""})
    assert channels.send_imessage("hi") is False


def test_send_imessage_invokes_osascript_on_success(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["imessage"], "imessage_recipient": "+10000000000"})
    calls: list[list[str]] = []

    def _fake_run(cmd, **_k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert channels.send_imessage("buy NVDA") is True
    assert calls and calls[0][0] == "osascript"
    assert "+10000000000" in calls[0][-1]


def test_to_imessage_text_strips_markdown():
    src = "📈 **DCA 후보** (US)\n_Regime: neutral_\n## 제목\n> 인용\n⭐ **BWXT** score **60**"
    out = channels.to_imessage_text(src)
    assert "**" not in out
    assert "BWXT score 60" in out
    assert "Regime: neutral" in out
    assert "제목" in out and "## " not in out
    assert out.count(">") == 0


def test_to_imessage_text_keeps_dollar_and_single_underscore():
    src = "종가 $199.27 → 실시간 $214.25 · take_profit"
    out = channels.to_imessage_text(src)
    assert "$199.27" in out and "$214.25" in out
    assert "take_profit" in out  # 단일 언더스코어는 보존


def test_send_imessage_does_not_backslash_escape_dollar(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["imessage"], "imessage_recipient": "+10000000000"})
    captured: list[str] = []

    def _fake_run(cmd, **_k):
        captured.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert channels.send_imessage("buy **NVDA** at $214.25") is True
    sent = captured[0]
    assert "\\$" not in sent          # 달러 앞 백슬래시 없음
    assert "$214.25" in sent
    assert "**" not in sent           # 마크다운 제거됨


def test_send_imessage_false_on_nonzero_returncode(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"channels": ["imessage"], "imessage_recipient": "+10000000000"})

    def _fail_run(cmd, **_k):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fail_run)
    assert channels.send_imessage("x") is False
