"""Corvin Jarvis — Notification channel routing (single source of truth).

`notification.channels` in config.json decides which channels cron-driven
notifiers may use. Default routing (2026-05-28, 폐하 지시): iMessage only —
크론잡 alert은 Discord로 보내지 않고 iMessage로만 보낸다.

채널 이름: "discord", "imessage", "log_only".
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Final

CONFIG_FILE: Final = Path(__file__).resolve().parent / "config.json"

# config에 channels가 없을 때의 안전 기본값 — iMessage only.
DEFAULT_CHANNELS: Final = ("imessage", "log_only")

log = logging.getLogger("corvin.channels")


def _notification_cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text()).get("notification", {})
    except (OSError, json.JSONDecodeError):
        return {}


def enabled_channels() -> set[str]:
    chans = _notification_cfg().get("channels")
    if not isinstance(chans, list) or not chans:
        return set(DEFAULT_CHANNELS)
    return {str(c) for c in chans}


def is_enabled(channel: str) -> bool:
    return channel in enabled_channels()


def imessage_recipient() -> str | None:
    rec = _notification_cfg().get("imessage_recipient")
    return rec if isinstance(rec, str) and rec else None


def to_imessage_text(text: str) -> str:
    """Discord 마크다운을 iMessage용 평문으로. iMessage는 마크다운 렌더 없음."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)        # **굵게**
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", text)  # # 헤더
    text = re.sub(r"(?m)^[ \t]*>[ \t]?", "", text)       # > 인용
    text = re.sub(r"_([^_\n]+?)_", r"\1", text)          # _기울임_
    return text


def send_imessage(body: str) -> bool:
    """macOS Messages.app via osascript. recipient는 config에서. 비활성/미설정이면 False."""
    if not is_enabled("imessage"):
        return False
    recipient = imessage_recipient()
    if not recipient:
        log.warning("iMessage enabled이나 recipient 미설정")
        return False
    # AppleScript 문자열 이스케이프는 백슬래시·따옴표만 필요 ($ 는 특수문자 아님).
    safe = to_imessage_text(body).replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to participant "{recipient}" of targetService
    send "{safe}" to targetBuddy
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True
        log.warning("iMessage stderr: %s", result.stderr.strip()[:200])
        return False
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        log.warning("iMessage send failed: %s", e)
        return False
