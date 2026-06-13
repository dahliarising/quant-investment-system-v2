"""Corvin Jarvis — Notification channel routing (single source of truth).

`notification.channels` in config.json decides which channels cron-driven
notifiers may use.

채널 이름: "telegram", "discord", "imessage", "log_only".
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Final

import requests

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


def _telegram_cfg() -> dict:
    cfg = _notification_cfg().get("telegram", {})
    return cfg if isinstance(cfg, dict) else {}


def telegram_bot_token() -> str | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or _telegram_cfg().get("bot_token")
    return token if isinstance(token, str) and token else None


def telegram_chat_id() -> str | None:
    cid = os.environ.get("TELEGRAM_CHAT_ID") or _telegram_cfg().get("chat_id")
    return str(cid) if cid else None


def _to_telegram_text(text: str) -> str:
    """Discord 마크다운 → Telegram MarkdownV2 근사 변환. 실패 시 plain text 사용."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)   # **bold** → *bold*
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)         # # 헤더 제거
    text = re.sub(r"(?m)^>{1}\s?", "", text)            # > 인용 제거
    return text


def send_telegram(body: str) -> bool:
    """Telegram Bot API sendMessage. token/chat_id는 env 또는 config에서."""
    if not is_enabled("telegram"):
        return False
    token = telegram_bot_token()
    chat_id = telegram_chat_id()
    if not token or not chat_id:
        log.warning("Telegram enabled이나 bot_token 또는 chat_id 미설정")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": _to_telegram_text(body), "parse_mode": "Markdown",
               "link_preview_options": {"is_disabled": True}}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True
        # Markdown 파싱 실패 시 plain text 재시도
        if r.status_code == 400:
            r2 = requests.post(url, json={"chat_id": chat_id, "text": body[:4000]}, timeout=15)
            return r2.status_code == 200
        log.warning("Telegram HTTP %d: %s", r.status_code, r.text[:200])
        return False
    except requests.RequestException as e:
        log.warning("Telegram send failed: %s", e)
        return False


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
