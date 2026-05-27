"""Corvin Jarvis — Multi-channel Notification (Phase 4 강화)

Discord webhook + iMessage + file 다중 채널 fallback.
webhook 없어도 iMessage로 alert 발송 (macOS osascript 활용).

채널 우선순위:
    1. Discord webhook (있으면)
    2. iMessage (macOS, 항상 시도)
    3. file (state/push_pending.json, 안전망)

사용:
    python corvin_jarvis/notify.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
ALERTS_FILE = STATE_DIR / "alerts.json"
BRIEFING_FILE = STATE_DIR / "briefing.md"
CONFIG_FILE = BASE_DIR / "config.json"
PORTFOLIO_FILE = BASE_DIR.parent / "portfolio.json"
DEDUP_FILE = STATE_DIR / "push_dedup.json"
PENDING_FILE = STATE_DIR / "push_pending.json"
LOG_FILE = STATE_DIR / "notify.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("corvin.notify")

SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEV_EMOJI = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "·"}
DEFAULT_COOLDOWN = 3600


@dataclass(frozen=True)
class NotifyResult:
    pushed: int
    skipped_dedup: int
    skipped_severity: int
    channels_delivered: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _config() -> dict[str, Any]:
    return _load_json(CONFIG_FILE)


def _min_severity() -> str:
    return _config().get("notification", {}).get("minimum_alert_severity", "high")


def _imessage_recipient() -> str | None:
    rec = _config().get("notification", {}).get("imessage_recipient")
    return rec if rec else None


def _webhook_url() -> str | None:
    env = os.environ.get("CORVIN_DISCORD_WEBHOOK")
    if env:
        return env.strip()
    url = _config().get("notification", {}).get("discord_webhook_url")
    return url if isinstance(url, str) and url.startswith("http") else None


def _filter_severity(alerts: list[dict[str, Any]], min_sev: str) -> list[dict[str, Any]]:
    th = SEV_RANK.get(min_sev, 2)
    return [a for a in alerts if SEV_RANK.get(a["severity"], 0) >= th]


def _filter_dedup(alerts: list[dict[str, Any]], cooldown_s: int) -> tuple[list[dict[str, Any]], int]:
    dedup = _load_json(DEDUP_FILE)
    now = datetime.now(timezone.utc)
    fresh: list[dict[str, Any]] = []
    skipped = 0
    for a in alerts:
        key = f"{a['category']}::{a['metric']}::{a['severity']}"
        last = dedup.get(key)
        if last:
            try:
                if now - datetime.fromisoformat(last) < timedelta(seconds=cooldown_s):
                    skipped += 1
                    continue
            except ValueError:
                pass
        dedup[key] = now.isoformat()
        fresh.append(a)
    DEDUP_FILE.write_text(json.dumps(dedup, indent=2))
    return fresh, skipped


def _held_symbols() -> set[str]:
    pf = _load_json(PORTFOLIO_FILE)
    return {str(h["symbol"]) for h in pf.get("holdings", []) if h.get("symbol")}


def _actionability(alert: dict[str, Any], held: set[str]) -> int:
    blob = f"{alert.get('metric', '')} {alert.get('message', '')}"
    return 1 if any(sym in blob for sym in held) else 0


def _rank_alerts(alerts: list[dict[str, Any]], held: set[str]) -> list[dict[str, Any]]:
    return sorted(
        alerts,
        key=lambda a: (SEV_RANK.get(a["severity"], 0), _actionability(a, held), abs(a.get("value") or 0)),
        reverse=True,
    )


def _format_message(alerts: list[dict[str, Any]], compact: bool = False) -> str:
    if not alerts:
        return ""
    header = f"🦅 Corvin Jarvis — {datetime.now().strftime('%H:%M KST')}"
    if compact:
        lines = [header, f"신규 alert {len(alerts)}건:"]
        for a in alerts[:5]:
            emoji = SEV_EMOJI[a["severity"]]
            lines.append(f"{emoji} {a['message'][:120]}")
        if len(alerts) > 5:
            lines.append(f"…외 {len(alerts) - 5}건")
        return "\n".join(lines)
    lines = [f"## {header}", f"\n신규 alert **{len(alerts)}건**:\n"]
    for a in alerts:
        lines.append(f"{SEV_EMOJI[a['severity']]} **[{a['severity'].upper()}]** {a['message']}")
    lines.append("\n_상세: briefing.md_")
    return "\n".join(lines)


def _send_discord(webhook: str, content: str) -> bool:
    try:
        r = requests.post(webhook, json={"content": content[:1900]}, timeout=15)
        return r.status_code in (200, 204)
    except requests.RequestException as e:
        log.error("Discord error: %s", e)
        return False


def _send_imessage(recipient: str, body: str) -> bool:
    """macOS Messages.app via osascript. Escape special chars."""
    safe = body.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
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


def _queue_file(content: str, alert_count: int) -> None:
    pending = _load_json(PENDING_FILE) or {"messages": []}
    pending.setdefault("messages", []).append({
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "alert_count": alert_count,
    })
    PENDING_FILE.write_text(json.dumps(pending, indent=2, ensure_ascii=False))


def notify(cooldown_s: int = DEFAULT_COOLDOWN) -> NotifyResult:
    alerts = _load_json(ALERTS_FILE).get("alerts", [])
    min_sev = _min_severity()

    sev_filtered = _filter_severity(alerts, min_sev)
    skipped_sev = len(alerts) - len(sev_filtered)
    fresh, skipped_dedup = _filter_dedup(sev_filtered, cooldown_s)

    log.info(
        "alerts=%d sev_pass=%d fresh=%d (skip_sev=%d, skip_dedup=%d)",
        len(alerts), len(sev_filtered), len(fresh), skipped_sev, skipped_dedup,
    )

    if not fresh:
        return NotifyResult(pushed=0, skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, channels_delivered=[])

    msg_long = _format_message(fresh, compact=False)
    msg_short = _format_message(fresh, compact=True)
    delivered: list[str] = []

    webhook = _webhook_url()
    if webhook:
        if _send_discord(webhook, msg_long):
            delivered.append("discord")
            log.info("Discord 전송 성공")
        else:
            log.warning("Discord 전송 실패")

    imsg = _imessage_recipient()
    if imsg:
        if _send_imessage(imsg, msg_short):
            delivered.append("imessage")
            log.info("iMessage 전송 성공 → %s", imsg)
        else:
            log.warning("iMessage 전송 실패 — file queue fallback")

    if not delivered:
        _queue_file(msg_long, len(fresh))
        log.warning("어떤 채널도 전송 실패 — pending file에 적재")

    return NotifyResult(
        pushed=len(fresh),
        skipped_dedup=skipped_dedup,
        skipped_severity=skipped_sev,
        channels_delivered=delivered,
    )


if __name__ == "__main__":
    result = notify()
    log.info("결과: %s", result)
