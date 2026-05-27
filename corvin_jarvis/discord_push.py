"""Corvin Jarvis — Discord Push (Phase 4)

briefing.md를 Discord webhook으로 push. severity 기반 filtering + 1시간 cooldown.

설정:
    - 환경변수: CORVIN_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
    - 또는 config.json의 notification.discord_webhook_url

webhook 없으면 file-only 모드로 작동 (state/push_pending.json에 적재).

사용:
    python corvin_jarvis/discord_push.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
ALERTS_FILE = STATE_DIR / "alerts.json"
BRIEFING_FILE = STATE_DIR / "briefing.md"
CONFIG_FILE = BASE_DIR / "config.json"
DEDUP_FILE = STATE_DIR / "push_dedup.json"
PENDING_FILE = STATE_DIR / "push_pending.json"
LOG_FILE = STATE_DIR / "push.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("corvin.push")

SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEV_EMOJI = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "·"}
DEFAULT_COOLDOWN_SECONDS = 3600  # 1시간


@dataclass(frozen=True)
class PushResult:
    pushed: int
    skipped_dedup: int
    skipped_severity: int
    delivered: bool


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _get_webhook_url() -> str | None:
    env = os.environ.get("CORVIN_DISCORD_WEBHOOK")
    if env:
        return env.strip()
    cfg = _load_json(CONFIG_FILE)
    url = cfg.get("notification", {}).get("discord_webhook_url")
    return url if isinstance(url, str) and url.startswith("http") else None


def _min_severity() -> str:
    cfg = _load_json(CONFIG_FILE)
    return cfg.get("notification", {}).get("minimum_alert_severity", "high")


def _filter_by_severity(alerts: list[dict[str, Any]], min_sev: str) -> list[dict[str, Any]]:
    threshold = SEV_RANK.get(min_sev, 2)
    return [a for a in alerts if SEV_RANK.get(a["severity"], 0) >= threshold]


def _filter_by_dedup(alerts: list[dict[str, Any]], cooldown_s: int) -> tuple[list[dict[str, Any]], int]:
    dedup = _load_json(DEDUP_FILE)
    now = datetime.now(timezone.utc)
    fresh: list[dict[str, Any]] = []
    skipped = 0

    for a in alerts:
        key = f"{a['category']}::{a['metric']}::{a['severity']}"
        last_iso = dedup.get(key)
        if last_iso:
            try:
                last = datetime.fromisoformat(last_iso)
                if now - last < timedelta(seconds=cooldown_s):
                    skipped += 1
                    continue
            except ValueError:
                pass
        dedup[key] = now.isoformat()
        fresh.append(a)

    DEDUP_FILE.write_text(json.dumps(dedup, indent=2))
    return fresh, skipped


def _format_discord_message(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return ""
    lines = [f"## 🦅 **Corvin Jarvis Alert** — {datetime.now().strftime('%H:%M KST')}"]
    lines.append(f"\n신규 alert **{len(alerts)}건**:\n")
    for a in alerts:
        emoji = SEV_EMOJI[a["severity"]]
        lines.append(f"{emoji} **[{a['severity'].upper()}]** {a['message']}")
    lines.append("\n_상세: briefing.md_")
    return "\n".join(lines)


def _post_webhook(webhook_url: str, content: str) -> bool:
    parsed = urlparse(webhook_url)
    if parsed.scheme != "https" or "discord" not in parsed.netloc:
        log.error("유효하지 않은 Discord webhook URL")
        return False
    try:
        r = requests.post(
            webhook_url,
            json={"content": content[:1900]},
            timeout=15,
        )
        if r.status_code in (200, 204):
            return True
        log.error("Discord push failed: %s %s", r.status_code, r.text[:200])
        return False
    except requests.RequestException as e:
        log.error("Discord post exception: %s", e)
        return False


def push_alerts(cooldown_s: int = DEFAULT_COOLDOWN_SECONDS) -> PushResult:
    alerts_data = _load_json(ALERTS_FILE)
    raw_alerts = alerts_data.get("alerts", [])

    min_sev = _min_severity()
    sev_filtered = _filter_by_severity(raw_alerts, min_sev)
    skipped_sev = len(raw_alerts) - len(sev_filtered)

    fresh, skipped_dedup = _filter_by_dedup(sev_filtered, cooldown_s)

    log.info(
        "총 alert=%d, severity 통과=%d, dedup 통과=%d (skipped_sev=%d, skipped_dedup=%d)",
        len(raw_alerts), len(sev_filtered), len(fresh), skipped_sev, skipped_dedup,
    )

    if not fresh:
        return PushResult(pushed=0, skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, delivered=False)

    msg = _format_discord_message(fresh)
    webhook = _get_webhook_url()

    if webhook:
        ok = _post_webhook(webhook, msg)
        log.info("Discord push %s — %d alerts", "성공" if ok else "실패", len(fresh))
        return PushResult(pushed=len(fresh), skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, delivered=ok)

    # Webhook 없으면 pending file에 적재
    pending = _load_json(PENDING_FILE) or {"messages": []}
    pending.setdefault("messages", []).append({
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "content": msg,
        "alert_count": len(fresh),
    })
    PENDING_FILE.write_text(json.dumps(pending, indent=2, ensure_ascii=False))
    log.warning("Webhook 없음 — pending file에 적재됨: %s", PENDING_FILE)
    return PushResult(pushed=len(fresh), skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, delivered=False)


if __name__ == "__main__":
    result = push_alerts()
    log.info("결과: %s", result)
