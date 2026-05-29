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
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from . import channels, market_hours
except ImportError:  # script/launchd 실행 (python notify.py)
    import channels  # type: ignore[no-redef]
    import market_hours  # type: ignore[no-redef]

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
ALERTS_FILE = STATE_DIR / "alerts.json"
VERDICTS_FILE = STATE_DIR / "verdicts.json"
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


_SECTOR_KO = {
    "semiconductor": "반도체", "battery": "2차전지", "bio": "바이오",
    "auto": "자동차", "internet": "인터넷", "steel": "철강",
    "shipbuilding": "조선", "defense": "방산", "bigtech": "빅테크",
    "finance": "금융", "pharma": "제약",
    "humanoid": "휴머노이드", "space": "우주", "stem_cell": "줄기세포", "quantum": "양자",
}


def _friendly_name(alert: dict[str, Any], fallback: str) -> str:
    """alert message의 '종목명(코드)' 패턴에서 사람이 읽는 종목명 추출. 없으면 fallback."""
    msg = alert.get("message", "")
    if "(" in msg:
        name = msg.split("(", 1)[0].replace("✅", "").replace("🟡", "").strip()
        if name:
            return name
    return fallback


def _interpret(alert: dict[str, Any]) -> str:
    """alert을 일상어 한 줄 해석으로 변환 (전문용어 제거)."""
    cat = alert.get("category", "")
    metric = alert.get("metric", "")
    v = alert.get("value")
    v = v if isinstance(v, (int, float)) else 0.0
    up = v > 0
    parts = metric.split("_")

    if cat == "universe":
        sym = parts[1] if len(parts) > 1 else metric
        name = _friendly_name(alert, sym)
        return f"{name} 주가가 {abs(v):.1f}% {'급등' if up else '급락'} — 큰 변동이라 주목"
    if cat == "sector":
        sec = parts[1] if len(parts) > 1 else ""
        return f"{_SECTOR_KO.get(sec, sec)} 업종이 평균 {abs(v):.1f}% {'동반 상승' if up else '동반 하락'} — 섹터 전체 움직임"
    if cat == "leading_rs":
        sym = parts[1] if len(parts) > 1 else metric
        name = _friendly_name(alert, sym)
        return f"{name}이(가) 시장 지수보다 {abs(v):.1f}%p {'더 강함 → 주도주' if up else '더 약함 → 소외주'}"
    if cat == "portfolio":
        sym = parts[-1]
        if metric.startswith("stop_loss"):
            return f"보유 {sym} 손절선 도달 — 손실 관리 검토 필요"
        if metric.startswith("take_profit"):
            return f"보유 {sym} 익절선 도달 — 차익실현 검토"
        return f"보유 {sym} 평가손익 {v:+.1f}% — 관찰 수준(아직 급한 액션 아님)"
    if cat == "index":
        return f"{metric.upper()} 지수 {abs(v):.1f}% {'상승' if up else '하락'} — 시장 전체 분위기"
    if cat == "commodity":
        return f"{metric.upper()}(원자재) {abs(v):.1f}% {'상승' if up else '하락'}"
    if cat == "fx":
        if "daily" in metric:
            return f"원/달러 환율이 하루새 {abs(v):.1f}% {'상승(원화 약세)' if up else '하락(원화 강세)'}"
        return f"원/달러 {v:.0f}원 — 원화 약세 영역(보유 미국주식엔 환차익 우호)"
    if cat == "risk":
        return "공포지수(VIX) 상승 — 시장 불안 신호"
    if cat == "acceleration":
        return "직전보다 변동성 가속 — 빠르게 상황 변화 중"
    if cat == "narrative":
        if "foreign" in metric:
            return "외국인이 한국주식을 평소보다 훨씬 많이 순매수 — 이례적 급증(증시 강세 신호)"
        return "시장 자금·심리 흐름에 이례적 변화 감지"
    if cat == "earnings":
        return "실적 발표 임박 — 변동성 주의"
    if cat == "regime":
        return "시장 국면(위험선호↔회피) 전환 신호"
    return alert.get("message", "")


_ACTION_EMOJI = {"매수": "🟢", "분할매수": "🔵", "홀딩": "⚪", "비중축소": "🟠", "매도": "🔴", "관망": "⏸"}


def _format_message(alerts: list[dict[str, Any]], compact: bool = False,
                    title: str = "", limit: int = 5,
                    verdicts: dict[str, Any] | None = None) -> str:
    if not alerts:
        return ""
    header = title or f"🦅 Corvin Jarvis — {datetime.now().strftime('%H:%M KST')}"
    shown = alerts[:limit]
    extra = len(alerts) - len(shown)
    # 의미있는 판정만 노출 — "관망+신뢰도 하"(신호 없음)는 노이즈라 숨김
    vshow = {s: vd for s, vd in (verdicts or {}).items()
             if vd.get("action") != "관망" or vd.get("confidence") != "하"}
    if compact:
        lines = [header, f"신규 alert {len(alerts)}건:"]
        for a in shown:
            emoji = SEV_EMOJI[a["severity"]]
            lines.append(f"{emoji} {a['message'][:120]}")
        if extra > 0:
            lines.append(f"…외 {extra}건")
        lines.append("")
        lines.append("📖 해석")
        for a in shown:
            lines.append(f"• {_interpret(a)}")
        if vshow:
            lines.append("")
            lines.append("🎯 판정")  # compact는 짧게 (non-compact는 "행동 판정")
            for sym, vd in vshow.items():
                label = vd.get("name") or sym
                act = vd.get("action", "?")
                lines.append(f"{_ACTION_EMOJI.get(act, '')} {label} {act}")
        return "\n".join(lines)
    lines = [f"## {header}", f"\n신규 alert **{len(alerts)}건**:\n"]
    for a in shown:
        lines.append(f"{SEV_EMOJI[a['severity']]} **[{a['severity'].upper()}]** {a['message']}")
    if extra > 0:
        lines.append(f"\n…외 {extra}건")
    lines.append("\n_상세: briefing.md_")
    lines.append("\n**📖 해석**")
    for a in shown:
        lines.append(f"- {_interpret(a)}")
    if vshow:
        lines.append("\n**🎯 행동 판정**")
        for sym, vd in vshow.items():
            act = vd.get("action", "?")
            e = _ACTION_EMOJI.get(act, "")
            label = f"{vd['name']}({sym})" if vd.get("name") else sym
            lines.append(f"{e} **{label} {act}** (신뢰도 {vd.get('confidence', '?')}) — {vd.get('rationale', '')}")
    return "\n".join(lines)


def _send_discord(webhook: str, content: str) -> bool:
    try:
        r = requests.post(webhook, json={"content": content[:1900]}, timeout=15)
        return r.status_code in (200, 204)
    except requests.RequestException as e:
        log.error("Discord error: %s", e)
        return False


def _queue_file(content: str, alert_count: int) -> None:
    pending = _load_json(PENDING_FILE) or {"messages": []}
    pending.setdefault("messages", []).append({
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "alert_count": alert_count,
    })
    PENDING_FILE.write_text(json.dumps(pending, indent=2, ensure_ascii=False))


def notify(mode: str = "urgent", cooldown_s: int = DEFAULT_COOLDOWN) -> NotifyResult:
    alerts = _load_json(ALERTS_FILE).get("alerts", [])
    cfg = _config().get("notification", {})
    held = _held_symbols()

    if mode == "digest":
        min_sev = cfg.get("digest_min_severity", "medium")
        limit = int(cfg.get("digest_max_items", 8))
        title = f"📋 Corvin 일일 다이제스트 — {datetime.now().strftime('%m/%d %H:%M KST')}"
    else:
        min_sev = cfg.get("urgent_min_severity", "high")
        limit = int(cfg.get("max_per_push", 8))
        title = f"🦅 Corvin 긴급 — {datetime.now().strftime('%H:%M KST')}"

    sev_filtered = _filter_severity(alerts, min_sev)
    skipped_sev = len(alerts) - len(sev_filtered)

    if mode == "digest":
        fresh, skipped_dedup = sev_filtered, 0
    else:
        fresh, skipped_dedup = _filter_dedup(sev_filtered, cooldown_s)

    fresh = _rank_alerts(fresh, held)

    # 장마감 시장 억제 — urgent만 (다이제스트는 전체 요약이라 미적용). 거시는 항상 통과.
    skipped_closed = 0
    if mode != "digest":
        now = datetime.now(timezone.utc)
        sector_markets = market_hours.load_sector_markets()
        kept = [a for a in fresh if not market_hours.should_suppress(a, now, sector_markets)]
        skipped_closed = len(fresh) - len(kept)
        fresh = kept

    log.info(
        "mode=%s alerts=%d sev_pass=%d fresh=%d (skip_sev=%d, skip_dedup=%d, skip_closed=%d)",
        mode, len(alerts), len(sev_filtered), len(fresh), skipped_sev, skipped_dedup, skipped_closed,
    )

    if not fresh:
        return NotifyResult(pushed=0, skipped_dedup=skipped_dedup, skipped_severity=skipped_sev, channels_delivered=[])

    verdicts = _load_json(VERDICTS_FILE) or None
    msg_long = _format_message(fresh, compact=False, title=title, limit=limit, verdicts=verdicts)
    msg_short = _format_message(fresh, compact=True, title=title, limit=limit, verdicts=verdicts)
    delivered: list[str] = []

    webhook = _webhook_url()
    if channels.is_enabled("discord") and webhook:
        if _send_discord(webhook, msg_long):
            delivered.append("discord")
            log.info("Discord 전송 성공")
        else:
            log.warning("Discord 전송 실패")

    if channels.is_enabled("imessage"):
        if channels.send_imessage(msg_short):
            delivered.append("imessage")
            log.info("iMessage 전송 성공 → %s", channels.imessage_recipient())
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "urgent"
    result = notify(mode=mode)
    log.info("결과(%s): %s", mode, result)
