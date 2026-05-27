"""Corvin Jarvis — Compare (Phase 2)

최신 snapshot을 직전 snapshot + config thresholds와 비교하여 alert을 생성.

사용:
    python corvin_jarvis/compare.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
SNAPSHOTS_DIR = STATE_DIR / "snapshots"
LATEST_FILE = STATE_DIR / "latest.json"
CONFIG_FILE = BASE_DIR / "config.json"
ALERTS_FILE = STATE_DIR / "alerts.json"
LOG_FILE = STATE_DIR / "compare.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("corvin.compare")


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


@dataclass(frozen=True)
class Alert:
    category: str
    metric: str
    severity: Severity
    message: str
    value: float | None
    threshold: float | None
    delta_from_prev: float | None = None


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.error("JSON parse failed for %s: %s", path, e)
        return None


def _classify_severity(value: float, threshold: float) -> Severity:
    """절대값 / 임계값 비율로 severity 결정."""
    ratio = abs(value) / abs(threshold) if threshold else 0
    if ratio >= 3:
        return Severity.CRITICAL
    if ratio >= 2:
        return Severity.HIGH
    if ratio >= 1:
        return Severity.MEDIUM
    return Severity.LOW


def _previous_snapshot() -> dict[str, Any] | None:
    if not SNAPSHOTS_DIR.exists():
        return None
    files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    if len(files) < 2:
        return None
    return _load(files[-2])


def check_indices(latest: dict[str, Any], thresholds: dict[str, float]) -> list[Alert]:
    alerts: list[Alert] = []
    indices = latest.get("indices", {})
    for name, threshold_pct in thresholds.items():
        q = indices.get(name)
        if not q or q.get("price") is None:
            continue
        pct = q.get("pct_change")
        if pct is None:
            continue
        if abs(pct) >= threshold_pct:
            sev = _classify_severity(pct, threshold_pct)
            direction = "급락" if pct < 0 else "급등"
            alerts.append(Alert(
                category="index",
                metric=name,
                severity=sev,
                message=f"{name.upper()} {direction} {pct:+.2f}% (임계 ±{threshold_pct}%)",
                value=pct,
                threshold=threshold_pct,
            ))
    return alerts


def check_vix(latest: dict[str, Any], thresholds: dict[str, float]) -> list[Alert]:
    alerts: list[Alert] = []
    vix = latest.get("indices", {}).get("vix", {})
    price = vix.get("price")
    pct = vix.get("pct_change")
    abs_level = thresholds.get("absolute_level", 25)
    jump_pct = thresholds.get("daily_jump_pct", 15)

    if price is not None and price >= abs_level:
        sev = _classify_severity(price, abs_level)
        alerts.append(Alert(
            category="risk", metric="vix_level", severity=sev,
            message=f"VIX 절대값 {price:.2f} (임계 {abs_level} 이상) — 공포 모드 진입",
            value=price, threshold=abs_level,
        ))
    if pct is not None and abs(pct) >= jump_pct:
        sev = _classify_severity(pct, jump_pct)
        direction = "급등" if pct > 0 else "급락"
        alerts.append(Alert(
            category="risk", metric="vix_jump", severity=sev,
            message=f"VIX 일변동 {pct:+.2f}% {direction} (임계 ±{jump_pct}%)",
            value=pct, threshold=jump_pct,
        ))
    return alerts


def check_commodities(latest: dict[str, Any], thresholds: dict[str, float]) -> list[Alert]:
    alerts: list[Alert] = []
    cmds = latest.get("commodities", {})
    for name, threshold_pct in thresholds.items():
        q = cmds.get(name)
        if not q:
            continue
        pct = q.get("pct_change")
        if pct is None or abs(pct) < threshold_pct:
            continue
        sev = _classify_severity(pct, threshold_pct)
        direction = "급등" if pct > 0 else "급락"
        alerts.append(Alert(
            category="commodity", metric=name, severity=sev,
            message=f"{name.upper()} {direction} {pct:+.2f}% (임계 ±{threshold_pct}%)",
            value=pct, threshold=threshold_pct,
        ))
    return alerts


def check_fx(latest: dict[str, Any], thresholds: dict[str, Any]) -> list[Alert]:
    alerts: list[Alert] = []
    fx = latest.get("fx", {})
    usd_krw = fx.get("usd_krw", {})
    price = usd_krw.get("price")
    pct = usd_krw.get("pct_change")
    abs_lvl = thresholds.get("usd_krw_absolute", 1500)
    daily_pct = thresholds.get("usd_krw_daily_pct", 1.0)

    if price is not None and price >= abs_lvl:
        sev = _classify_severity(price - abs_lvl, abs_lvl * 0.02)
        alerts.append(Alert(
            category="fx", metric="usd_krw_level", severity=sev,
            message=f"USD/KRW {price:.2f} — {abs_lvl} 임계 초과 (원화 약세 가속)",
            value=price, threshold=abs_lvl,
        ))
    if pct is not None and abs(pct) >= daily_pct:
        sev = _classify_severity(pct, daily_pct)
        alerts.append(Alert(
            category="fx", metric="usd_krw_daily", severity=sev,
            message=f"USD/KRW 일변동 {pct:+.2f}% (임계 ±{daily_pct}%)",
            value=pct, threshold=daily_pct,
        ))
    return alerts


def check_portfolio(latest: dict[str, Any], thresholds: dict[str, float]) -> list[Alert]:
    alerts: list[Alert] = []
    pnl_alert = thresholds.get("pnl_alert_pct", 10.0)
    stop_loss = thresholds.get("stop_loss_pct", -8.0)
    take_profit = thresholds.get("take_profit_pct", 25.0)

    for pos in latest.get("portfolio", []):
        sym = pos.get("symbol")
        pnl = pos.get("pnl_pct")
        if pnl is None:
            continue

        if pnl <= stop_loss:
            alerts.append(Alert(
                category="portfolio", metric=f"stop_loss_{sym}",
                severity=Severity.CRITICAL,
                message=f"🛑 {sym} STOP LOSS 도달: PnL {pnl:+.2f}% (임계 {stop_loss}%) — 손절 검토",
                value=pnl, threshold=stop_loss,
            ))
        elif pnl >= take_profit:
            alerts.append(Alert(
                category="portfolio", metric=f"take_profit_{sym}",
                severity=Severity.HIGH,
                message=f"🎯 {sym} TAKE PROFIT 도달: PnL {pnl:+.2f}% (임계 +{take_profit}%) — 차익실현 검토",
                value=pnl, threshold=take_profit,
            ))
        elif abs(pnl) >= pnl_alert:
            direction = "수익" if pnl > 0 else "손실"
            alerts.append(Alert(
                category="portfolio", metric=f"pnl_{sym}",
                severity=Severity.MEDIUM,
                message=f"{sym} {direction} {pnl:+.2f}% (관찰 임계 ±{pnl_alert}%)",
                value=pnl, threshold=pnl_alert,
            ))
    return alerts


def check_delta(latest: dict[str, Any], prev: dict[str, Any] | None) -> list[Alert]:
    """직전 snapshot과 비교한 delta — VIX 급변, KOSPI 추가 하락 등 가속 신호."""
    if prev is None:
        return []
    alerts: list[Alert] = []

    # VIX delta acceleration
    v_now = latest.get("indices", {}).get("vix", {}).get("price")
    v_prev = prev.get("indices", {}).get("vix", {}).get("price")
    if v_now and v_prev and v_now >= v_prev * 1.1:
        alerts.append(Alert(
            category="acceleration", metric="vix_intra_pulse",
            severity=Severity.HIGH,
            message=f"VIX intra-pulse 가속: {v_prev:.2f} → {v_now:.2f} (직전 대비 +10% 이상)",
            value=v_now, threshold=v_prev * 1.1, delta_from_prev=v_now - v_prev,
        ))

    return alerts


def run_compare() -> list[Alert]:
    latest = _load(LATEST_FILE)
    if latest is None:
        log.error("latest.json 없음 — pulse.py를 먼저 실행")
        return []

    config = _load(CONFIG_FILE) or {}
    th = config.get("alert_thresholds", {})

    prev = _previous_snapshot()

    alerts: list[Alert] = []
    alerts.extend(check_indices(latest, th.get("index_pct_change", {})))
    alerts.extend(check_vix(latest, th.get("vix", {})))
    alerts.extend(check_commodities(latest, th.get("commodity_pct_change", {})))
    alerts.extend(check_fx(latest, th.get("fx", {})))
    alerts.extend(check_portfolio(latest, th.get("portfolio_position", {})))
    alerts.extend(check_delta(latest, prev))

    alerts.sort(key=lambda a: SEVERITY_ORDER[a.severity], reverse=True)

    ALERTS_FILE.write_text(json.dumps(
        {
            "generated_at": datetime.now().isoformat(),
            "snapshot_ts_kst": latest.get("timestamp_kst"),
            "count": len(alerts),
            "alerts": [asdict(a) for a in alerts],
        },
        indent=2, default=str, ensure_ascii=False,
    ))

    log.info("=== Compare Result ===")
    log.info("총 %d개 alert", len(alerts))
    for a in alerts:
        emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "·"}[a.severity.value]
        log.info("  %s [%s] %s", emoji, a.severity.value.upper(), a.message)

    return alerts


if __name__ == "__main__":
    run_compare()
