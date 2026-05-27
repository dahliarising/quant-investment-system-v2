"""Corvin Jarvis — Orchestrator (Phase 3)

cron entry point. 모든 phase를 묶어 briefing.md를 생성.

사용:
    python corvin_jarvis/jarvis.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

sys.path.insert(0, str(BASE_DIR.parent))

from compare import run_compare  # noqa: E402
from corvin_jarvis import earnings  # noqa: E402
from corvin_jarvis import narrative  # noqa: E402
from corvin_jarvis import attribution  # noqa: E402
from corvin_jarvis import predict  # noqa: E402
from corvin_jarvis import regime  # noqa: E402
from geo_signal import build_geo_signal  # noqa: E402
from narrate import build_context  # noqa: E402
from pulse import run_pulse  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
STATE_DIR = BASE_DIR / "state"
BRIEFING_FILE = STATE_DIR / "briefing.md"
BRIEFING_HISTORY = STATE_DIR / "briefings"
LATEST_FILE = STATE_DIR / "latest.json"
ALERTS_FILE = STATE_DIR / "alerts.json"
WIKI_DIR = Path("/Users/thethethe/Claude/llm-wiki/wiki/corvin-sessions")

log = logging.getLogger("corvin.jarvis")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _recent_wiki_strategies(limit: int = 3) -> list[Path]:
    if not WIKI_DIR.exists():
        return []
    files = sorted(WIKI_DIR.glob("*.md"), reverse=True)
    return files[:limit]


def _format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _format_alerts_section(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return "## 🟢 알람\n\n현재 임계값을 초과한 신호 없음.\n"
    by_sev: dict[str, list[dict[str, Any]]] = {"critical": [], "high": [], "medium": [], "low": []}
    for a in alerts:
        by_sev.setdefault(a["severity"], []).append(a)
    sections: list[str] = ["## 🚨 알람"]
    sev_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "·"}
    sev_label = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
    for sev in ["critical", "high", "medium", "low"]:
        if not by_sev[sev]:
            continue
        sections.append(f"\n### {sev_emoji[sev]} {sev_label[sev]} ({len(by_sev[sev])}건)\n")
        for a in by_sev[sev]:
            sections.append(f"- **[{a['category']}/{a['metric']}]** {a['message']}")
    return "\n".join(sections) + "\n"


def _format_market_section(latest: dict[str, Any]) -> str:
    indices = latest.get("indices", {})
    commodities = latest.get("commodities", {})
    fx = latest.get("fx", {})

    out = ["## 📊 시장 스냅샷\n"]
    out.append("| 카테고리 | 자산 | 가격 | 변동 |")
    out.append("|---|---|---|---|")
    for name, q in indices.items():
        out.append(f"| 지수 | {name.upper()} | {q.get('price')} | {_format_pct(q.get('pct_change'))} |")
    for name, q in commodities.items():
        out.append(f"| 원자재 | {name.upper()} | {q.get('price')} | {_format_pct(q.get('pct_change'))} |")
    for name, q in fx.items():
        out.append(f"| FX | {name.upper()} | {q.get('price')} | {_format_pct(q.get('pct_change'))} |")
    return "\n".join(out) + "\n"


def _format_portfolio_section(latest: dict[str, Any]) -> str:
    portfolio = latest.get("portfolio", [])
    summary = latest.get("portfolio_summary", {})

    out = ["## 💼 포트폴리오 상태\n"]
    out.append(f"- 총 가치: **KRW ₩{summary.get('total_value_krw'):,}** + **USD ${summary.get('total_value_usd'):,}**")
    out.append(f"- 포지션: {summary.get('position_count')}개 (수익 {summary.get('winners_count')}, 손실 {summary.get('losers_count')})")
    out.append(f"- 최고/최저 PnL: {_format_pct(summary.get('best_pnl_pct'))} / {_format_pct(summary.get('worst_pnl_pct'))}")
    out.append(f"- portfolio.json 마지막 업데이트: {summary.get('stale_as_of')}\n")

    out.append("| 종목 | 보유 | 평단 | 현재가 | PnL | 평가액 |")
    out.append("|---|---|---|---|---|---|")
    for pos in portfolio:
        sym = pos.get("symbol")
        shares = pos.get("shares")
        avg = pos.get("avg_price")
        cur = pos.get("current_price")
        pnl = pos.get("pnl_pct")
        mv = pos.get("market_value")
        ccy = pos.get("currency", "USD")
        sign = "₩" if ccy == "KRW" else "$"
        pnl_emoji = "🟢" if pnl and pnl > 0 else ("🔴" if pnl and pnl < 0 else "⚪")
        out.append(
            f"| {sym} | {shares} | {sign}{avg:,.2f} | {sign}{cur:,.2f} | "
            f"{pnl_emoji} {_format_pct(pnl)} | {sign}{mv:,.2f} |"
        )
    return "\n".join(out) + "\n"


def _format_continuity_section() -> str:
    ctx_path = STATE_DIR / "continuity_context.json"
    ctx = _load(ctx_path)
    if not ctx:
        return "## 🧠 전략 연속성\n\n과거 corvin-sessions 기록 없음.\n"

    out = ["## 🧠 전략 연속성\n"]
    themes = ctx.get("open_themes", [])
    if themes:
        out.append(f"**Open themes**: {', '.join(themes)}\n")

    pa = ctx.get("portfolio_alignment", {})
    aligned = pa.get("held_and_discussed", [])
    orphan = pa.get("held_but_not_in_recent_strategy", [])
    candidates = pa.get("mentioned_but_not_held", [])

    if aligned:
        out.append(f"- ✅ 보유 + 전략 일관: {', '.join(aligned)}")
    if orphan:
        out.append(f"- ⚠️ 보유 but 최근 전략 미언급 (orphan): {', '.join(orphan)}")
    if candidates:
        out.append(f"- 💡 후보 종목 (언급되었으나 미보유): {', '.join(candidates[:10])}")

    out.append("\n### 최근 세션")
    for s in ctx.get("sessions", [])[:3]:
        out.append(f"- `{s.get('file')}` — {s.get('title') or '(no title)'}")
        if s.get("tldr"):
            tldr = s["tldr"].replace("\n", " ").strip()
            out.append(f"  > {tldr[:200]}{'…' if len(tldr) > 200 else ''}")
    return "\n".join(out) + "\n"


def _format_recommendations_section(alerts: list[dict[str, Any]]) -> str:
    """단순 룰베이스 추천 — narrate phase에서 LLM이 강화."""
    critical = [a for a in alerts if a["severity"] == "critical"]
    high = [a for a in alerts if a["severity"] == "high"]
    tp = [a for a in alerts if a.get("metric", "").startswith("take_profit_")]
    sl = [a for a in alerts if a.get("metric", "").startswith("stop_loss_")]

    out = ["## 🎯 자동 추천 (advisory only)\n"]
    if not (critical or high or tp or sl):
        out.append("- 현 시점 강한 액션 신호 없음 — 관찰 유지\n")
        return "\n".join(out)

    if tp:
        out.append("### 차익실현 검토")
        for a in tp:
            sym = a["metric"].replace("take_profit_", "")
            out.append(f"- **{sym}** PnL {a['value']:+.2f}%: 일부(예: 30~50%) 차익실현 + 트레일링 스톱 검토")
    if sl:
        out.append("\n### 손절 트리거")
        for a in sl:
            sym = a["metric"].replace("stop_loss_", "")
            out.append(f"- **{sym}** PnL {a['value']:+.2f}%: -8% 룰 도달 — 즉시 손절 후 재진입 시나리오")
    if critical:
        out.append("\n### 시장 충격 대응")
        kospi_crit = [a for a in critical if a.get("metric") == "kospi"]
        if kospi_crit:
            out.append("- KOSPI critical 급락 — 한국 종목 비중 재검토, 신규 진입 보류")
        sp_crit = [a for a in critical if a.get("metric") == "sp500"]
        if sp_crit:
            out.append("- S&P500 critical 급락 — defensive sleeve(GLD/금/유틸리티) 비중 증가 검토")
    return "\n".join(out) + "\n"


def compose_briefing() -> str:
    latest = _load(LATEST_FILE) or {}
    alerts_data = _load(ALERTS_FILE) or {}
    alerts = alerts_data.get("alerts", [])

    now_kst = datetime.now(KST)
    header = (
        f"# 🦅 Corvin Jarvis Briefing\n\n"
        f"**Generated**: {now_kst.strftime('%Y-%m-%d %H:%M KST')}\n"
        f"**Snapshot**: {latest.get('timestamp_kst', 'N/A')}\n"
        f"**Alert 총합**: {len(alerts)}건 "
        f"(critical: {sum(1 for a in alerts if a['severity'] == 'critical')}, "
        f"high: {sum(1 for a in alerts if a['severity'] == 'high')})\n\n"
        f"---\n"
    )

    sections = [
        header,
        _format_geo_section(),
        _format_alerts_section(alerts),
        _format_recommendations_section(alerts),
        _format_market_section(latest),
        _format_portfolio_section(latest),
        _format_continuity_section(),
    ]
    return "\n".join(sections)


def _format_geo_section() -> str:
    geo = _load(STATE_DIR / "geo_signal.json")
    if not geo:
        return ""
    regime_emoji = {
        "CRISIS": "🆘", "RISK_OFF": "🛡️", "NEUTRAL": "⚖️",
        "RISK_ON": "🚀", "EUPHORIA": "🎢",
    }.get(geo.get("regime", ""), "")
    out = [f"## {regime_emoji} 지정학/리스크 시그널\n"]
    out.append(f"- **Composite Risk Score**: {geo.get('composite_score')} / 10")
    out.append(f"- **Regime**: {geo.get('regime')}")
    out.append(f"- **Drivers**:")
    for d in geo.get("headline_drivers", [])[:3]:
        out.append(f"  - {d}")
    out.append("")
    return "\n".join(out)


def _collect_us_symbols() -> list[str]:
    """holdings + watchlist 중 US 종목만 (yfinance.calendar용)."""
    syms: set[str] = set()
    pf_file = BASE_DIR.parent / "portfolio.json"
    if pf_file.exists():
        try:
            pf = json.loads(pf_file.read_text())
            for h in pf.get("holdings", []):
                syms.add(h["symbol"])
        except (json.JSONDecodeError, OSError):
            pass
    cfg_file = BASE_DIR / "config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
            for s in cfg.get("watchlist", []):
                syms.add(str(s))
        except (json.JSONDecodeError, OSError):
            pass
    return [s for s in sorted(syms)
            if not (s.isdigit() or s.endswith(".KS") or s.endswith(".KQ"))]


def refresh_earnings_and_merge_alerts() -> int:
    """holdings + watchlist 어닝 refresh + alerts.json에 D-N alert merge."""
    db_path = STATE_DIR / "timeseries.db"
    us_syms = _collect_us_symbols()
    log.info("Earnings refresh for %d US symbols: %s", len(us_syms), us_syms)
    earnings.refresh_earnings_calendar(db_path, us_syms)
    today = datetime.now(KST).date()
    e_alerts = earnings.build_earnings_alerts(db_path, today=today)
    if not e_alerts:
        log.info("No earnings D-7/D-3/D-1 today")
        return 0

    if ALERTS_FILE.exists():
        data = _load(ALERTS_FILE) or {}
        data.setdefault("alerts", []).extend(e_alerts)
        data["count"] = len(data["alerts"])
    else:
        data = {
            "alerts": e_alerts,
            "count": len(e_alerts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    log.info("Earnings alerts merged: %d", len(e_alerts))
    return len(e_alerts)


def merge_narrative_alerts() -> int:
    """narrative-shift-detector signals.db Z-score alerts merge."""
    n_alerts = narrative.build_narrative_alerts(
        narrative.DEFAULT_SIGNALS_DB, threshold=2.0, lookback_days=30, market="KR",
    )
    if not n_alerts:
        log.info("No narrative Z-score alerts (|Z|<2σ)")
        return 0
    if ALERTS_FILE.exists():
        data = _load(ALERTS_FILE) or {}
        data.setdefault("alerts", []).extend(n_alerts)
        data["count"] = len(data["alerts"])
    else:
        data = {
            "alerts": n_alerts,
            "count": len(n_alerts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    log.info("Narrative alerts merged: %d", len(n_alerts))
    return len(n_alerts)


def run_weekly_attribution(force: bool = False) -> Path | None:
    """일요일이면 weekly outcome attribution report 생성 → state/attribution-YYYYWW.json."""
    now_kst = datetime.now(KST)
    if not force and now_kst.weekday() != 6:  # 6 = Sunday
        return None
    wiki = Path("/Users/thethethe/Claude/llm-wiki/wiki/corvin-sessions")
    db_path = STATE_DIR / "timeseries.db"
    report = attribution.weekly_report(wiki, db_path, today=now_kst.date(), lookback_days=7)
    iso_year, iso_week, _ = now_kst.date().isocalendar()
    out_file = STATE_DIR / f"attribution-{iso_year}W{iso_week:02d}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log.info("Weekly attribution report → %s (%d sessions)", out_file.name, report["sessions_analyzed"])
    return out_file


def merge_predictive_alerts() -> int:
    """config.json predictive_levels로 forecasting alerts merge."""
    cfg_file = BASE_DIR / "config.json"
    if not cfg_file.exists():
        return 0
    try:
        cfg = json.loads(cfg_file.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    levels = cfg.get("predictive_levels", {})
    levels = {k: v for k, v in levels.items() if isinstance(v, dict)}
    if not levels:
        return 0
    db_path = STATE_DIR / "timeseries.db"
    p_alerts = predict.build_predictive_alerts(db_path, levels)
    if not p_alerts:
        return 0
    if ALERTS_FILE.exists():
        data = _load(ALERTS_FILE) or {}
        data.setdefault("alerts", []).extend(p_alerts)
        data["count"] = len(data["alerts"])
    else:
        data = {
            "alerts": p_alerts, "count": len(p_alerts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    log.info("Predictive alerts merged: %d", len(p_alerts))
    return len(p_alerts)


def merge_signal_alerts() -> int:
    """monitored universe 종목별 + 섹터 바스켓 alert을 phase 라벨과 함께 merge."""
    from corvin_jarvis.signals import market_phase, universe_monitor

    latest = _load(LATEST_FILE) or {}
    if not latest.get("universe"):
        return 0
    cfg = _load(BASE_DIR / "config.json") or {}
    th = cfg.get("alert_thresholds", {})
    now = datetime.now(KST)

    # market별로 universe를 쪼개 각자의 phase로 검사
    by_market: dict[str, list[dict[str, Any]]] = {}
    for e in latest["universe"]:
        by_market.setdefault(e.get("market", "US"), []).append(e)

    s_alerts: list[dict[str, Any]] = []
    for market, entries in by_market.items():
        phase = market_phase.phase_for(market, now)
        sub = {"universe": entries}
        s_alerts.extend(universe_monitor.check_tickers(sub, th, phase))
        s_alerts.extend(universe_monitor.check_sectors(sub, th, phase))

    if not s_alerts:
        log.info("No universe/sector signal alerts")
        return 0

    if ALERTS_FILE.exists():
        data = _load(ALERTS_FILE) or {}
        data.setdefault("alerts", []).extend(s_alerts)
        data["count"] = len(data["alerts"])
    else:
        data = {"alerts": s_alerts, "count": len(s_alerts),
                "generated_at": datetime.now().isoformat(timespec="seconds")}
    ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    log.info("Signal alerts merged: %d", len(s_alerts))
    return len(s_alerts)


def detect_and_merge_regime_alert() -> dict[str, Any]:
    """regime 라벨링 + 전환 시 alert merge. 반환: regime dict."""
    snapshot = _load(LATEST_FILE) or {}
    if not snapshot:
        log.warning("no latest snapshot — skip regime detection")
        return {}
    db_path = STATE_DIR / "timeseries.db"
    state_file = STATE_DIR / "last_regime.json"
    out = regime.detect_regime(snapshot, db_path, state_file=state_file)
    log.info("Regime: %s (score=%+.0f, transition=%s)",
             out["label"], out["score"], out["transition"])
    if out.get("alert"):
        if ALERTS_FILE.exists():
            data = _load(ALERTS_FILE) or {}
            data.setdefault("alerts", []).append(out["alert"])
            data["count"] = len(data["alerts"])
        else:
            data = {
                "alerts": [out["alert"]],
                "count": 1,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ALERTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        log.info("Regime transition alert merged")
    return out


def run_jarvis() -> Path:
    BRIEFING_HISTORY.mkdir(parents=True, exist_ok=True)
    log.info("=== Jarvis Orchestrator 시작 ===")

    run_pulse()
    run_compare()
    refresh_earnings_and_merge_alerts()
    merge_narrative_alerts()
    merge_predictive_alerts()
    merge_signal_alerts()
    detect_and_merge_regime_alert()
    run_weekly_attribution()
    build_geo_signal()
    build_context()

    briefing = compose_briefing()
    BRIEFING_FILE.write_text(briefing)
    ts = datetime.now(KST).strftime("%Y%m%d-%H%M")
    (BRIEFING_HISTORY / f"briefing-{ts}.md").write_text(briefing)

    log.info("Briefing 작성 완료 → %s", BRIEFING_FILE)
    return BRIEFING_FILE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    path = run_jarvis()
    print("\n" + "=" * 60)
    print(path.read_text())
