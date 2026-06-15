"""Corvin 자동매매 페이퍼트레이더 — 장중 사이클 진입점 (시뮬레이터).

예측/verdict → 종목선정 → 가드(행동·리스크) → 가상체결 → 성과추적. 실주문 0.
KR+US 통합(KRW기준). 시장 세션 게이트로 장중에만 동작.

실행:
    python3 -m corvin_jarvis.run_auto_trader            # 현재 세션 자동감지
    python3 -m corvin_jarvis.run_auto_trader --observe  # 체결 미반영(관찰만)
    python3 -m corvin_jarvis.run_auto_trader --force KR # 세션 무시 강제

스케줄(수동 등록 — crontab 자동등록 금지 정책):
    */30 9-15 * * 1-5  → KR 세션
    30,0 23,0-5 * * *  → US 세션 (run_auto_trader.sh 래퍼 경유)
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from corvin_jarvis import auto_strategy as st
from corvin_jarvis import channels
from corvin_jarvis import paper_portfolio as pp
from corvin_jarvis import quote_provider as qp
from corvin_jarvis.signals import calibration as cal
from corvin_jarvis.signals import ledger as sl

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.auto_trader")

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = Path(__file__).resolve().parent
PORTFOLIO_STATE = BASE_DIR / "state" / "paper_portfolio.json"
VERDICTS_FILE = BASE_DIR / "state" / "verdicts.json"

# 시장 세션 (KST). US는 자정 넘김(22:30~05:00).
KR_OPEN, KR_CLOSE = time(9, 0), time(15, 30)
US_OPEN, US_CLOSE = time(22, 30), time(5, 0)

BUY_ACTIONS = frozenset({"매수", "분할매수", "비중확대"})
SELL_FULL = frozenset({"매도"})
SELL_PARTIAL = frozenset({"비중축소"})
PARTIAL_FRACTION = 1 / 3

# 기본 정책 (조정 가능)
BUY_KRW = 2_000_000
MAX_POSITION_PCT = 25.0
FX_FALLBACK = 1510.0


def current_session(now: datetime) -> str | None:
    """KST now → 'KR' | 'US' | None(휴장)."""
    t = now.time()
    if KR_OPEN <= t <= KR_CLOSE:
        return "KR"
    if t >= US_OPEN or t <= US_CLOSE:  # 자정 넘김
        return "US"
    return None


def signals_from_verdicts(
    verdicts: dict[str, dict[str, Any]], pf: pp.PaperPortfolio,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """verdict → (매수신호, 매도신호). 매도는 페이퍼 보유분에만."""
    buys: list[dict[str, Any]] = []
    sells: list[dict[str, Any]] = []
    for sym, v in verdicts.items():
        action = v.get("action", "")
        if action in BUY_ACTIONS:
            buys.append({"symbol": sym, "action": action})
        elif action in SELL_FULL or action in SELL_PARTIAL:
            held = pf.holdings.get(sym)
            if not held:
                continue
            qty = held.qty if action in SELL_FULL else int(held.qty * PARTIAL_FRACTION)
            if qty >= 1:
                sells.append({"symbol": sym, "qty": qty, "reason": action})
    return buys, sells


def _rsi(p: list[float], n: int = 14) -> float | None:
    if len(p) < n + 1:
        return None
    g = [max(p[i] - p[i - 1], 0) for i in range(1, len(p))]
    l = [max(p[i - 1] - p[i], 0) for i in range(1, len(p))]
    ag, al = statistics.mean(g[-n:]), statistics.mean(l[-n:])
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def chase_metrics_for(symbol: str) -> st.ChaseMetrics:
    """라이브 일봉 → 추격 판정 지표 (MA20대비·RSI·52주고대비)."""
    c = qp.get_stock_daily_closes(symbol, days=260)
    if len(c) < 20:
        return st.ChaseMetrics(None, None, None)
    price, ma20, hi = c[-1], statistics.mean(c[-20:]), max(c[-252:])
    return st.ChaseMetrics(
        (price / ma20 - 1) * 100 if ma20 else None,
        _rsi(c),
        (price / hi - 1) * 100 if hi else None,
    )


def _fx() -> float:
    """USD/KRW — FDR 사용 (yfinance는 KR 관련 stale/flaky). 실패 시 fallback."""
    try:
        import FinanceDataReader as fdr
        start = (datetime.now(KST).date().toordinal() - 10)
        from datetime import date
        df = fdr.DataReader("USD/KRW", date.fromordinal(start).isoformat())
        return float(df["Close"].iloc[-1])
    except Exception as e:  # noqa: BLE001 — FX 실패는 fallback으로 흡수
        log.warning("FX 조회 실패 (%s) — fallback %.0f", e, FX_FALLBACK)
        return FX_FALLBACK


def _calibration_context() -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    """캘리브레이션 맵 + 심볼→(engine,kind) (열린 신호 기준). 실패 시 빈 값(게이트 무력)."""
    try:
        calibrations = cal.compute()
    except Exception as e:  # noqa: BLE001
        log.warning("calibration 로드 실패 (%s) — 게이트 중립", e)
        return {}, {}
    sym_key: dict[str, tuple[str, str]] = {}
    try:
        for s in sl.fetch_open():
            sym = s.get("symbol")
            if sym and sym not in sym_key:
                sym_key[sym] = (s.get("engine", ""), s.get("kind", ""))
    except Exception as e:  # noqa: BLE001
        log.warning("ledger 조회 실패 (%s)", e)
    return calibrations, sym_key


def run_cycle(*, observe: bool = False, session: str | None = "KR") -> dict[str, Any]:
    """한 사이클: 신호→가드→가상체결→저장→성과. 반환=요약 dict."""
    verdicts = json.loads(VERDICTS_FILE.read_text())
    pf = pp.load(PORTFOLIO_STATE)
    buys, sells = signals_from_verdicts(verdicts, pf)

    # 캘리브레이션 게이트 컨텍스트 — 매수신호에 engine/kind 부착
    calibrations, sym_key = _calibration_context()
    for b in buys:
        key = sym_key.get(b["symbol"])
        if key:
            b["engine"], b["kind"] = key

    fx = _fx()
    cand = [b["symbol"] for b in buys] + [s["symbol"] for s in sells]
    prices: dict[str, float] = {}
    metrics: dict[str, st.ChaseMetrics] = {}
    for sym in cand:
        q = qp.get_stock_quote(sym)
        if not q.price:
            continue
        prices[sym] = float(q.price) * (1 if qp.is_kr_stock(sym) else fx)
    for b in buys:
        metrics[b["symbol"]] = chase_metrics_for(b["symbol"])

    plans = st.plan_cycle(
        pf, buy_signals=buys, sell_signals=sells, prices_krw=prices,
        chase_metrics=metrics, buy_krw=BUY_KRW, max_position_pct=MAX_POSITION_PCT,
        calibrations=calibrations,
    )

    executed = []
    ts = datetime.now(KST).isoformat(timespec="seconds")
    for p in plans:
        if p["status"] != "planned":
            continue
        if observe:
            executed.append({**p, "applied": False})
            continue
        try:
            if p["side"] == "buy":
                pf = pp.buy(pf, p["symbol"], p["qty"], p["price_krw"],
                           currency="KRW" if qp.is_kr_stock(p["symbol"]) else "USD",
                           ts=ts, reason=p["reason"])
            else:
                pf = pp.sell(pf, p["symbol"], p["qty"], p["price_krw"], ts=ts, reason=p["reason"])
            executed.append({**p, "applied": True})
        except ValueError as e:
            executed.append({**p, "applied": False, "error": str(e)})

    if not observe and executed:
        pp.save(pf, PORTFOLIO_STATE)

    snap = pp.mark_to_market(pf, prices)
    return {"session": session, "observe": observe, "plans": plans,
            "executed": executed, "snapshot": snap}


def format_report(result: dict[str, Any]) -> str:
    snap = result["snapshot"]
    mode = "관찰(미반영)" if result["observe"] else "체결반영"
    lines = [f"🤖 자동매매 시뮬 [{result['session']}·{mode}]", "━" * 12]
    acted = [e for e in result["executed"]]
    if acted:
        for e in acted:
            mark = "✅" if e.get("applied") else "👁"
            sidot = "➕" if e["side"] == "buy" else "➖"
            lines.append(f"  {mark}{sidot} {e['symbol']} {e['side']} {e['qty']}주 @{e['price_krw']:,.0f}")
    skipped = [p for p in result["plans"] if p["status"] == "skipped"]
    for s in skipped:
        lines.append(f"  🚫 {s['symbol']} 스킵 — {s['skip_reason']}")
    if not acted and not skipped:
        lines.append("  신호 없음 (홀딩/관망)")
    lines.append("━" * 12)
    lines.append(f"💰 가상자산 ₩{snap['total_value_krw']:,.0f} "
                 f"(현금 ₩{snap['cash_krw']:,.0f}) · "
                 f"P&L {snap['pnl_krw']:+,.0f}원 ({snap['pnl_pct']:+.2f}%)")
    return "\n".join(lines)


def notify_trades(result: dict[str, Any], *, sender: Any = None) -> bool:
    """실제 체결(applied=True)이 있을 때만 Telegram 알림. 0건/스킵엔 무알림(노이즈 방지)."""
    sender = sender or channels.send_telegram
    applied = [e for e in result.get("executed", []) if e.get("applied")]
    if not applied:
        return False
    return bool(sender(format_report(result)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Corvin 자동매매 페이퍼트레이더")
    ap.add_argument("--observe", action="store_true", help="체결 미반영(관찰만)")
    ap.add_argument("--force", metavar="SESSION", help="세션 무시 강제 (KR/US)")
    args = ap.parse_args()

    session = args.force or current_session(datetime.now(KST))
    if session is None:
        print("⏸ 휴장 — 자동매매 사이클 스킵")
        return 0
    result = run_cycle(observe=args.observe, session=session)
    print(format_report(result))
    if not args.observe:
        if notify_trades(result):
            log.info("거래 알림 전송됨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
