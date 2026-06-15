"""Corvin 신호→주문 dry-run — on-demand 진입점 (Phase ③ 실행화).

실제 verdicts.json + portfolio 보유 + 라이브 현재가로 "오늘 뭘 주문할지" 산출.
기본 dry-run(주문 0발). --place 주면 order_guard 통해 모의주문(실전은 이중확인 필요).

실행:
    python3 -m corvin_jarvis.run_signal_dryrun            # dry-run (안전)
    python3 -m corvin_jarvis.run_signal_dryrun --place    # 모의주문 실행
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from corvin_jarvis import kis_auth, kis_order, quote_provider, signal_router

# ⚠️ 현재 주문 모듈은 국내(place_kr_order)만. 해외주식은 KRW 트랜치/주문 TR이 달라
#    별도 모듈(④) 필요 → dry-run에서 KR만 라우팅, US/기타는 '보류'로 분리.

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("corvin.run_signal_dryrun")

BASE_DIR = Path(__file__).resolve().parent
PORTFOLIO_FILE = BASE_DIR.parent / "portfolio.json"
VERDICTS_FILE = BASE_DIR / "state" / "verdicts.json"

SKIP_ACTIONS = frozenset({"홀딩", "관망"})


def holdings_from_portfolio(pf: dict[str, Any]) -> dict[str, int]:
    """portfolio dict → {symbol: shares} (순수)."""
    return {
        h["symbol"]: int(h.get("shares", 0))
        for h in pf.get("holdings", [])
        if h.get("symbol")
    }


def actionable(verdicts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """홀딩·관망 제외 — 주문 후보 verdict만 (순수)."""
    return {s: v for s, v in verdicts.items() if v.get("action") not in SKIP_ACTIONS}


def split_by_market(
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """후보를 (국내=라우팅 가능, 해외=보류)로 분리 (순수).

    현재 주문 모듈은 국내(place_kr_order)만 — 해외는 ④에서 별도 구현 전까지 보류.
    """
    kr = {s: v for s, v in candidates.items() if quote_provider.is_kr_stock(s)}
    us = {s: v for s, v in candidates.items() if not quote_provider.is_kr_stock(s)}
    return kr, us


def format_plans(plans: list[signal_router.RoutedPlan]) -> str:
    """RoutedPlan 목록 → 스캔 가능한 텍스트 (순수)."""
    if not plans:
        return "주문 후보 없음 — 현재 전부 홀딩/관망이거나 사이징 불가."
    icon = {"dry_run": "🔬", "placed": "✅", "duplicate": "♻️",
            "blocked": "🛑", "rejected": "⚠️", "error": "❌"}
    lines = []
    for p in plans:
        it = p.intent
        sidot = "➖" if it.side == "sell" else "➕"
        tail = f" — {p.detail}" if p.detail else ""
        lines.append(
            f"{icon.get(p.status, '•')} [{p.status}] {sidot} {it.action} "
            f"{it.symbol} {it.side} {it.qty}주 @{it.price:,.0f}{tail}"
        )
    return "\n".join(lines)


def holdings_from_balance(bal: "kis_order.Balance") -> dict[str, int]:
    """모의계좌 잔고 → {종목: 보유수량} (순수). --place 사이징은 모의계좌 실보유 기준."""
    out: dict[str, int] = {}
    for h in bal.holdings:
        sym = h.get("pdno")
        if sym:
            try:
                out[sym] = int(h.get("hldg_qty", 0))
            except (TypeError, ValueError):
                out[sym] = 0
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Corvin 신호→주문 dry-run")
    parser.add_argument("--place", action="store_true",
                        help="모의주문 실제 실행 (기본=dry-run)")
    args = parser.parse_args()

    verdicts = _load_json(VERDICTS_FILE)
    kr, us = split_by_market(actionable(verdicts))

    if not kr and not us:
        print("🔬 신호→주문 dry-run\n주문 후보 없음 — 현재 전부 홀딩/관망.")
        return 0

    # --place: 모의계좌 env + *모의계좌 실보유* 기준 사이징 (자기일관).
    # dry-run: 실제 portfolio.json 보유 기준 (실보유 분석).
    env = None
    if args.place:
        env = kis_auth.load_mock_env()
        holdings = holdings_from_balance(kis_order.get_kr_balance(env=env))
    else:
        holdings = holdings_from_portfolio(_load_json(PORTFOLIO_FILE))

    # 라이브 현재가 (국내 후보만 — 라우팅 대상)
    prices: dict[str, float] = {}
    for sym in kr:
        q = quote_provider.get_stock_quote(sym)
        if q.price:
            prices[sym] = float(q.price)
        else:
            log.warning("가격 조회 실패 %s (%s) — 제외", sym, q.source)

    plans = signal_router.route_verdicts(
        kr, holdings=holdings, prices=prices, env=env, place=args.place,
    )

    mode = "모의주문 실행" if args.place else "DRY-RUN (주문 0발)"
    print(f"🔬 신호→주문 {mode} — 국내(KR)만")
    print("━" * 12)
    print(format_plans(plans))
    if us:
        labels = ", ".join(f"{s}({v.get('action')})" for s, v in us.items())
        print("\n⏸ 해외(US) 보류 — 해외주문 모듈(④) 미구현:")
        print(f"  {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
