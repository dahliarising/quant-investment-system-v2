"""선행 경보 순수함수 코어 — 부수효과 없음, 전부 단위 테스트 가능.

상태: GREEN/AMBER/RED. 게이지: BUY/HOLD/REDUCE/SELL (사세요/멈추세요/줄이세요/파세요).
"""
from __future__ import annotations

GREEN, AMBER, RED = "green", "amber", "red"
BUY, HOLD, REDUCE, SELL = "BUY", "HOLD", "REDUCE", "SELL"

_SEVERITY = {GREEN: 0, AMBER: 1, RED: 2}

_MESSAGES = {
    "semis": "반도체 줄이세요 — 지수보다 먼저 빠짐",
    "vix_term": "헤지/현금 — 단기 스트레스(VIX 역전)",
    "breadth": "방어 — 시장폭 악화(상승종목 감소)",
    "hy": "위험자산 줄이세요 — 신용 경고(스프레드 확대)",
    "curve": "방어 전환 — 침체 신호(커브)",
}


# ── 밴드 분류기 (단순 임계) ───────────────────────────────────
def classify_vix_term(ratio: float, cfg: dict) -> str:
    c = cfg["vix_term"]
    if ratio <= c["green_max"]:
        return GREEN
    if ratio <= c["amber_max"]:
        return AMBER
    return RED


def classify_breadth(pct_above_ma200: float, cfg: dict) -> str:
    c = cfg["breadth"]
    if pct_above_ma200 >= c["green_min"]:
        return GREEN
    if pct_above_ma200 >= c["amber_min"]:
        return AMBER
    return RED


def classify_hy(oas: float, chg_5d: float, cfg: dict) -> str:
    c = cfg["hy"]
    if oas > c["amber_max"] or chg_5d >= c["rise_red_5d"]:
        return RED
    if oas > c["green_max"] or chg_5d >= c["rise_amber_5d"]:
        return AMBER
    return GREEN


def classify_curve(spread: float, chg_5d: float, cfg: dict) -> str:
    c = cfg["curve"]
    if spread < c["amber_min"] or abs(chg_5d) >= c["fast_move_5d"]:
        return RED
    if spread < c["green_min"]:
        return AMBER
    return GREEN


# ── 반도체 리더십 (다이버전스) ────────────────────────────────
def classify_semis(ratio: float, ratio_ma50: float, slope_5d: float,
                   spx_dist_from_high_pct: float, cfg: dict) -> str:
    """SOXX/SPY 상대강도. 지수 고점 근처서 반도체가 먼저 깨지면(다이버전스) RED."""
    near_high = spx_dist_from_high_pct >= -cfg["semis"]["divergence_high_dist_pct"]
    below_ma = ratio < ratio_ma50
    falling = slope_5d < 0
    if below_ma and falling and near_high:
        return RED            # 다이버전스: 지수 멀쩡한데 반도체만 깨짐
    if falling:
        return AMBER          # 롤오버 시작
    return GREEN


# ── 종합 게이지 + 행동 지시 ───────────────────────────────────
def composite_gauge(states: dict) -> tuple[str, int]:
    """states: {key: green/amber/red}. 반환 (게이지, red개수)."""
    reds = sum(1 for s in states.values() if s == RED)
    if reds >= 3:
        gauge = SELL
    elif states.get("hy") == RED and states.get("breadth") == RED:
        gauge = SELL                      # 콤보: 신용+시장폭 동시 = 시스템 위험
    elif reds == 2 or states.get("semis") == RED:
        gauge = REDUCE
    elif reds == 1:
        gauge = HOLD
    else:
        gauge = BUY
    return gauge, reds


def action_label(gauge: str, held: bool) -> str:
    if gauge == SELL:
        return "🔴 파세요 (분할매도·현금확보)" if held else "🔴 진입 미루세요 (방어)"
    if gauge == REDUCE:
        return "🟠 줄이세요 (비중축소·헤지)" if held else "🟠 진입 미루세요"
    if gauge == HOLD:
        return "🟡 멈추세요 (신규매수 보류)"
    return "🟢 사세요 (비중확대·진입 OK)"


# ── 전환 감지 + 지표 메시지 ───────────────────────────────────
def detect_transitions(prev: dict, cur: dict) -> list[dict]:
    """악화(severity 증가) 전환만 반환. 유지/개선은 무시."""
    out = []
    for key, now in cur.items():
        before = prev.get(key, GREEN)
        if _SEVERITY[now] > _SEVERITY[before]:
            out.append({"key": key, "from": before, "to": now})
    return out


def indicator_message(key: str, state: str) -> str:
    return _MESSAGES.get(key, key)


# ── -8% 하드스톱 ──────────────────────────────────────────────
def hard_stop(positions: list[dict], threshold: float) -> list[dict]:
    """보유종목 live pnl ≤ threshold(예: -8) → '파세요 {sym}'. pnl None은 무시."""
    out = []
    for p in positions:
        pnl = p.get("pnl_pct")
        if pnl is not None and pnl <= threshold:
            out.append({"sym": p["sym"], "pnl_pct": pnl,
                        "message": f"파세요 {p['sym']} — 손절선 {threshold:.0f}% 돌파 ({pnl:.1f}%)"})
    return out
