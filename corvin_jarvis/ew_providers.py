"""선행 지표 데이터 어댑터 — fetcher 콜백 DI. 데이터 누락 시 None(스킵, 추측 금지).

순수 reading 함수(테스트 가능) + 라이브 IO 바인딩(yahoo/FRED) 분리.
"""
from __future__ import annotations

from typing import Callable


def _ma(vals: list[float], n: int) -> float | None:
    return sum(vals[-n:]) / n if len(vals) >= n else None


# ── 순수 reading 함수 (DI) ────────────────────────────────────
def semis_reading(closes_fetcher: Callable[[str, int], list[float]],
                  spx_high_fetcher: Callable[[], float | None]) -> dict | None:
    soxx = closes_fetcher("SOXX", 60)
    spy = closes_fetcher("SPY", 60)
    if len(soxx) < 55 or len(spy) < 55:
        return None
    ratio = [s / p for s, p in zip(soxx, spy)]
    ma50 = _ma(ratio, 50)
    if ma50 is None:
        return None
    slope_5d = ratio[-1] - ratio[-6]
    spx_high = spx_high_fetcher()
    if not spx_high:
        return None
    spx_dist = (spy[-1] / spx_high - 1) * 100
    return {"ratio": ratio[-1], "ratio_ma50": ma50, "slope_5d": slope_5d,
            "spx_dist_from_high_pct": spx_dist}


def vix_term_reading(quote_fetcher: Callable[[str], float | None]) -> dict | None:
    vix = quote_fetcher("^VIX")
    vix3m = quote_fetcher("^VIX3M")
    if not vix or not vix3m:
        return None
    return {"ratio": vix / vix3m}


def breadth_reading(universe: list[str],
                    closes_fetcher: Callable[[str, int], list[float]]) -> dict | None:
    above = 0
    counted = 0
    for sym in universe:
        cl = closes_fetcher(sym, 200)
        if len(cl) < 200:
            continue
        counted += 1
        if cl[-1] > sum(cl[-200:]) / 200:
            above += 1
    if counted == 0:
        return None
    return {"pct_above_ma200": above / counted * 100}


def fred_reading(series_fetcher: Callable[[str, int], list[float]], code: str) -> dict | None:
    vals = series_fetcher(code, 8)
    if len(vals) < 6:
        return None
    return {"value": vals[-1], "chg_5d": vals[-1] - vals[-6]}


# ── 라이브 IO 바인딩 (네트워크 경계) ──────────────────────────
def live_closes_fetcher(sym: str, n: int) -> list[float]:
    """yahoo 일별 종가 n개. 실패 시 []."""
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="1y")["Close"].dropna().tolist()
        return h[-n:]
    except Exception:
        return []


def live_vix_fetcher(sym: str) -> float | None:
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) if len(h) else None
    except Exception:
        return None


def live_fred_fetcher(code: str, n: int) -> list[float]:
    """FRED 일별 시리즈 마지막 n개. code 예: 'BAMLH0A0HYM2','T10Y2Y'.

    fredgraph.csv를 직접 호출 — bounded timeout + User-Agent + 1회 재시도.
    (FDR는 timeout/UA 없이 requests.get → cron 행 위험 + 일부 시리즈 throttle.)
    실패 시 [] 반환 → 해당 지표 graceful skip.
    """
    import csv
    import io
    from datetime import datetime, timedelta

    import requests

    cosd = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={cosd}"
    headers = {"User-Agent": "Mozilla/5.0 (corvin-jarvis early-warning)"}
    for attempt in range(2):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200 or "content-disposition" not in r.headers:
                continue
            rows = list(csv.reader(io.StringIO(r.text)))
            vals = []
            for row in rows[1:]:  # skip header
                if len(row) < 2 or row[1] in ("", "."):
                    continue
                try:
                    vals.append(float(row[1]))
                except ValueError:
                    continue
            return vals[-n:]
        except requests.RequestException:
            continue
    return []
