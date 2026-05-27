"""Corvin Jarvis — 통합 시세 어댑터 (Tier 2.6)

모든 주식/지수/원자재/FX/펀더멘털 데이터의 *single source of truth*.

데이터 소스 라우팅:
    KR 주식 (6자리):   KIS primary → pykrx fallback
    US 주식 (알파벳):  KIS primary → yfinance fallback
    KR 지수 (KS11):    FDR primary (KIS 국내지수 API는 별도, 일단 유지)
    US 지수 (^GSPC):   yfinance only (KIS overseas-index 미사용)
    원자재 (=F):       yfinance only
    FX (=X):           yfinance only
    Fundamentals KR:   pykrx
    Fundamentals US:   yfinance

설계 원칙:
- 호출자는 *심볼만* 넘기면 됨. 내부 라우팅은 quote_provider가 결정.
- KIS env 미설정 시 자동으로 fallback (silent, single warning log).
- 모든 함수 반환 = Quote dataclass (price + pct_change + source + error).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
_NY_TZ = ZoneInfo("America/New_York")
# Regular-session close times (local market tz). A daily bar dated "today" is only
# complete after its market closes; before that it is a still-forming intraday bar.
_MARKET_CLOSE = {"KR": (15, 30), "US": (16, 0)}
log = logging.getLogger("corvin.quote_provider")

BASE_DIR = Path(__file__).resolve().parent
_UNIVERSE_FILE = BASE_DIR / "universe.json"

# scripts/kr_data.py를 fallback으로 사용하려면 sys.path에 추가 필요.
import sys as _sys  # noqa: E402
_SCRIPTS_DIR = BASE_DIR.parent / "scripts"
if _SCRIPTS_DIR.exists() and str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))


@dataclass(frozen=True)
class Quote:
    price: float | None
    pct_change: float | None  # daily % vs 전일종가
    source: str  # kis_live | kis_daily | yfinance | pykrx | fdr | error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "pct_change": self.pct_change,
            "source": self.source,
            "error": self.error,
        }


# ============================================================
# 심볼 분류
# ============================================================


def is_kr_stock(symbol: str) -> bool:
    """6자리 숫자 = KR 종목코드."""
    return symbol.isdigit() and len(symbol) == 6


def is_us_index(symbol: str) -> bool:
    """^GSPC / ^IXIC / ^DJI / ^VIX / ^TNX 등."""
    return symbol.startswith("^")


def is_kr_index(symbol: str) -> bool:
    """KS11 (KOSPI) / KQ11 (KOSDAQ)."""
    return symbol.upper() in ("KS11", "KQ11")


def is_commodity(symbol: str) -> bool:
    """BZ=F GC=F CL=F SI=F HG=F 등."""
    return symbol.endswith("=F")


def is_fx(symbol: str) -> bool:
    """KRW=X EURUSD=X DX-Y.NYB 등."""
    return symbol.endswith("=X") or symbol == "DX-Y.NYB"


def is_crypto(symbol: str) -> bool:
    """BTC-USD / ETH-USD 등 yfinance crypto suffix."""
    return symbol.upper().endswith("-USD") or symbol.upper().endswith("-USDT")


# ============================================================
# KIS env helpers (lazy + cached)
# ============================================================


_kis_env_cache: Any = None
_kis_env_attempted: bool = False
_universe_exchange_map_cache: dict[str, str] | None = None


def _get_kis_env() -> Any:
    """KIS env lazy load. 실패 시 None 캐싱."""
    global _kis_env_cache, _kis_env_attempted
    if _kis_env_attempted:
        return _kis_env_cache
    _kis_env_attempted = True
    try:
        from corvin_jarvis import kis_auth
        _kis_env_cache = kis_auth.load_env()
        log.info("KIS env loaded — env=%s", _kis_env_cache.env)
    except Exception as e:  # noqa: BLE001
        log.info("KIS unavailable (%s) — using yfinance/pykrx fallback", e)
        _kis_env_cache = None
    return _kis_env_cache


def _exchange_for(symbol: str) -> str | None:
    """universe.json에서 US 종목의 거래소 추출. 캐시됨."""
    global _universe_exchange_map_cache
    if _universe_exchange_map_cache is None:
        _universe_exchange_map_cache = {}
        if _UNIVERSE_FILE.exists():
            try:
                import json
                data = json.loads(_UNIVERSE_FILE.read_text())
                for u in data.get("tickers", []):
                    if "exchange" in u and "symbol" in u:
                        _universe_exchange_map_cache[u["symbol"]] = u["exchange"]
            except (OSError, ValueError):
                pass
    return _universe_exchange_map_cache.get(symbol)


def reset_caches() -> None:
    """테스트용 — 캐시 초기화."""
    global _kis_env_cache, _kis_env_attempted, _universe_exchange_map_cache
    _kis_env_cache = None
    _kis_env_attempted = False
    _universe_exchange_map_cache = None


# ============================================================
# 주식 시세 (현재가 + 일봉)
# ============================================================


def get_stock_quote(symbol: str) -> Quote:
    """주식 현재가 + 전일대비%. KIS 우선 fallback to yfinance/pykrx."""
    env = _get_kis_env()
    if is_kr_stock(symbol):
        return _kr_stock_quote(symbol, env)
    return _us_stock_quote(symbol, env)


def _kr_stock_quote(symbol: str, env: Any) -> Quote:
    if env is not None:
        try:
            from corvin_jarvis import kis_quote
            price, pct = kis_quote.get_kr_quote(symbol, env=env)
            if price is not None:
                return Quote(price=price, pct_change=pct, source="kis_live")
        except Exception as e:  # noqa: BLE001
            log.warning("KIS KR quote failed for %s: %s — pykrx fallback", symbol, e)
    # pykrx fallback (legacy scripts/kr_data.py)
    try:
        from kr_data import get_kr_stock_data  # noqa: PLC0415
        data = get_kr_stock_data(symbol)
        if "에러" in data:
            return Quote(price=None, pct_change=None, source="pykrx", error=data["에러"])
        return Quote(
            price=float(data["현재가"]),
            pct_change=float(data.get("등락률(%)", 0) or 0),
            source="pykrx",
        )
    except Exception as e:  # noqa: BLE001
        return Quote(price=None, pct_change=None, source="pykrx", error=str(e))


def _us_stock_quote(symbol: str, env: Any) -> Quote:
    if env is not None:
        try:
            from corvin_jarvis import kis_quote
            price, pct = kis_quote.get_us_quote(
                symbol, exchange=_exchange_for(symbol), env=env,
            )
            if price is not None:
                return Quote(price=price, pct_change=pct, source="kis_live")
        except Exception as e:  # noqa: BLE001
            log.warning("KIS US quote failed for %s: %s — yfinance fallback", symbol, e)
    return _yfinance_quote(symbol)


def _yfinance_quote(symbol: str) -> Quote:
    try:
        import yfinance as yf  # noqa: PLC0415
        hist = yf.Ticker(symbol).history(period="5d")
        if hist.empty:
            return Quote(price=None, pct_change=None, source="yfinance", error="no data")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last
        pct = (last / prev - 1) * 100 if prev else 0.0
        return Quote(
            price=round(last, 4),
            pct_change=round(pct, 2),
            source="yfinance",
        )
    except Exception as e:  # noqa: BLE001
        return Quote(price=None, pct_change=None, source="yfinance", error=str(e))


# ============================================================
# 일봉 (close 시계열)
# ============================================================


def _last_bar_incomplete(symbol: str, last_bar_date: date, now: datetime | None = None) -> bool:
    """True if last_bar_date is the *current* session and that session is still open.

    Daily technical signals (RSI/MA) must be computed on completed bars; scoring a
    half-formed intraday bar produces unstable, non-reproducible signals.
    """
    now = now or datetime.now(KST)
    if is_kr_stock(symbol):
        market_now = now.astimezone(KST)
        close_h, close_m = _MARKET_CLOSE["KR"]
    else:
        market_now = now.astimezone(_NY_TZ)
        close_h, close_m = _MARKET_CLOSE["US"]
    if last_bar_date != market_now.date():
        return False
    return (market_now.hour, market_now.minute) < (close_h, close_m)


def get_stock_daily_closes(symbol: str, days: int = 252, *, completed_only: bool = False) -> list[float]:
    """주식 일봉 종가 oldest→newest. KIS 우선 fallback to yf/pykrx.

    completed_only=True: 진행 중인(미완성) 오늘 봉을 제외해 완성봉만 반환.
    """
    env = _get_kis_env()
    if env is not None:
        try:
            from corvin_jarvis import kis_quote
            if is_kr_stock(symbol):
                closes = kis_quote.get_kr_daily_closes(symbol, days=days, env=env)
            else:
                closes = kis_quote.get_us_daily_closes(
                    symbol, exchange=_exchange_for(symbol), days=days, env=env,
                )
            if closes:
                return closes  # NOTE: KIS path returns no dates → completed_only는 fallback에서만 적용
        except Exception as e:  # noqa: BLE001
            log.warning("KIS daily closes failed for %s: %s — fallback", symbol, e)

    # fallback
    if is_kr_stock(symbol):
        try:
            from pykrx import stock  # noqa: PLC0415
            end = datetime.now(KST).strftime("%Y%m%d")
            start = (datetime.now(KST) - timedelta(days=days + 80)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(start, end, symbol)
            if df.empty:
                return []
            if completed_only and len(df) and _last_bar_incomplete(symbol, df.index[-1].date()):
                df = df.iloc[:-1]
            return [float(c) for c in df["종가"].tail(days).tolist() if c == c and c > 0]
        except Exception as e:  # noqa: BLE001
            log.warning("pykrx fallback failed for %s: %s", symbol, e)
            return []

    try:
        import yfinance as yf  # noqa: PLC0415
        period = "2y" if days > 252 else "1y"
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if hist.empty:
            return []
        if completed_only and len(hist):
            idx = hist.index[-1]
            last_d = idx.date() if hasattr(idx, "date") else idx
            if _last_bar_incomplete(symbol, last_d):
                hist = hist.iloc[:-1]
        return [float(p) for p in hist["Close"].tail(days).tolist() if p == p and p > 0]
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance fallback failed for %s: %s", symbol, e)
        return []


# ============================================================
# 지수 / 원자재 / FX (yfinance/FDR — KIS 미사용 영역)
# ============================================================


def get_kr_index_quote(code: str) -> Quote:
    """KS11 (KOSPI) / KQ11 (KOSDAQ) — FinanceDataReader."""
    try:
        from kr_data import get_kr_index_data  # noqa: PLC0415
        data = get_kr_index_data(code)
        if "에러" in data:
            return Quote(price=None, pct_change=None, source="fdr", error=data["에러"])
        return Quote(
            price=float(data["현재가"]),
            pct_change=float(data.get("전일대비(%)", 0) or 0),
            source="fdr",
        )
    except Exception as e:  # noqa: BLE001
        return Quote(price=None, pct_change=None, source="fdr", error=str(e))


def get_us_index_quote(symbol: str) -> Quote:
    """미국 지수 — yfinance."""
    return _yfinance_quote(symbol)


def get_commodity_quote(symbol: str) -> Quote:
    """원자재 — yfinance."""
    return _yfinance_quote(symbol)


def get_fx_quote(symbol: str) -> Quote:
    """FX — yfinance."""
    return _yfinance_quote(symbol)


# ============================================================
# Generic dispatcher (심볼 자체로 라우팅 가능)
# ============================================================


def get_crypto_quote(symbol: str) -> Quote:
    """Crypto — yfinance only (KIS 미지원)."""
    return _yfinance_quote(symbol)


def get_quote(symbol: str) -> Quote:
    """심볼 형식만으로 적절한 source 자동 선택."""
    if is_kr_index(symbol):
        return get_kr_index_quote(symbol)
    if is_us_index(symbol):
        return get_us_index_quote(symbol)
    if is_commodity(symbol):
        return get_commodity_quote(symbol)
    if is_fx(symbol):
        return get_fx_quote(symbol)
    if is_crypto(symbol):
        return get_crypto_quote(symbol)
    return get_stock_quote(symbol)


# ============================================================
# Fundamentals (KIS 부분 지원, but yfinance/pykrx가 풍부 → 유지)
# ============================================================


def get_stock_fundamentals(symbol: str) -> dict[str, Any]:
    """PER/PBR/시총 등. KR=pykrx, US=yfinance."""
    if is_kr_stock(symbol):
        try:
            from kr_data import get_kr_stock_data  # noqa: PLC0415
            return get_kr_stock_data(symbol)
        except Exception as e:  # noqa: BLE001
            return {"에러": str(e), "_source": "pykrx"}

    try:
        import yfinance as yf  # noqa: PLC0415
        info = yf.Ticker(symbol).info or {}
        return {
            "symbol": symbol,
            "PER": info.get("trailingPE"),
            "PBR": info.get("priceToBook"),
            "시가총액": info.get("marketCap"),
            "배당수익률": info.get("dividendYield"),
            "_source": "yfinance",
        }
    except Exception as e:  # noqa: BLE001
        return {"에러": str(e), "_source": "yfinance"}
