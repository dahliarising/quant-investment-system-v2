"""Phase 0 — 데이터 이중·삼중 교차검증 레이어 테스트.

cross_check 순수함수: 여러 소스 값 → 일치판정·신뢰도·flag. 네트워크 없음.
"""
from corvin_jarvis import data_verify as dv


def test_two_sources_agree_within_tolerance():
    # 두 소스가 허용오차(1%) 내 → 일치, 중앙값, 신뢰 high, flag 없음
    r = dv.cross_check("MSFT", {"yfinance": 416.67, "kis": 417.00}, tol_pct=1.0)
    assert r["agree"] is True
    assert r["confidence"] == "high"
    assert r["flag"] is None
    assert abs(r["value"] - 416.835) < 0.01      # median of two
    assert set(r["sources"]) == {"yfinance", "kis"}


def test_two_sources_disagree_flags_discrepancy():
    # 두 소스 괴리 5% > tol 1% → 불일치, 보수값(중앙값) 유지, 신뢰 low, flag
    r = dv.cross_check("XYZ", {"a": 100.0, "b": 108.0}, tol_pct=1.0)
    assert r["agree"] is False
    assert r["confidence"] == "low"
    assert r["flag"] == "discrepancy"
    assert r["value"] == 104.0                    # median (conservative)
    assert r["spread_pct"] > 1.0


def test_single_source_medium_confidence():
    # 한 소스만 있으면 그 값 쓰되 신뢰 medium + single_source flag
    r = dv.cross_check("KO", {"yfinance": 60.0, "kis": None}, tol_pct=1.0)
    assert r["value"] == 60.0
    assert r["confidence"] == "medium"
    assert r["flag"] == "single_source"
    assert r["sources"] == ["yfinance"]


def test_no_sources_returns_none_missing():
    # 모든 소스 결측 → None, 신뢰 none, missing flag (추측 금지)
    r = dv.cross_check("ZZZ", {"a": None, "b": None}, tol_pct=1.0)
    assert r["value"] is None
    assert r["confidence"] == "none"
    assert r["flag"] == "missing"
    assert r["sources"] == []


def test_three_sources_majority_median():
    # 세 소스 중 하나가 outlier여도 중앙값은 견고 (이중·삼중의 핵심)
    r = dv.cross_check("NVDA", {"a": 205.0, "b": 205.2, "c": 250.0}, tol_pct=1.0)
    assert r["value"] == 205.2                     # median resists outlier
    assert r["agree"] is False                     # spread(min..max) > tol
    assert r["flag"] == "discrepancy"


# ── reconcile_pnl: 저장 vs 라이브 시점 정합 (TSLA 버그 가드) ──────────
def test_reconcile_flags_stale_stored_pnl():
    # 저장 -3.54% vs 라이브 -10.53% → 괴리 큼 → stale 경고, 라이브 우선
    r = dv.reconcile_pnl(stored=-3.54, live=-10.53, tol_pct=3.0)
    assert r["value"] == -10.53                    # live 우선
    assert r["stale"] is True
    assert r["flag"] == "stored_stale"
    assert r["divergence_pp"] > 3.0


def test_reconcile_ok_when_close():
    r = dv.reconcile_pnl(stored=5.0, live=5.2, tol_pct=3.0)
    assert r["value"] == 5.2
    assert r["stale"] is False
    assert r["flag"] is None


def test_reconcile_uses_live_when_stored_missing():
    r = dv.reconcile_pnl(stored=None, live=-10.5, tol_pct=3.0)
    assert r["value"] == -10.5
    assert r["stale"] is False


# ── sanity: NaN/inf/음수가격 가드 ──────────────────────────────────
def test_sanity_rejects_nan_inf_none():
    assert dv.sanity(float("nan")) is False
    assert dv.sanity(float("inf")) is False
    assert dv.sanity(None) is False
    assert dv.sanity(100.0) is True


def test_sanity_positive_rejects_nonpositive_price():
    assert dv.sanity(-5.0, positive=True) is False
    assert dv.sanity(0.0, positive=True) is False
    assert dv.sanity(123.4, positive=True) is True
    assert dv.sanity(-3.5, positive=False) is True   # pnl can be negative


# ── verified: DI 다소스 묶음 (fetcher 예외/결측은 None 처리) ────────
def test_verified_runs_fetchers_and_cross_checks():
    def raises():
        raise RuntimeError("network down")
    fetchers = {
        "yfinance": lambda: 205.10,
        "kis": lambda: 205.30,
        "broken": raises,            # 예외 → None 처리(스킵), 분석 안 깨짐
    }
    r = dv.verified("NVDA", fetchers, tol_pct=1.0, positive=True)
    assert r["confidence"] == "high"             # 두 정상 소스 일치
    assert set(r["sources"]) == {"yfinance", "kis"}
    assert "broken" not in r["sources"]


def test_verified_drops_insane_values():
    fetchers = {"a": lambda: float("nan"), "b": lambda: -7.0, "c": lambda: 100.0}
    r = dv.verified("X", fetchers, tol_pct=1.0, positive=True)
    # nan·음수가격 제거 → 단일 정상소스만 → single_source
    assert r["value"] == 100.0
    assert r["flag"] == "single_source"
