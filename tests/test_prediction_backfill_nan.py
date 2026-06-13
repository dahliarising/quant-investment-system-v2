# tests/test_prediction_backfill_nan.py
"""NaN-safe row 변환 회귀 — 지수/FX/선물(vix·usd_krw·gold·copper·dxy)의
Volume=NaN 행이 int(NaN) ValueError로 일봉 누락되던 버그 방지."""
from corvin_jarvis.prediction import backfill


def test_row_from_nan_volume_becomes_zero():
    row = backfill._row_from("vix", "2026-06-13",
                             open=15.0, high=16.0, low=14.0, close=15.5,
                             volume=float("nan"), source="fdr")
    assert row["volume"] == 0
    assert isinstance(row["volume"], int)
    assert row["close"] == 15.5


def test_row_from_nan_price_becomes_zero():
    row = backfill._row_from("gold", "2026-06-13",
                             open=float("nan"), high=float("nan"),
                             low=float("nan"), close=float("nan"),
                             volume=float("nan"), source="fdr")
    assert row["open"] == 0.0 and row["close"] == 0.0
    assert row["volume"] == 0


def test_row_from_none_becomes_zero():
    row = backfill._row_from("dxy", "2026-06-13",
                             open=None, high=None, low=None, close=None,
                             volume=None, source="fdr")
    assert row["open"] == 0.0
    assert row["volume"] == 0


def test_row_from_valid_inputs_unchanged():
    row = backfill._row_from("kospi", "2026-06-13",
                             open=2700.0, high=2720.5, low=2690.0,
                             close=2710.25, volume=123456, source="fdr")
    assert row["open"] == 2700.0
    assert row["close"] == 2710.25
    assert row["volume"] == 123456
    assert isinstance(row["volume"], int)


def test_nan_volume_row_upserts_and_reads_back_as_zero(tmp_path):
    db = tmp_path / "daily.db"
    backfill.init_db(db)
    row = backfill._row_from("vix", "2026-06-13",
                             open=15.0, high=16.0, low=14.0, close=15.5,
                             volume=float("nan"), source="fdr")
    backfill.upsert_rows(db, [row])
    got = backfill.read_daily(db, "vix", lookback=10)
    assert len(got) == 1
    assert got[0]["volume"] == 0
    assert got[0]["close"] == 15.5
