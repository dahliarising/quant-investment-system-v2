from corvin_jarvis.brief.evidence import build_evidence


def test_validated_sample():
    calib = {"leading": {"sector": {"n": 12, "hit_rate": 0.58,
             "avg_confidence": 60, "calibrated_confidence": 59}}}
    ev = build_evidence(calib, engine="leading", kind="sector")
    assert ev.validation == "n=12·적중률 58%"


def test_insufficient_sample_is_honest():
    calib = {"predictive": {"EVENT": {"n": 1, "hit_rate": 0.0,
             "avg_confidence": 90, "calibrated_confidence": None}}}
    ev = build_evidence(calib, engine="predictive", kind="EVENT")
    assert ev.validation == "검증부족(n=1)"


def test_missing_entry_is_honest():
    ev = build_evidence({}, engine="x", kind="y")
    assert ev.validation == "검증부족"
    assert ev.edge_summary is None
