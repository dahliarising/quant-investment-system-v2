# tests/test_prediction_orchestrator.py
from corvin_jarvis.prediction import run_prediction_digest as orch


def test_build_digest_isolates_module_errors(monkeypatch):
    # 한 모듈이 예외를 던져도 전체 다이제스트는 생성된다
    def boom(*a, **k):
        raise RuntimeError("module down")
    monkeypatch.setattr(orch, "_run_velocity", boom)
    text = orch.build_digest(date_str="2026-06-15",
                             holdings=[{"symbol": "META", "price": 100}],
                             universe=[], db_path=None, geo_payload=None,
                             stops={}, closes_by_sym={}, daily_by_feature={})
    assert "2026-06-15" in text     # 생성 성공
    assert isinstance(text, str)


def test_dry_run_does_not_send(monkeypatch, capsys):
    sent = {"called": False}
    monkeypatch.setattr(orch, "_send", lambda body: sent.__setitem__("called", True))
    orch.main(["--dry-run"])
    assert sent["called"] is False   # dry-run은 전송 안 함


def test_build_results_by_model_runs_gated_systems():
    # respect_gate=False면 게이트 탈락 모델도 산출(모델별 뷰용)
    final, syms = orch.build_results(
        holdings=[], universe=[], geo_payload=None, stops={},
        closes_by_sym={}, daily_by_feature={}, gate={}, respect_gate=False)
    systems = {r.system for r in final}
    assert "logistic" in systems   # 게이트 무관 산출
    assert "band" in systems


def test_by_model_does_not_send(monkeypatch):
    sent = {"called": False}
    monkeypatch.setattr(orch, "_send", lambda body: sent.__setitem__("called", True))
    orch.main(["--by-model"])
    assert sent["called"] is False   # by-model도 stdout만
