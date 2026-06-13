# Corvin 검증·토론 도구 — 실사용 가이드

2026-06-06 구축. 전부 **추가 과금 0원**(로컬 + claude CLI 구독 인증). 브랜치: `feat/corvin-early-warning`.

---

## 1. 실시간 투자 토론 (메인 기능)

투자방법별/스탠스별 에이전트가 보유 포트를 놓고 **실시간 카톡 대화**로 토론. 모든 발언은 데이터와 대조돼 검증 배지가 붙는다.

```bash
python3 -m corvin_jarvis.dashboard         # 포트 8765 (자동으로 브라우저 열림)
# 폰/테일넷 접속: CORVIN_DASHBOARD_HOST=0.0.0.0 python3 -m corvin_jarvis.dashboard
#                + tailscale serve --bg 8765
```
브라우저에서 **`http://127.0.0.1:8765/debate`** 접속 → 모드 고르고 **▶ 토론 시작**.

- **모드 2종**: `투자방법 5인`(가치·성장·추세·매크로·퀀트) / `손절·매수·홀딩`
- **검증 배지**: ✓ 검증(데이터 일치) · ⚠️ 미검증(컨텍스트에 없는 수치) · ✗ 불일치(틀린 인용)
- **데이터**: jarvis 스냅샷(KIS 라이브 가격) + 오늘 하락 원인규명 자동 포함
- **LLM**: claude CLI(폐하 구독) — 토론 1회 ~2~4분(턴이 하나씩 실시간 등장). $0.

> `?fake=1`을 붙이면 claude CLI 없이 즉시 캔드 응답으로 UI만 빠르게 확인(오프라인 데모).

---

## 2. 자동 분석 (jarvis 펄스)

cron이 돌리는 메인 파이프라인. 선행경보 + 원인규명이 자동 포함됨.

```bash
python3 corvin_jarvis/jarvis.py
```
- **선행경보(EW)**: 5지표(반도체·VIX기간·시장폭·HY·커브) 상태 악화 전환 + −8% 하드스톱 → `state/alerts.json`
- **원인규명**: 오늘 하락의 섹터별 분해("왜 빠졌나") → `state/cause.json`
- 결과는 `state/briefing.md`. 알림은 Telegram 달리아봇으로만(config channels).

---

## 3. EW 임계 백테스트 (신호 검증)

EW 신호 임계가 실제로 폭락을 앞섰는지 과거 데이터로 검증.

```python
from corvin_jarvis import ew_backtest as bt
hist = bt.build_history(series, dates, warmup=251)   # series=일별 시계열
res = bt.run_backtest(hist, cfg, horizon=10, thresh_pct=-3.0)
# res["semis_red"] = {precision, recall, f1, lead_time_avg, false_alarm_rate}
ranked = bt.sweep_param(hist, cfg, "semis", "divergence_high_dist_pct",
                        [1,3,5,10], "semis_red", 10, -3.0)   # 최적 임계 탐색
```
> ⚠️ **검증 결과(505일)**: 반도체(semis) 신호는 정밀도 18%≈기저율 16% = **엣지 약함**. vix_term만 2배 엣지. → 반도체 amber 신호 과신 금지. 정밀도는 항상 기저율과 비교.

---

## 4. 데이터 교차검증 (정확·신뢰의 토대)

```python
from corvin_jarvis import data_verify as dv
r = dv.verified("NVDA", {"yfinance": yf_fn, "kis": kis_fn}, positive=True)
# {value(중앙값), confidence(high/medium/low/none), flag, sources, spread_pct}
dv.reconcile_pnl(stored=-3.5, live=-10.5)   # 저장-라이브 괴리 → stale 경고
```
> 핵심: 한 소스만 믿지 말 것. ⚠️ yfinance·FDR은 둘 다 야후라 가짜 독립 — 진짜 2소스는 KIS.

---

## 5. 적대적 검증 (그룹씽크 차단)

토론/전략 결론을 4렌즈(신선도·상관관계·임계미검증·원인부재)로 반증. 과반 반증이면 폐기.

```python
from corvin_jarvis import adversarial as adv
r = adv.verify_conclusion("전량 매도하자", adv.live_lens_runner)
# {survives, refuted_count, weakest, votes}
```

---

## 핵심 원칙

- **모든 분석은 advisory only** — 실제 매매 주문 아님(모의).
- **데이터는 코드가 검증** — LLM이 만든 컨텍스트가 아니라 data_verify가 만든 사실을 기준 삼아 fact-check.
- **$0** — claude CLI는 구독 인증(API 키 불필요), 나머지는 로컬.
- 테스트 557개. 도구별 순수 코어는 네트워크 없이 전부 단위 테스트.
