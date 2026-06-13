# Corvin 예측 다이제스트 Phase 2 설계서 (시스템 6~10)

**작성:** 2026-06-13
**전제:** Phase 1(시스템 1~5) 완료. 동일 `PredictionResult` 계약·`daily_history` 데이터·다이제스트 조립기 재사용.

## 목표

예측 방법론을 5종 추가하고, **백테스트로 검증 통과한 것만** 다이제스트에 합류시킨다.

| # | 카탈로그 이름 | 목적 | 구현 수단 (의존성 0) |
|---|--------------|------|---------------------|
| ⑥ | ML 분류기 | "다음 H일 상승 확률 P%" | **numpy 로지스틱 회귀** (XGBoost 대체) |
| ⑦ | 시계열 | "다음 H일 가격 밴드 [p10,p90]" | **경험적 분위수 밴드** (Prophet/LSTM 대체) |
| ⑧ | 센티먼트 | "뉴스 분위기 방향" | geo MCP GDELT 주입식 (Phase 1 geo 패턴) |
| ⑨ | 몬테카를로 | "1개월 뒤 분포 + 손절이탈 확률" | **numpy GBM 시뮬** |
| ⑩ | 앙상블 | "방법론 합의 방향" | 백테스트 가중 다수결 |

## 핵심 설계 결정

### 1. 의존성 0 (numpy + scipy만)
폐하 맥에 xgboost/prophet/torch 미설치 + Phase 1 철학 계승. 모든 모델은 numpy/scipy 순수구현.
- ⑥ 로지스틱: 경사하강 self-contained (sigmoid + BCE). feature = Phase 1 벡터 feature(지수·매크로 z-score 수익률).
- ⑦ 밴드: 과거 H일 누적수익률 경험분포의 분위수 → 현재가에 곱해 가격 밴드.
- ⑨ 몬테카를로: 일일 로그수익률 μ,σ 추정 → GBM 경로 N개 → H일 종가 분포.

### 2. 백테스트 게이트 (안전 척추)
`backtest.py` — 각 방법론을 walk-forward로 과거 검증:
- **방향계 모델(⑥⑩)**: directional hit-rate. baseline = 다수클래스(naive) 적중률. **hit > baseline + margin** 통과.
- **분포계 모델(⑨)**: 실현 H일 수익률이 예측 [p5,p95] 안에 든 비율(coverage) ≈ 0.9 ±tol 통과.
- **밴드계(⑦)**: 동일 coverage 검증.
- 결과 → `state/phase2_backtest.json`. 오케스트레이터가 읽어 **통과 모델만** 다이제스트 합류.
- 미검증/탈락 모델은 PredictionResult로 안 나가거나 `data_ok=False`("백테스트 미통과 · 보류").

### 3. 정직성 게이트 유지
표본 부족/모델 미수렴/백테스트 미통과 → `insufficient()`. 가짜 정밀도 금지(Corvin 철칙).

### 4. 다이제스트 통합
- ⑥⑨ → 보유종목 라인에 인라인 병합(Phase 1 consolidation 재사용).
- ⑦ → 시장/종목 밴드 (market scope: nasdaq/kospi 밴드).
- ⑧ → 시장 방향 섹션(geo 옆).
- ⑩ → 시장 방향 최상단 "합의" 라인.

## 모듈 구조

| File | 책임 |
|------|------|
| `m_logistic.py` | ⑥ numpy 로지스틱 방향 분류 (순수함수 + 어댑터) |
| `m_band.py` | ⑦ 경험적 분위수 가격 밴드 |
| `m_sentiment.py` | ⑧ GDELT 센티먼트 주입식 변환 |
| `m_montecarlo.py` | ⑨ GBM 몬테카를로 분포 |
| `m_ensemble.py` | ⑩ 백테스트 가중 합의 |
| `backtest.py` | walk-forward 검증 + 게이트 JSON |

## 비목표 (Phase 2 범위 밖)
- 실시간 SNS 크롤링(⑧는 GDELT만), GPU 학습, 하이퍼파라미터 auto-tune(Phase 3 auto-tuner).
- 종목별 개별 ML(데이터·과적합 리스크) — 우선 시장지수 레벨에서 검증 후 확장.
