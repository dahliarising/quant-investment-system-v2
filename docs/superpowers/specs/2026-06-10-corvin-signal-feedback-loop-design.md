# Corvin 시그널 피드백 루프 + 통합 중재 — 설계서

- **작성**: 2026-06-10
- **상태**: 설계 승인 (폐하, Discord 2026-06-10 08:31 KST "승인")
- **동기**: 예측 시스템 5개(predictive_engine·early_warning·regime·predict·leading)가
  파편화되어 상충 신호 가능 + 적중률 추적 부재 → confidence 고정값(65/60/90)이
  근거 없이 유지됨. 데이터 신뢰성 사고 반복(6/6 저장vs라이브 불일치, 6/8 가격 시점 라벨).

## 1. 진단된 문제 (우선순위 확정: P4→P2→P1→P3)

| # | 문제 | 증거 |
|---|------|------|
| P4 | 피드백 루프 부재 — 신호 발화 후 결과 추적·보정 없음 | counterfactual.py(월간)·ew_backtest는 예측신호와 미연결 |
| P2 | 예측 서브시스템 5개 파편화 — 상호 무인지, 상충 중재자 없음 | EW "줄이세요" + playbook "BUY_NOW" 동시 가능 |
| P1 | 예측 엔진 단순 — 선형 기울기(변동성 무시), confidence 하드코딩 | predictive_engine.py:95 `confidence=65.0` |
| P3 | 데이터 검증 수동 — 발화 전 자동 교차검증 게이트 없음 | 6/6 MA50 오류·FRED 불안정, 6/8 '현재가' 라벨 사고 |

## 2. 채택 접근: 🅐 점진 원장-우선 (폐하 승인)

기존 엔진 무수정(additive). 모든 신호를 발화 시점에 SQLite 원장에 기록 →
만기 채점 → 적중률 기반 confidence 보정 → 같은 원장 위에 중재 레이어 →
예측 고도화 → 검증 게이트. 단계별 독립 배포, 무중단.

- 대안 🅑 빅뱅 통합 플랫폼: 회귀 리스크 큼, 기각.
- 대안 🅒 외부 프레임워크(vectorbt 등): 의존성 무거움·이중구조, 기각.

## 3. 전체 아키텍처

```
[기존 5엔진: signal_engine·predictive·early_warning·leading·playbook]
        │ 발화 직후 ledger.record() 1줄 (additive 연결)
        ▼
signals/ledger.py ─ SQLite (state/signal_ledger.db)     ← Phase 1
        │
signals/scorer.py ─ 만기 신호 채점 cron (16:30 KST)      ← Phase 1
        │
signals/calibration.py ─ 엔진×종류별 적중률 → confidence  ← Phase 1
        │
signals/arbiter.py ─ 종목별 상충 중재 → 단일 최종 액션    ← Phase 2
        │
predictive_engine 고도화 (ATR 정규화·신뢰구간·적응 임계값) ← Phase 3
        │
signals/data_gate.py ─ 발화 전 다출처 검증               ← Phase 4
```

## 4. Phase 1 — 신호 적중률 원장 (P4)

### 4.1 `signals/ledger.py`

발화 신호 기록. 공통 스키마 (EngineSignal·PredictiveSignal·LeadingSignal 합집합,
무손실 기록):

```sql
CREATE TABLE IF NOT EXISTS signal_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,            -- 발화 시각 (KST ISO)
  engine TEXT NOT NULL,        -- signal_engine|predictive|early_warning|leading|playbook
  symbol TEXT NOT NULL,        -- ""=매크로/시장 전체
  kind TEXT NOT NULL,          -- STOP|WATCH|VELOCITY|RS_WEAK|EVENT|EW_GAUGE|...
  direction TEXT,              -- bull|bear|neutral|NULL
  urgency INTEGER,             -- 0-100
  confidence REAL,             -- 발화 시점 confidence (보정 전 원본)
  horizon_days INTEGER,        -- 평가 만기. NULL이면 kind별 기본 (STOP/WATCH=5, 그 외=10)
  evidence TEXT,               -- JSON (발화 근거 + 당시 가격/지표)
  status TEXT DEFAULT 'open',  -- open|hit|late_hit|miss|unscorable
  scored_at TEXT,              -- 채점 시각
  outcome TEXT                 -- JSON (실제 결과 데이터)
);
```

- **중복 방지**: 같은 (engine, symbol, kind) open 신호가 있으면 재기록 생략
  (동일 상태 지속 = 1건 유지, 08:01 KOSPI식 재발송 금지 원칙과 동일).
- timeseries.py의 SQLite 패턴 재사용 (`state/` 디렉토리, init_db/write 구조).
- 순수 함수 + 명시적 경로 주입으로 테스트 가능.

### 4.2 `signals/scorer.py` — 채점 규칙

만기(horizon_days 경과) open 신호를 평가. **HIT / LATE_HIT / MISS / UNSCORABLE** 4단계:

| kind | HIT 기준 | LATE_HIT |
|------|----------|----------|
| VELOCITY | horizon일 내 종가가 손절선 도달 | horizon×1.5 내 도달 |
| RS_WEAK | 이후 10일 벤치마크 대비 추가 언더퍼폼 (rs<0) | — |
| EVENT | 이벤트 당일 실현변동성 > 직전 20일 평균 ×1.3 | — |
| STOP/WATCH | 신호 후 5일 내 추가 하락 (방어 신호의 유효성) | — |
| EW/leading (bear) | 이후 horizon일 지수 하락 | horizon×1.5 |
| EW/leading (bull) | 이후 horizon일 지수 상승 | horizon×1.5 |

- 가격 데이터 미확보 시 UNSCORABLE (적중률 분모에서 제외).
- cron: 매일 16:30 KST (장마감 후, run_digest.sh 패턴 재사용 — `run_scorer.sh`).

### 4.3 `signals/calibration.py`

- 엔진×kind별 집계: `hit_rate = (hit + 0.5×late_hit) / (hit+late_hit+miss)`
- 표본 < 10건이면 보정 보류 (원본 confidence 유지 + "미보정" 태그).
- 보정 공식: `calibrated = 원본 confidence × shrink + hit_rate×100 × (1-shrink)`,
  shrink는 표본 수 기반 (n=10→0.7, n≥50→0.2). 베이지안 수축으로 소표본 과보정 방지.
- 출력: `state/calibration.json` — Phase 3에서 predictive_engine이 주입받음.

### 4.4 연결점 (additive 1줄)

- `dashboard/snapshot.py` `_engine_signals`/`_predictive_signals` 직후 `ledger.record_batch()`
- `run_leading.py` collect 직후
- `jarvis.py` alert 발화 직후
- 실패해도 신호 흐름 무영향 (`_safe` 패턴 — 원장 기록 실패는 로그만).

### 4.5 대시보드

command_center에 "엔진 적중률" 미니패널: 엔진별 hit_rate + 표본 수 + open 신호 수.
snapshot에 `signal_scoreboard` 섹션 추가.

## 5. Phase 2 — 통합 중재자 (P2)

### 5.1 `signals/arbiter.py`

- 입력: 종목별 active 신호 묶음 (ledger의 open 신호 + 실시간 평가분).
- 중재 규칙 (우선순위 순):
  1. **안전 우선**: STOP(손절)·EW 🔴 계열은 어떤 매수 신호보다 우선.
  2. **적중률 가중**: 상충 시 calibration hit_rate 높은 엔진 우세.
     양쪽 다 미보정(표본<10)이면 보수적 쪽(방어 액션) 선택.
  3. **최신 데이터 우선**: 같은 엔진 내 신호는 최신 발화분만.
- 출력: 종목당 단일 `FinalAction(symbol, action, urgency, rationale, sources)`.
  rationale에 상충 내역 명시 (예: "playbook BUY_NOW vs EW 🟠 상충 — EW 적중률 우세로 보류").
- 소비자 전환: snapshot → digest → debate 순으로 arbiter 출력 사용 (단계적).

## 6. Phase 3 — 예측 엔진 고도화 (P1)

- **VELOCITY**: 단순 평균 기울기 → ATR(14) 정규화 기울기.
  `slope_norm = slope / ATR` — 변동성 큰 장세 오탐 제거.
- **신뢰구간**: days_to_stop 점추정 → 범위 (기울기 표준오차 기반 "3~7일").
- **confidence**: 고정값 제거 → `state/calibration.json` 주입 (없으면 기존 기본값 fallback).
- **RS_WEAK 적응 임계값**: 고정 -5%p → 종목별 60일 rs 분포의 하위 10분위.
  데이터 부족 시 기존 고정값 fallback.
- 기존 테스트 31건 유지 + 신규 케이스 추가 (회귀 0 원칙).

## 7. Phase 4 — 데이터 검증 게이트 (P3)

### 7.1 `signals/data_gate.py`

신호 발화 **전** 자동 검증 (기존 data_verify.py 패턴 확장):

| 체크 | 규칙 | 실패 시 |
|------|------|---------|
| 가격 교차 | KIS vs yahoo ±1% 초과 불일치 | 신호 보류 + ⚠️ 경고 발화 |
| 시점 라벨 | 장마감 상태면 '전일종가/종가' 라벨 강제 | 라벨 자동 교정 |
| staleness | 데이터 3일 초과 경과 | 신호 차단 + 갱신 요청 |
| NaN/inf | 비정상 수치 | 해당 신호만 drop |

- KR 종목은 pykrx/KIS만 교차 (yfinance 금지 규칙 준수 — KR은 KIS vs pykrx).
- 게이트는 ledger.record 직전 훅 — 통과 신호만 기록·발화.

## 8. 에러 처리 / 안전 원칙

- 원장·채점·중재 모두 실패해도 기존 신호 흐름 무영향 (`_safe` 격리 패턴).
- 실주문 0 유지 (advisory only). 자동 alert은 Telegram 달리아봇 라우팅 규칙 준수.
- 알림 노이즈 금지: 적중률 리포트는 일일 다이제스트에 통합 (별도 push 없음).

## 9. 테스트 전략

- 각 Phase TDD (RED→GREEN→IMPROVE), 커버리지 80%+.
- scorer는 가격 데이터 주입(closes_by_sym 패턴)으로 순수 함수 테스트.
- 실제 전송 금지 (conftest autouse 안전망 유지).
- Phase 완료마다 전체 회귀 (`pytest tests/`) + 대시보드 스크린샷 보고 → 폐하 승인 게이트.

## 10. 구현 순서 / 산출물

| Phase | 산출물 | 검증 |
|-------|--------|------|
| 1 | ledger.py·scorer.py·calibration.py + run_scorer.sh cron + 대시보드 패널 | 신호 기록→채점→적중률 표시 E2E |
| 2 | arbiter.py + snapshot/digest 소비 전환 | 상충 시나리오 테스트 + 실데이터 중재 결과 |
| 3 | predictive_engine ATR·신뢰구간·캘리브레이션 주입 | 기존 31 테스트 + 신규, 오탐률 비교 |
| 4 | data_gate.py + record 전 훅 | 불일치/낡은 데이터 주입 테스트 |

Phase당 1세션. 각 Phase 완료 시 폐하 승인 후 다음 진행 (조건부 승인 게이트 규칙).
