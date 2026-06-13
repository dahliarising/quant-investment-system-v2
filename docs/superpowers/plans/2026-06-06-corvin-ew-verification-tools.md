# Corvin 검증 툴 (EW Verification Suite) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 또는 executing-plans로 task-by-task 구현. 모든 스텝 체크박스(`- [ ]`) 추적.

**Goal:** 토론/전략 결론의 *진짜 검증*을 위한 3개 툴 구현 — ① EW 임계 백테스트(미검증 임계 해결), ② 촉발원인 규명(why 공백 해결), ③ 적대적 검증 에이전트(그룹씽크 해결). 2026-06-06 토론 허점 분석에서 도출.

**해결 대상 허점** (Discord 분석 ①⑤⑦):
- ⑦ EW 임계값(−8%·semis 3% 등)이 백테스트 0인 오늘 만든 초기 추정치 → **Tool 1 백테스트**
- ⑤ "왜 빠졌나" 원인 진단 공백 → **Tool 2 원인규명**
- ① 합의=착시(동일 데이터·동일 룰 3변주, 스펙트럼 좁힘) → **Tool 3 적대 검증**

**Architecture:** 기존 EW 패턴 계승 — 순수함수 코어(테스트 가능) + DI 데이터 경계 + 별도 진입점. EW 분류기(`early_warning.py`)·reading(`ew_providers.py`)는 **이미 순수/DI라 백테스트는 과거 데이터만 주입(코어 변경 0)**.

**Tech Stack:** Python 3.9, pytest, yfinance/FRED(과거 시계열), 기존 geopolitical-risk MCP·narrative.db·geo_signal.py·narrative.py 재사용.

**우선순위 (폐하 2026-06-06 "2" = 전부 순서대로):** Phase 0(데이터 교차검증) → 1(백테스트) → 2(원인) → 3(적대). Phase 0가 모든 것의 토대 — 틀린 숫자 위엔 백테스트도 무의미.

**🚨 횡단 원칙 (폐하 2026-06-06 지시, 모든 Phase 적용):**
- **데이터 이중·삼중 교차검증**: 모든 핵심 수치는 ≥2 독립 소스 교차확인 후 사용. 불일치 시 flag + 보수적 처리(중앙값 or 보류), 추측 금지. → Phase 0가 제공, Tool 1·2·3은 raw fetch 대신 Phase 0 경유.
- **실시간 대화 표현방식**: 백테스트·원인규명·적대검증 결과를 카톡/에이전트 회의 스타일로 렌더 — 데이터를 서로 교차확인하는 재검증 과정이 보이게. (참조: feedback_realtime_debate_reverify)

---

## Phase 0 — 데이터 이중·삼중 교차검증 레이어 (`data_verify.py`) ★토대

**목적:** 모든 증거 데이터를 ≥2 독립 소스로 교차확인 → 틀린 숫자가 분석에 못 들어오게. 이번 세션 데이터 불일치(저장vs라이브 pnl, MA50 오류, FRED 불안정) 재발 방지.

**Files:**
- Create `corvin_jarvis/data_verify.py`
- Create `tests/test_data_verify.py`

### Task 0.1: 교차검증 코어 (순수, DI)
- [ ] `cross_check(key, source_values: dict[str,float|None], tol_pct: float) -> dict` — 여러 소스 값 비교 → `{value, agree: bool, sources, spread_pct, confidence, flag}`. 일치(허용오차내)면 중앙값+high, 불일치면 flag+보수처리(중앙값 사용하되 confidence low), 단일소스면 confidence medium, 0소스면 None(스킵).
- [ ] 테스트: 2소스 일치/불일치/단일/결측 4케이스 → 기대 confidence·flag.

### Task 0.2: 다소스 fetcher 묶음 (DI 경계)
- [ ] `verified_price(sym, fetchers) -> dict` — US: yfinance + (KIS quote_provider) / KR: pykrx + FDR. 각 소스 호출 후 cross_check. 라이브 바인딩은 기존 quote_provider·dca_timing.default_fetcher·kr_data 재사용.
- [ ] `verified_fred(code, fetchers)` — fredgraph + 캐시/대체. 
- [ ] 테스트: fake 다소스 주입 → verified dict (네트워크 X).

### Task 0.3: 시점·정합 가드
- [ ] `reconcile_pnl(stored, live, snapshot) -> dict` — 세 시점 비교, 괴리 크면 flag(저장 stale 경고). live 우선.
- [ ] `sanity(value) -> bool` — NaN/inf/음수가격/outlier 가드.
- [ ] 테스트: 저장-라이브 괴리 시나리오 → stale flag.
- [ ] **성공기준:** 임의 종목 → 2소스 교차검증가 + confidence + flag. 불일치 데이터는 분석 진입 차단.

---

## Phase 1 — EW 임계 백테스트 (`ew_backtest.py`)

**목적:** 5지표 임계가 실제로 과거 드로다운을 *선행*했는지 + 거짓경보율 측정 + 최적 임계 산출.

**Files:**
- Create `corvin_jarvis/ew_backtest.py`
- Create `tests/test_ew_backtest.py`
- Create `corvin_jarvis/state/ew_history_cache.json` (과거 시계열 1회 수집 캐시)

### Task 1.1: 과거 시계열 수집기
- [ ] `fetch_history(start, end) -> dict[date -> readings]` — SOXX/SPY·^VIX/^VIX3M·HY(FRED BAMLH0A0HYM2)·curve(FRED T10Y2Y)·breadth(유니버스 MA200) 일별. yfinance/live_fred_fetcher 재사용. 캐시 저장.
- [ ] 테스트: 주입된 fake 시계열 → 날짜별 readings dict 정확 산출 (네트워크 X).
- [ ] 결측 처리: 지표별 누락일 스킵(추측 금지, 기존 원칙).

### Task 1.2: 히스토리컬 게이지 시계열
- [ ] `replay(history, cfg) -> dict[date -> {states, gauge, transitions}]` — 각 날짜에 `early_warning.classify_*` + `composite_gauge` + `detect_transitions`(전일 대비) 재현. **순수함수만 호출**.
- [ ] 테스트: 알려진 입력 시퀀스 → 기대 게이지/전환 시계열.

### Task 1.3: 이벤트 라벨링 + 평가
- [ ] `label_drawdowns(spx_closes, horizon=5, thresh=-3.0) -> set[date]` — 향후 horizon일 내 SPX(또는 보유 바스켓) 드로다운 ≥ thresh = "폭락 이벤트".
- [ ] `evaluate(replay, events) -> dict` — 지표/게이지별 **precision·recall·lead-time(평균 며칠 선행)·false-alarm rate** + confusion matrix.
- [ ] 테스트: 합성 시나리오(신호 후 폭락 O/X)로 precision/recall 정확 계산.

### Task 1.4: 임계 스윕 + 리포트
- [ ] `sweep(history, grid) -> ranked configs` — green_max/amber_max/hard_stop 등 그리드 → F1·lead-time 최적 임계 랭킹 → **config 권장값 산출**.
- [ ] `report(...)` — 텍스트/HTML: 지표별 성적표 + 권장 임계 + "EW 신호 따름 vs Buy&Hold" equity curve.
- [ ] 테스트: 스윕이 더 나은 임계를 더 높게 랭크.

### Task 1.5: 진입점 + 스모크
- [ ] `run_backtest.py` CLI — 기간 인자, 캐시 사용, 리포트 출력.
- [ ] 스모크: 실제 2018~now 1회 실행 → 5지표 성적표 출력 확인.
- [ ] **성공기준:** semis/vix/hy/curve/breadth 각 precision·recall·lead-time 수치 + 권장 임계 + equity curve 산출. (config _comment "초기값" → "백테스트값"으로 교체 근거 확보.)

---

## Phase 2 — 촉발원인 규명 (`cause_attribution.py`)

**목적:** "오늘 왜 빠졌나"를 데이터로 → 패닉 vs 추세 판단 근거.

**Files:**
- Create `corvin_jarvis/cause_attribution.py`
- Create `tests/test_cause_attribution.py`

### Task 2.1: 멀티소스 이상치 수집 (DI)
- [ ] `gather_candidates(date, fetchers) -> list[cause]` — 후보 원인별 당일 이상치:
  - 금리/스프레드 일변(FRED: DGS10·T10Y2Y·HY) · 섹터 등락 분해(어디서 시작 — 반도체발? SOXX vs SPY) · 환율/유가(geopolitical-risk MCP: ecos_exchange_rate·commodity_prices) · 뉴스/내러티브(gdelt_risk_score·narrative.db Z-score).
- [ ] 테스트: fake fetcher 주입 → 후보 리스트 + 이상치 점수.

### Task 2.2: 원인 랭킹 + 메시지
- [ ] `rank_causes(candidates) -> ranked` — 이상치 크기·동시성으로 top-3 + 신뢰도.
- [ ] `attribution_message(ranked)` — "오늘 −4% = ① 금리 10Y +Xbp ② 반도체 차익실현(SOXX −5.2%) ③ [헤드라인]" 형식.
- [ ] 테스트: 명확한 단일 원인 시나리오 → 해당 원인 top-1.

### Task 2.3: jarvis 브리핑 통합 + 스모크
- [ ] `merge_cause_attribution()` — 브리핑에 "오늘 급락 원인(추정)" 1줄 추가. 기존 merge_* 패턴.
- [ ] **성공기준:** 임의 급락일 입력 → top-3 원인 후보 + 신뢰도 산출.

---

## Phase 3 — 적대적 검증 에이전트 (`adversarial.py`)

**목적:** 토론 결론을 *반증*하려는 회의론자 패널 → 그룹씽크 차단, 살아남으면 confidence↑.

**Files:**
- Create `corvin_jarvis/adversarial.py`
- Create `tests/test_adversarial.py`

### Task 3.1: 회의론자 패널 (다양한 렌즈)
- [ ] `build_skeptics() -> list[lens]` — 4 렌즈: (i)데이터 신선도/장중 미완성봉, (ii)상관관계/가짜분산, (iii)임계 미검증(백테스트 근거 요구), (iv)촉발원인 부재.
- [ ] 각 렌즈 = 결론 입력받아 `{refuted: bool, reason, confidence}` 구조화 출력(schema). 기존 Agent/agents 인프라 재사용.

### Task 3.2: 다수결 판정 + 통합
- [ ] `verify_conclusion(conclusion, readings) -> {survives, votes, weakest_point}` — ≥과반 반증 시 survives=False → 결론 재검토 플래그.
- [ ] Phase 1 백테스트 결과를 렌즈(iii) 입력으로 연결(임계 검증됨/안됨).
- [ ] 테스트: fake 투표로 survives 로직 검증.
- [ ] **성공기준:** 임의 결론 입력 → 4 렌즈 판정 + survives + 최약점.

---

## 공통 원칙 / Self-Review

- **순수 코어 + DI 경계**: EW 패턴 계승. 백테스트/원인/적대 전부 네트워크/디스크 격리 테스트.
- **추측 금지**: 데이터 결측 시 스킵(기존 원칙). 백테스트 lookahead bias 금지(미래 데이터 누설 방지 — readings는 t시점 정보만).
- **노이즈**: 원인규명은 브리핑 1줄, 적대검증은 전략 산출물 게이트 — 알림 폭증 X(alert_noise_aversion 준수).
- **재사용**: geo_signal·narrative·ew_providers·agents 기존 자산 활용, 신규 코드 최소.
- **데드앵글 주의**: 백테스트도 같은 모델이 짜고 검증 → 가능하면 out-of-sample(학습기간/검증기간 분리)로 과적합 방지.
- **스펙 커버리지**: 허점 ①(Tool3)·⑤(Tool2)·⑦(Tool1) 대응. 나머지 허점(②④⑥⑧)은 백테스트·원인규명 부산물로 일부 완화(②VIX 절대수준=백테스트 feature, ④상관=원인분해, ⑥미완성봉=lookahead 가드, ⑧약세홀드=백테스트가 MA이탈 종목 성적 노출).
