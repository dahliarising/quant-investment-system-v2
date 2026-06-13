# Corvin 실시간 투자 토론 툴 (Live Debate Tool) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans. 모든 스텝 체크박스(`- [ ]`) 추적. 코드는 TDD(superpowers:test-driven-development).

**Goal:** 두 종류의 투자 토론을 **실시간(스트리밍)·정확(교차검증)·신뢰(환각차단)** 하게 생성하는 재사용 도구. 지금까지 수동으로 하던 "에이전트 배치 → 정적 HTML"을 라이브·검증 파이프라인으로 승격.

**두 대화 종류 (모드):**
- **STANCE 모드** — 행동 스탠스 토론: 손절 / 매수 / 홀딩
- **STYLE 모드** — 투자방법 토론: 가치(버핏) / 성장(우드) / 추세(미너비니) / 매크로(달리오) / 퀀트(시먼스)

**3대 요구 → 3대 메커니즘 (1:1):**
| 요구 | 메커니즘 |
|---|---|
| **실시간** | 턴을 SSE로 스트리밍 → 카톡 UI에 말풍선 라이브 등장(타이핑 인디케이터) |
| **정확** | 모든 수치 `data_verify.verified()` ≥2소스 교차검증 + `ew_backtest`로 신호 신뢰도 태깅(semis=약함/vix=2배엣지) |
| **신뢰** | 에이전트 발언의 숫자를 컨텍스트와 대조하는 **fact-check 패스** → 환각/근거없는 수치 flag |

**재사용 부품 (이미 구현):** `data_verify`(Phase 0)·`ew_backtest`(Phase 1)·`ew_providers`·기존 카톡 HTML 렌더·`qualitative.py`(claude CLI 헤드리스 패턴).

**Tech 결정 (권장):**
- LLM 백엔드 = **claude CLI 헤드리스**(구독 인증, API키 불필요, qualitative.py 패턴 계승). 대안: Claude API(키 필요, 빠름).
- 실시간 전송 = **SSE**(단방향 스트림, 단순). 대안: WebSocket.
- 호스팅 = 기존 대시보드 서버(포트 8501)에 `/debate` 라우트 추가. 대안: standalone.

---

## Architecture (4 레이어)

```
[검증 컨텍스트] verified data + backtest 신뢰도   ← 정확·신뢰의 토대
        ↓ (공유 컨텍스트, 모든 에이전트 동일)
[토론 엔진] personas × rounds(opening→rebuttal→synthesis), LLM=DI
        ↓ (턴 생성)
[fact-check] 발언 숫자 ↔ 컨텍스트 대조, flag        ← 신뢰
        ↓ (검증된 턴)
[실시간 UI] SSE 스트림 → 카톡 말풍선 라이브        ← 실시간
```

순수 코어(엔진·fact-check·컨텍스트조립)는 LLM/네트워크 DI로 전부 테스트.

---

## Phase A — 검증 컨텍스트 빌더 (`live_debate/context.py`)

**목적:** 모든 에이전트가 공유할 "검증된 사실 묶음" 1개. 여기 없는 수치는 토론에 못 씀.

**Files:** Create `corvin_jarvis/live_debate/context.py`, `tests/test_debate_context.py`

- [ ] **A.1** `build_context(holdings, fetchers, ew_cfg) -> dict` — 보유 각 종목 가격을 `data_verify.verified()`(≥2소스)로, pnl은 `reconcile_pnl`(저장vs라이브)로. 각 값에 `{value, confidence, flag, sources}` 부착.
- [ ] **A.2** `attach_signal_reliability(ctx, backtest_report) -> ctx` — EW 신호별 백테스트 신뢰도 태깅: `semis="low-edge(P18%≈기저)"`, `vix_term="2x-edge"`. 에이전트가 약한 신호 과신 못 하게.
- [ ] **A.3** `context_facts(ctx) -> set[fact]` — fact-check가 대조할 "허용된 사실"(수치·라벨) 집합 추출.
- [ ] 테스트: fake fetcher 주입 → 교차검증된 컨텍스트 + 저신뢰 종목 flag + 신호 신뢰도 태그. (네트워크 X)
- [ ] **성공기준:** 보유 종목 컨텍스트가 소스·신뢰도·flag 포함, 불일치 데이터는 confidence=low로 표시.

---

## Phase B — 토론 엔진 (`live_debate/engine.py`)

**목적:** 페르소나 × 라운드 오케스트레이션. LLM 호출은 DI(테스트는 fake LLM).

**Files:** Create `corvin_jarvis/live_debate/personas.py`, `corvin_jarvis/live_debate/engine.py`, tests

- [ ] **B.1** `personas.py` — STANCE/STYLE 두 세트. 각 페르소나 `{id, name, avatar, accent, method, system_prompt}`. (이번 세션 검증된 5+3 프롬프트 이식.)
- [ ] **B.2** `run_round(personas, context, round_type, llm) -> list[turn]` — round_type∈{opening, rebuttal, synthesis}. 각 페르소나에 공유 컨텍스트+직전 라운드 주고 LLM 호출 → turns. `llm` 콜백 DI.
- [ ] **B.3** `debate(mode, context, llm, rounds=[...]) -> generator[turn]` — 라운드 순차 실행, 턴을 **yield**(스트리밍 토대).
- [ ] 테스트: fake llm(고정 응답) 주입 → opening/rebuttal 턴 생성·순서·페르소나 매핑 검증.
- [ ] **성공기준:** mode 전환(stance↔style)으로 다른 페르소나 세트, 턴 제너레이터가 라운드별 순차 yield.

---

## Phase C — Fact-check 패스 (`live_debate/factcheck.py`)

**목적:** 에이전트가 인용한 숫자가 검증 컨텍스트와 맞는지 대조 → 환각 차단(신뢰).

**Files:** Create `corvin_jarvis/live_debate/factcheck.py`, tests

- [ ] **C.1** `extract_claims(turn_text) -> list[claim]` — 발언에서 수치·종목·라벨 추출(정규식/간단 파서).
- [ ] **C.2** `verify_turn(turn, context_facts, tol_pct) -> {ok, flags}` — 각 claim을 허용 사실과 대조. 불일치(예: 컨텍스트엔 −10.5%인데 −5%라 말함) → flag. 컨텍스트에 없는 새 수치 → "unverified" flag.
- [ ] **C.3** `annotate(turn, verdict) -> turn` — 턴에 ✓검증 / ⚠️미검증 / ✗불일치 배지 부착(UI 표시용).
- [ ] 테스트: 정확한 인용→ok, 틀린 수치→불일치 flag, 없는 수치→unverified.
- [ ] **성공기준:** 컨텍스트와 어긋난 수치를 인용한 턴이 ✗로 flag되어 UI에 경고 표시.

---

## Phase D — 실시간 서버 + 스트리밍 UI (`live_debate/server.py` + `debate_view.html`)

**목적:** 토론을 SSE로 스트리밍 → 카톡 UI에 말풍선 라이브 등장.

**Files:** Modify dashboard server(또는 Create `corvin_jarvis/live_debate/server.py`), Create `corvin_jarvis/dashboard/static/debate_view.html`

- [ ] **D.1** `GET /debate/stream?mode=style` — 컨텍스트 빌드 → `debate()` 제너레이터 → 각 턴 fact-check → **SSE event 전송**(`data: {turn json}`).
- [ ] **D.2** `debate_view.html` — 이번 세션 카톡 CSS 재사용. SSE 구독 → 턴 도착마다 말풍선 append + 타이핑 인디케이터 + fact-check 배지(✓/⚠️/✗).
- [ ] **D.3** 모드 토글(스탠스/스타일) + "토론 시작" 버튼 + 데이터 신뢰도 헤더(예: "가격 2소스 일치 ✓").
- [ ] **D.4** 트랜스크립트+데이터 스냅샷 로그 저장(재현성).
- [ ] 테스트: SSE 라우트가 fake debate 제너레이터로 순차 event 방출(httpx/TestClient). UI는 수동 스모크.
- [ ] **성공기준:** 브라우저에서 "토론 시작" → 말풍선이 하나씩 라이브 등장, 각 턴에 검증 배지, 모드 전환 동작.

---

## Phase E — 통합 + 스모크

- [ ] **E.1** 라이브 LLM(claude CLI) 바인딩 + 실제 보유 컨텍스트로 1회 스타일 토론 스트리밍 스모크.
- [ ] **E.2** 두 모드 모두 동작 확인, fact-check가 실제 환각 1건이라도 잡는지 관찰.
- [ ] **E.3** 알림 노이즈 0(이건 on-demand 도구, cron push 아님 — alert_noise_aversion 무관).

---

## 신뢰·정확 강조 (왜 이게 "정확·신뢰"한가)

- **정확**: 토론의 모든 사실이 단일 소스 raw fetch가 아니라 `verified()` ≥2소스 중앙값 + 신뢰도. 저장-라이브 괴리는 `reconcile_pnl`이 잡음.
- **신뢰**: ① 신호 신뢰도가 백테스트로 태깅돼 에이전트가 semis 같은 약한 신호 과신 못 함. ② fact-check가 에이전트 환각 수치를 ✗로 표시 → 보는 사람이 어느 발언이 근거있는지 즉시 구분.
- **실시간**: 배치-후-렌더가 아니라 SSE 스트림 → 진짜 "대화가 일어나는" 경험.

## Self-Review

- **재사용**: data_verify·ew_backtest·카톡 CSS·qualitative CLI 패턴 전부 부품으로. 신규 최소.
- **순수/DI**: context·engine·factcheck 전부 LLM/네트워크 주입 → 테스트 가능. 서버만 IO.
- **모드 확장성**: personas.py에 세트 추가하면 새 토론 종류(예: 거시 시나리오 토론) 확장.
- **데드앵글**: fact-check는 같은 LLM이 짠 컨텍스트를 기준 삼음 → 컨텍스트 자체가 틀리면 못 잡음. 그래서 컨텍스트는 LLM이 아니라 data_verify(코드)가 만든다(핵심).
- **횡단 원칙**: 실시간 대화 표현 + 이중·삼중 재검증([[feedback_realtime_debate_reverify]]) 정확 반영.
- **스펙 커버리지**: 두 모드(B.1)·실시간(D)·정확(A)·신뢰(C) 전부 대응.
