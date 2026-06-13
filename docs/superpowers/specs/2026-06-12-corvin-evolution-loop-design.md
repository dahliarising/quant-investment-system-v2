# Corvin Evolution Loop — 자가진화 예측·설득 루프 설계서

- **작성**: 2026-06-12
- **상태**: 설계 승인 (폐하, Discord 2026-06-12 09:32 KST "승인!")
- **동기**: 폐하 요청 — "루프 시스템으로 실시간 정보+다양한 내러티브 신호를 종합해
  가장 예측 잘맞는 주식 시스템을 완성하고, 크론으로 효과적으로 알리고 다양한 방식으로
  설득해 실행가능하게 하라." (Boris Cherny 루프/ralph-loop 개념 차용)
- **선행 설계**: [2026-06-10 시그널 피드백 루프](./2026-06-10-corvin-signal-feedback-loop-design.md)가
  Loop 1(라이브 신호)·Loop 2(채점/보정)·arbiter를 이미 구축. 본 설계는 그 위에
  **Loop B(자가진화)** 와 **설득 브리핑 레이어**를 얹는다.

---

## 0. brainstorming 확정 사항 (Discord 2026-06-12)

| 분기 | 결정 | 의미 |
|------|------|------|
| 루프 종류 | **B3-full** | 코드 자가수정 + **무인 자동머지** |
| 자가수정 scope | **ⓒ 레포 전체** | quant_investment_system_v2 레포 내 무제한 |
| 목적함수 | **ⓓ 복합 (백테스트 게이트 필수)** | walk-forward OOS 엣지 = 머지 필수관문, 적중률·설득성공 보조 |
| 케이던스 | **ⓑ 분리** | 브리핑=매일+이벤트, evolution=야간배치+온디맨드 |
| evolution 구현 | **② 커스텀 러너** | 전용 러너 + 결정론적 머지하니스 (토너먼트③는 후속 확장) |

---

## 1. 핵심 원칙 (B3-full 안전 척추)

> **LLM은 코드를 제안만 하고, 머지는 LLM 손 밖의 결정론적 하니스가 판정한다.**

무인 자동머지 + 레포 전체 자가수정이 결합되면, 머지 결정을 LLM이 내릴 경우 루프가
Goodhart로 "개선됐다"를 합리화해 무엇이든(자기 브레이크 제거 포함) 머지할 수 있다.
따라서:

1. **심판은 레포 밖**: 러너·머지하니스·pinned 백테스트 평가자·킬스위치는
   `~/corvin_evolution/`(레포 밖)에 둔다. 루프는 *주식 시스템의 뇌*를 무제한 수정해도,
   *자기를 심판하는 자*는 물리적으로 못 건드린다. ⓒ(주식코드 무제한)와 모순되지 않음 —
   진화 인프라는 피수정 대상과 분리된 별개 레이어다.
2. **실험장 vs 판사 분리**: in-repo 백테스트 하니스는 루프의 실험 도구(자유 수정 가능),
   external pinned 백테스트 평가자는 머지 판정용(고정). 같은 walk-forward를 두 곳에서
   돌리되, **머지 권한은 외부 평가자가 후보에 재실행한 결과에만** 있다.
3. **능력 부재로 안전 보장**: no-trade는 코드 가드(루프가 제거 가능)가 아니라
   *실매매 능력을 애초에 안 붙이는 것*으로 보장한다. Corvin은 모의 전용, 브로커 쓰기
   경로 없음. 머지하니스는 diff에 주문실행 API import가 추가되면 거부한다.

### 1.1 불변식 (폐하가 명시적으로 풀지 않은 한 동결)

- **INV-1 (no-trade)**: 실매매/주문 실행 경로를 시스템에 추가하지 않는다.
  `portfolio.json`은 읽기 전용(스크린샷 파싱 외 자동 쓰기 금지).
- **INV-2 (no self-disable)**: 루프는 자신의 킬스위치·머지하니스·pinned 평가자·러너를
  수정할 수 없다(레포 밖이라 worktree에 포함되지 않음).
- **INV-3 (scope sanity)**: 머지하니스는 diff가 quant_investment_system_v2 레포를
  벗어나거나, 세이프티 관련 경로를 건드리면 거부한다.

---

## 2. 전체 아키텍처 — 2개 루프

```
┌─────────────────────────────────────────────────────────────┐
│ 🔵 Loop A — 브리핑 루프 (매일 + 이벤트 트리거)                  │
│                                                              │
│  실시간 시세(KIS/quote_provider) + 내러티브(narrative.db,     │
│  geo_signal) ── 종합 ──▶ 5엔진 → arbiter → final_actions    │
│        │                                                     │
│        ├─ 어제 콜 채점/반성 (ledger scorer 결과)              │
│        ▼                                                     │
│  brief/ 설득 빌더 ──▶ 고확신 실행 브리핑 1건 ──▶ 텔레그램 달리아봇 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🟣 Loop B — evolution 루프 (야간 배치 + 온디맨드)              │
│                                                              │
│  [레포 밖] ~/corvin_evolution/runner.sh (launchd 03:00 KST)  │
│     1. 킬스위치 체크: ~/.corvin_killswitch 있으면 즉시 종료     │
│     2. git worktree 격리 브랜치 생성                          │
│     3. Claude 서브에이전트 dispatch:                          │
│          목표 = "walk-forward 백테스트 OOS 엣지를 baseline    │
│                  대비 개선하라"                               │
│          입력 = ledger·calibration·baseline 스냅샷·백테스트결과 │
│          행동 = 가설 → 레포 코드 수정 → 테스트+in-repo 백테스트 │
│     4. [레포 밖] merge_gate.py 결정론 검증:                   │
│          ✓ 전체 테스트 통과 (현재 773)                        │
│          ✓ pinned_backtest.py 재실행: OOS 엣지 ≥ baseline+ε   │
│            AND 과적합 플래그 없음 (IS−OOS 갭 임계 이하)         │
│          ✓ 세이프티 불변식 INV-1/2/3 통과 (정적 diff 검사)     │
│          ✓ diff 범위 정상                                    │
│     5. 통과 → main 자동머지 + evolution_ledger 기록           │
│        실패 → worktree 폐기 + 사유 로그                       │
│     6. budget/iteration cap까지 반복                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 컴포넌트

### 3.1 레포 밖 — 진화 인프라 (`~/corvin_evolution/`)

| 파일 | 역할 |
|------|------|
| `runner.sh` | launchd가 호출. 킬스위치 체크 → worktree 생성 → 에이전트 dispatch → merge_gate 호출 → 머지/폐기 → 반복. |
| `merge_gate.py` | **결정론적 머지 판정자.** 후보 브랜치에 테스트+pinned 백테스트 재실행, 불변식 정적검사, 통과 시에만 `git merge`. LLM 호출 없음. |
| `pinned_backtest.py` | 고정 walk-forward 평가자. baseline과 후보를 동일 OOS 윈도우로 평가. 루프 불가침. |
| `baseline.json` | 현재 main의 OOS 엣지 기준선 스냅샷. 머지 성공 시 갱신. |
| `config.yaml` | budget cap, iteration cap, ε(개선 임계), 과적합 갭 임계, 스케줄. |
| `evolution.log` | 모든 시도·머지·폐기·사유 기록. |
| `~/.corvin_killswitch` | 존재하면 러너가 시작 거부 (수동 비상정지). |

### 3.2 레포 안 — 진화 대상 + 실험 도구 (`corvin_jarvis/evolution/`)

| 파일 | 역할 |
|------|------|
| `objective.md` | 서브에이전트에게 주는 목표·제약·금지사항(INV-1/2/3 명시) 프롬프트. |
| `backtest_harness.py` | in-repo walk-forward 실험 도구(루프가 자유 사용). pinned 평가자와 동일 로직 시작점이나, 머지 판정엔 안 쓰임. |
| `evolution_ledger.db` | 세대별 변경·엣지·머지여부 추적 (자가진화 히스토리). |
| `state/baseline_snapshot.json` | 진화 시작점 메트릭 (참조용). |

### 3.3 설득 브리핑 레이어 (`corvin_jarvis/brief/`)

`notify.py` digest를 확장해 **다각도 설득 브리핑 빌더** 신설:

1. **포지션 오버레이** — 보유분 평단 대비 손익 + 액션라벨(✂️ 축소 / ✅ 유지 / ➕ 추가존 / 👀 관찰),
   관찰 종목 재진입존. (← `feedback_brief_actionable_overlay`)
2. **확신 근거 스택** — 수치 근거 + 백테스트 엣지 + "n=X 표본·적중률 Y%".
   표본 부족 시 **'검증부족' 정직 라벨** (거짓 확신 금지, ← `meaningful_metrics`).
3. **다각 프레이밍** — bull / bear / base 3케이스 + "안 하면 기회비용"(counterfactual).
4. **투자심리 가드** — 폐하 4대 패턴(기대피로 항복매수·사건전 FOMO·드로다운 흔들림·앵커링)
   트리거 시 "룰 진입 vs 감정 반응?" 자문 1줄. (← `feedback_investor_psychology`)
5. **스캔 포맷** — 구분선·여백·상태아이콘·종목당 한 줄·통합 1메시지. (← `feedback_discord_scannable_format`)
6. **가격 신선도** — 종가/전일종가 라벨 + 시장상태 명시. (← `feedback_price_freshness_labeling`)

라우팅: **텔레그램 달리아봇만** (cron 자동 push 규칙, ← `feedback_cron_imessage_only`).

---

## 4. 목적함수 상세 (ⓓ)

```
머지 게이트 (필수, 결정론):
  pass ⟺  tests_pass
          AND  oos_edge(candidate) ≥ oos_edge(baseline) + ε
          AND  overfit_gap = is_edge − oos_edge ≤ gap_threshold
          AND  INV-1 ∧ INV-2 ∧ INV-3

보조 지표 (로그만, 초기엔 비게이팅):
  - 라이브 ledger 적중률 추세 (표본 누적 후 게이트 승격 검토)
  - 설득 성공률 = 폐하가 실제 행동한 브리핑 비율 (피드백 신호 수집)
```

- **walk-forward**: train 윈도우로 파라미터 결정 → 직후 OOS 윈도우로 평가. 롤링.
- **엣지 메트릭**: 초기엔 hit-rate 가중 수익(또는 Sharpe). config로 교체 가능하나
  *평가자는 외부 고정*이라 루프가 메트릭 정의를 못 바꾼다.
- **과적합 차단**: IS−OOS 갭이 임계 초과면 거부 (루프가 in-sample만 맞추는 진화 방지).

---

## 5. 크론 / 스케줄

| 시각(KST) | 작업 | 신규/기존 |
|-----------|------|-----------|
| 16:00 | run_digest 확장 → 설득 브리핑 (장마감 후 KR/US) | 기존 확장 |
| 이벤트 | 급변(가격·geo) 트리거 시 브리핑 | 신규 트리거 |
| 03:00 | `~/corvin_evolution/runner.sh` 야간 evolution 배치 | 신규 launchd |
| 온디맨드 | 수동 evolution 1회 실행 | 신규 |

킬스위치: `touch ~/.corvin_killswitch` 로 evolution 즉시 동결 (브리핑 루프는 무관하게 유지).

---

## 6. 단계별 구현 (무중단·additive)

- **Phase E1 — 설득 브리핑 레이어** (가치 즉시, 진화 무관): `brief/` 빌더 + digest 통합.
  TDD. → 폐하 매일 더 나은 브리핑 수령. *진화 없이도 단독 가치.*
- **Phase E2 — 진화 인프라 골격**: `~/corvin_evolution/` 러너·merge_gate·pinned_backtest·
  baseline. 킬스위치. **드라이런 모드**(머지 안 함, 후보만 평가·로그)로 먼저 검증.
- **Phase E3 — in-repo 진화 대상**: `corvin_jarvis/evolution/` objective·harness·ledger.
- **Phase E4 — 자동머지 활성화**: 드라이런 N회 안전 확인 후 실머지 on. budget cap 보수적 시작.
- **Phase E5 — 보조지표 게이트 승격**: 표본 누적 후 적중률/설득성공을 게이트로 검토.

각 Phase 독립 배포·테스트. E1은 진화 인프라 없이 단독 출시 가능.

---

## 7. 리스크 / 미해결

| 리스크 | 완화 |
|--------|------|
| 무인머지가 라이브 신호 품질 훼손 | 외부 결정론 게이트 + 드라이런 선행(E4 전) + 킬스위치 + main 백업/롤백 |
| 진화 루프 비용 폭주 | budget·iteration cap (config), 야간 1회, 데이터 일단위라 빈도 무의미 |
| 과적합 진화 | IS−OOS 갭 게이트, walk-forward OOS, pinned 평가자 |
| pinned 평가자 자체 결함 | E4 전 드라이런으로 baseline 일관성 검증, 평가자는 수동 리뷰 후 고정 |
| 루프가 우회로로 게이트 무력화 시도 | INV 정적검사 + diff scope 검사 + 외부 위치 + 정기 evolution.log 폐하 리뷰 |
| 표본 기근(현 채점 3건) | 백테스트 부트스트랩으로 검증 선행, 라이브 적중률은 보조로만 |

---

## 8. 성공 기준

- E1: 브리핑이 포지션 오버레이+근거스택+심리가드+스캔포맷 6요소를 모두 충족, 폐하 "실행가능" 인정.
- E2~E4: 드라이런에서 merge_gate가 baseline 대비 열위 후보를 100% 거부, 우위 후보만 통과.
  킬스위치로 즉시 동결 확인. INV-1/2/3 우회 시도가 정적검사에서 차단됨을 테스트로 증명.
- 전체: 무인 야간 진화가 main 무중단 유지하며, 머지된 변경이 OOS 엣지를 단조 비감소로 누적.
