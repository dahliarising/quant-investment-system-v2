# Corvin 선행 인텔리전스 시스템 (Leading Intelligence) — 설계 문서

- **작성일**: 2026-06-01
- **상태**: 설계 승인 완료 (폐하 OK 2026-06-01 08:50 KST)
- **목적**: 가격이 이미 움직인 뒤 알리는 반응형(reactive) 알림을, 시장보다 한발 앞선 선행(leading) 인텔리전스로 전환

---

## 1. 문제 정의

### 현재 한계
Corvin Jarvis의 알림은 모두 **price-triggered** — 가격이 임계치를 넘은 *뒤에야* 발송된다.
사용자(폐하) 지적: *"이미 주식장에서 소모되어 바뀐 다음 알려주는 거라 별 필요가 없어. 한발 앞서서 예측된 값이 필요해."*

### 리서치 기반 핵심 제약 (2026-06-01 deep-research, 27소스·25주장 적대검증)

| 발견 | 검증 | 설계 영향 |
|------|------|-----------|
| **OHLCV(가격) 단독 신호는 통계적으로 알파 생성 불가** (연수익 0.55%, Sharpe 0.33, p=0.34) | 3-0 만장일치 | "가격으로 가격 예측" 폐기. 신호 다각화가 생존 조건 |
| **교차시장 선행지표 효과 주장** (한국↔미국 spillover) | 0-3 기각 | 교차시장은 advisory(참고용)로 강등 |
| **DGDNN/MATCC 등 SOTA 모델 성능 수치** | 재현불가 기각 | 복잡한 GNN/Transformer 미채택 |
| **앙상블 스크리너 패턴** (xang1234/stock-screener: 6방법론+197산업군 RS) | high, 3-0 | 주축 패턴으로 차용 |
| **다채널 알림 디스패치** (CryptoSignal notify_all) | 3-0 | 알림 아키텍처 레퍼런스 (패턴만, 코드는 미유지) |
| **KR 실시간** (pykrx-mcp, KIS WebSocket subscribe) | 확인 | KR 데이터 표준 |

### 설계 철학
> **"가격으로 가격을 예측하지 않는다."**
> 대신 **펀더멘털 성장성 + 이벤트 촉매 + 앙상블 스크리닝**을 주축으로 삼고, 가격·교차시장 신호는 보조(advisory)로만 사용한다.

---

## 2. 아키텍처 — 4 Pillar (우선순위순)

```
┌─────────────────────────────────────────────────────────┐
│  Pillar 1: 펀더멘털 성장 엔진  ⭐최우선 (근거 최강)        │
│    재무 수치 + Claude 정성 + GDELT 뉴스감정              │
│    → Growth Score 0-100 + Bull/Neutral/Bear              │
├─────────────────────────────────────────────────────────┤
│  Pillar 2: 이벤트 지평선 캘린더  ⭐선행성 최강            │
│    실적 D-3/D-1/D-0 · DART 공시 · FOMC/BOK 일정          │
│    → 순수 시간 기반 = 진짜 시장 앞섬                      │
├─────────────────────────────────────────────────────────┤
│  Pillar 3: 앙상블 스크리너  (stock-screener 패턴)        │
│    다중 방법론 병렬 + 멀티타임프레임 상대강도            │
│    → 복합 점수 (단일 신호 아님)                          │
├─────────────────────────────────────────────────────────┤
│  Pillar 4: 교차시장  (advisory only, 강등)               │
│    LMT/RTX/ITA 야간 → 012450 참고                        │
│    NQ 선물 → US 종목 참고                                 │
│    ⚠️ 항상 "참고용" 태그                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
            신뢰도 게이트 (0-100, <60 무음)
                          ↓
            다채널 디스패치 (Telegram Dahlia봇)
```

### 공통 출력 계약
모든 Pillar는 동일한 신호 객체를 반환한다:

```python
@dataclass(frozen=True)
class LeadingSignal:
    pillar: str            # "fundamental" | "event" | "ensemble" | "cross_market"
    symbol: str
    direction: str         # "bull" | "neutral" | "bear"
    confidence: float      # 0-100
    score: float | None    # pillar별 원점수 (Growth Score 등)
    horizon: str           # "intraday" | "days" | "weeks"
    advisory: bool         # True면 "참고용" 태그 강제 (Pillar 4)
    message: str           # 한국어 해석
    evidence: dict         # 근거 수치
```

이 단일 계약 덕분에 노이즈 게이트·디스패치·포맷이 Pillar에 무관하게 작동한다.

---

## 3. Pillar별 상세 설계

### Pillar 1 — 펀더멘털 성장 엔진 (`signals/fundamental.py`)

**책임**: 종목의 성장 방향성을 0-100 Growth Score로 산출.

**3개 하위 신호 통합** (사용자 요구: "넷 다"):

1. **재무 수치 (financial.py 하위 또는 함수)**
   - 소스: yfinance financials (US), pykrx/DART (KR)
   - 지표: EPS 성장률(YoY), 영업이익률 추세, 부채비율, 매출 성장률
   - → 정량 점수 0-100

2. **Claude 정성 분석**
   - 모델: `claude-haiku-4-5` (비용 최소화)
   - 입력: 최근 실적 요약 + 사업 개요 + 뉴스 헤드라인
   - 출력: 기술 방향성·경영 의사결정·산업 포지셔닝 → bull/neutral/bear + 한 줄 근거
   - ⚠️ API 호출 비용 발생 → 주간 1회 + 실적 후로 제한 (실시간 금지)

3. **GDELT 뉴스 감정**
   - 소스: 기존 geopolitical-risk MCP (`gdelt_news_sentiment`)
   - → 감정 스코어 -1~+1 → 0-100 정규화

**종합**: 가중 평균 (재무 50% / 정성 30% / 뉴스 20%) → Growth Score.
- ≥70 = Bull, 40-70 = Neutral, <40 = Bear

**트리거**: 주간 정기(일 09:00) + 실적 발표 후 즉시 + 뉴스 급변 감지 시.

---

### Pillar 2 — 이벤트 지평선 캘린더 (`signals/event_calendar.py`)

**책임**: "알려진 미래 촉매"를 가격과 무관하게 미리 경보.

**기존 `earnings.py` 확장** (이미 earnings_calendar 테이블 + D-7/3/1 로직 존재):
- 실적: 기존 로직 재사용 (D-3/D-1/D-0)
- **DART 공시 감지** (신규): 012450 등 KR 종목 공시 polling → 계약 수주·증자 등 키워드 필터
- **거시 일정** (신규): FOMC·BOK 금리결정일 하드코딩 캘린더 (분기 갱신)

**출력**: 시간 기반 신호 (방향성 없음, horizon="days"). "D-3 실적 임박" 같은 advisory.

---

### Pillar 3 — 앙상블 스크리너 (`signals/ensemble.py`)

**책임**: 단일 신호 대신 여러 방법론의 복합 점수.

**stock-screener 패턴 차용** (xang1234, 검증됨):
- 다중 방법론 병렬:
  - Minervini 트렌드 템플릿 (MA 정렬 + 52주 고저 위치)
  - 기존 DCA score (`dca_timing.py` 재사용)
  - 거래량 돌파 (Volume breakthrough)
- **멀티타임프레임 상대강도**: 1W/1M/3M/6M 각각 벤치마크 대비 RS (기존 `signals/leading.py` 확장)
- → 복합 점수 0-100

**트리거**: 프리오픈 (08:30 KST / 09:00 ET).

---

### Pillar 4 — 교차시장 (advisory only) (`signals/cross_market.py`)

**책임**: 시간대 시차 활용한 *참고용* 방향 힌트.

⚠️ **검증 근거 약함 (0-3 기각)** → `advisory=True` 강제, 항상 "참고용" 태그.

- KR 종목용: LMT/RTX/ITA 야간 마감(미국장이 KR보다 먼저 닫힘) → 012450 방향 힌트
- US 종목용: NQ/ES 선물 → META/MSFT/NVDA/TSLA 힌트
- USD/KRW 방향성 → KR 포트 전반 영향

신뢰도 산정 시 자동 30% 페널티 (검증 약함 반영).

---

## 4. 알림 스케줄 & 노이즈 제어

### 발송 스케줄

| 시간 (KST) | 트리거 | Pillar | 내용 |
|-----------|--------|--------|------|
| 08:30 | cron | P2+P3+P4 | KR 프리오픈 브리프 |
| 22:30 (09:00 ET) | cron | P2+P3+P4 | US 프리오픈 브리프 |
| 일 09:00 | cron | P1 | 주간 Growth Score 전종목 |
| 실적 발표 후 | event | P1 | 펀더멘털 재평가 |
| 상시 | 신뢰도≥60 | any | 단건 즉시 알림 |

### 노이즈 제어 (메모리 `feedback_alert_noise_aversion` 준수)
- **신뢰도 < 60 = 무음** (로그만)
- 교차시장(P4)은 항상 "참고용" 태그 + 단독 알림 금지 (브리프 내 한 줄로만)
- 정시 브리프 = 묶음 1건. 단건 즉시 알림은 신뢰도≥60만.
- 라우팅: 기존 `config.json` channels=["telegram","log_only"] 준수 (Dahlia봇 chat_id 8479910478)

---

## 5. 데이터 소스 정리

| 영역 | 소스 | 비고 |
|------|------|------|
| KR 가격/재무 | pykrx, KIS API | 메모리 `feedback_no_yfinance_kr` 준수 (yfinance 금지) |
| KR 공시 | DART API | 신규 연동 |
| US 가격/재무 | yfinance financials, quote_provider | 기존 |
| 뉴스 감정 | geopolitical-risk MCP (gdelt_*) | 기존 |
| 정성 분석 | Claude API (haiku-4-5) | 신규, 비용 제한 |
| 거시 일정 | 하드코딩 캘린더 | 분기 갱신 |

---

## 6. 모듈 구조 (기존 `corvin_jarvis/` 확장)

```
corvin_jarvis/
├── signals/
│   ├── leading_signal.py      # 신규: LeadingSignal 공통 계약 dataclass
│   ├── fundamental.py         # 신규: Pillar 1 (재무+정성+뉴스)
│   ├── event_calendar.py      # 신규: Pillar 2 (earnings.py 위에 DART+거시)
│   ├── ensemble.py            # 신규: Pillar 3 (다방법론 복합)
│   ├── cross_market.py        # 신규: Pillar 4 (advisory)
│   ├── leading.py             # 기존: RS 로직 (ensemble이 재사용)
│   └── ...
├── earnings.py                # 기존: event_calendar가 재사용
├── dca_timing.py              # 기존: ensemble이 재사용
├── predict.py                 # 기존: 보조 통계 추정 (선행 아님 명시 유지)
└── leading_orchestrator.py    # 신규: 4 Pillar 수집→게이트→디스패치
```

각 모듈은 200-400줄 목표, 단일 책임. `leading_orchestrator.py`가 Pillar들을 조율.

---

## 7. 에러 처리 & 검증

- 각 Pillar는 데이터 부족 시 신호를 만들지 않는다 (None 반환). 추측 금지 (CLAUDE.md 원칙).
- Claude API 호출 실패 = 정성 점수 제외하고 재무+뉴스로만 산출 (graceful degrade).
- DART/외부 API 타임아웃 = 해당 Pillar 스킵, 나머지 진행.
- **테스트**: 메모리 `feedback_tests_no_real_sends` 준수 — 테스트에서 실제 전송 절대 금지, conftest autouse 안전망.
- TDD: 각 Pillar의 점수 산출 로직부터 테스트 우선 작성.

---

## 8. 성공 기준

1. 프리오픈 브리프가 장 시작 *전*에 발송된다 (시간 기반 검증).
2. Growth Score가 보유 5종목(META/MSFT/NVDA/TSLA/012450)에 대해 산출된다.
3. 신뢰도 <60 신호가 무음 처리됨 (노이즈 게이트 동작).
4. 교차시장 신호가 항상 "참고용" 태그로만 노출됨.
5. 모든 알림이 Telegram Dahlia봇으로만 라우팅됨.
6. 테스트 커버리지 80%+, 실제 전송 0건.

---

## 9. 단계적 구현 순서 (제안)

리서치 우선순위 = 구현 우선순위:
1. **공통 계약** (`leading_signal.py`) — 모든 Pillar 의존
2. **Pillar 2 이벤트 캘린더** — 기존 earnings.py 확장이 가장 빠르고 선행성 최강
3. **Pillar 1 펀더멘털** — 근거 최강, 핵심 가치
4. **Pillar 3 앙상블** — 기존 leading.py/dca_timing.py 재사용
5. **오케스트레이터 + 노이즈 게이트 + 디스패치**
6. **Pillar 4 교차시장** — advisory, 마지막 (근거 약함)

각 단계는 독립 TDD 사이클.

---

## 10. 비범위 (YAGNI)

- ❌ GNN/Transformer 딥러닝 모델 (재현불가, 과설계)
- ❌ 자체 백테스트 엔진 신규 구축 (기존 quant MCP 활용)
- ❌ 실시간 WebSocket 스트리밍 (현 cron 주기로 충분, 추후 검토)
- ❌ 교차시장 신호의 정량 알파 주장 (검증 안 됨 — advisory로만)
