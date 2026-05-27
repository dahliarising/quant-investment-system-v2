---
type: design-spec
domain: [investment]
project: corvin_jarvis
created: 2026-05-27
updated: 2026-05-27
status: approved
tags: [alert, signal-detection, sector-basket, leading-indicator, cross-market, hybrid-timing]
confidence: high
---

# Proactive Signal Detection — Design Spec

## 1. Problem Statement

2026-05-27 한국·미국 반도체가 동반 급등(삼성전자 +7.02%, SK하이닉스 +10.23%, 美 반도체 강세)했으나
Corvin alert 시스템이 이를 사용자에게 알리지 못했다. 근본 원인은 버그가 아니라 **설계상 관측 해상도의 한계**다.

현재 alert 엔진(`compare.py`)이 신호를 생성하는 단위는 두 가지뿐:

1. **매크로 집계** — 지수(KOSPI/KOSDAQ/S&P/NASDAQ), VIX, 원자재, FX
2. **내 보유 포지션** — `portfolio.json` holdings (META/MSFT/NVDA)

이로 인한 구조적 사각지대:

- **G1 — 지수 희석(dilution)**: 삼성 +7%·하이닉스 +10%인데 KOSPI는 +2.55%. 지수만 보면 섹터 급등 신호가 평균에 묻혀 복원 불가능.
- **G2 — 보유 종목에만 개별 alert**: NVDA(보유)는 +16.49%로 포착됐지만 비보유 반도체(삼성/하이닉스/AVGO/MU/TSM)는 정의상 사각지대.
- **G3 — 임계값이 "자산군" 단위**: config의 `alert_thresholds`는 index/commodity/fx/portfolio 카테고리에만 존재. "임의 종목이 N% 움직이면" 같은 일반 규칙이 없음.
- **G4 — universe(매수후보)와 alert(감시)의 분리**: `universe.json`(DCA 33종)과 alert 대상이 별개. 정보 사일로.
- **G5 — watchlist 미완성**: `pulse.fetch_watchlist`가 watchlist(삼성/하이닉스 포함) 시세를 fetch해 snapshot에 저장하지만, `compare.py`에 `check_watchlist`가 없어 **fetch만 하고 버려진다.**

운영 요인(별개): 2026-05-27 07:52 KST에 hourly alert cron(`run_jarvis.sh`)을 사용자 요청으로 비활성화 →
생성된 alert(KOSPI/NVDA)조차 push되지 않음.

## 2. Goals / Non-Goals

### Goals
- 지수를 견인하는 대표 대형주(韓+美)의 개별·섹터 급등락을 **사용자에게 미리/빠르게** 알린다.
- 비보유 종목도 감시 대상에 포함 (보유 여부와 alert 분리).
- 섹터 단위 신호로 "지수 희석" 문제를 복원.
- 선행 신호(RS·모멘텀)로 가격 폭발 *전* 주의 환기.
- 교차시장(美→韓) lead-lag는 **참고 전용** 별도 채널로 제공.

### Non-Goals
- 실제 매매 주문 (advisory only 유지, CLAUDE.md).
- alert을 DCA 매수 트리거로 사용 (DCA는 완성봉·장마감 기준 유지, 책임 분리).
- 전체 시장 종목 스캔 (지수 헤비웨이트 ~20종으로 한정, 노이즈 통제).
- 시총 가중 바스켓 (데이터 부재 → 동일가중으로 시작, 후순위 개선).

## 3. Requirements

### 3.1 Detection Levels
- **A — Reactive-fast**: 감시 종목이 ±X% 움직이면 alert. 섹터 바스켓 집계 alert 포함.
- **B1 — Leading (즉시 가능)**: 상대강도 RS(종목 pct − 지수 pct), 모멘텀(다일 누적, `timeseries.db`).
- **B2 — Leading (데이터 확장 후)**: 거래량 급증, 52주 신고가 근접. → `quote_provider` 확장 필요.
- **C — Cross-market lead-lag (참고 전용)**: 美 섹터 종가 → 익일 韓 대응 종목 flag + 촉매(narrative.db/지정학 MCP). 메인 push 없음.

### 3.2 Scope (감시 universe)
지수를 움직이는 대표 대형주 ~20종 (조정 가능):
- 韓: 삼성전자(005930), SK하이닉스(000660), LG에너지솔루션(373220), 삼성바이오로직스(207940), 현대차(005380), 기아(000270), NAVER(035420), 카카오(035720), 셀트리온(068270), POSCO홀딩스(005490), HD현대중공업(329180), 한화에어로스페이스(012450)
- 美: AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, LLY, JPM

### 3.3 Timing — Hybrid (③)
- 장중: `🟡 잠정(미확정)` — 라이브/미완성봉 기준 빠른 heads-up.
- 장마감 후: `✅ 확정` — 완성봉 기준 재확인.
- dedup 키에 `phase`(provisional/confirmed) 포함 → 잠정→확정 중복 핑 방지, 확정이 잠정을 승격/종료.
- **불변식**: alert은 "주의 환기"이지 "행동 강제"가 아니다. 잠정 신호도 라벨이 정확하면 가치 있음. 실제 매수(DCA)는 완성봉 유지.

## 4. Architecture

### 4.1 Pipeline 삽입
```
pulse → signals(A·B) → compare(기존 매크로/보유) → geo_signal → narrate → briefing
                          │
                          └─ A·B alert → alerts.json 병합 → notify(하이브리드 라벨 push)
        cross_market(C) ───→ state/reference_signals.json → briefing "📎 참고" 섹션 (push 없음)
```

### 4.2 신규 패키지 `corvin_jarvis/signals/`
| 모듈 | 책임 | 입력 | 출력 | 의존 |
|---|---|---|---|---|
| `monitored_universe.json` | 감시 종목 + sector 태그 (단일 소스) | — | 종목 메타 | — |
| `universe_monitor.py` (A) | 종목별 임계 + 섹터 바스켓 집계 | latest.json, config | Alert[] | compare.Alert, config |
| `leading.py` (B) | RS·모멘텀 [B1] / 거래량·신고가 [B2] | latest.json, timeseries.db, config | Alert[] | timeseries |
| `cross_market.py` (C) | 美→韓 lead-lag + 촉매 (참고) | latest.json, narrative.db, geo MCP | ReferenceSignal[] | narrative.db |
| `detector.py` | 오케스트레이터: universe 로드 → A·B 병합, C는 참고채널 | 위 전부 | alerts.json append, reference_signals.json | 위 전부 |

### 4.3 핵심 수정 (기존 코드)
- `pulse.py`: `fetch_watchlist(load_watchlist())` → `fetch_universe(load_universe())` 로 일반화.
  `monitored_universe.json`을 읽어 모든 감시 종목 시세를 snapshot에 저장. (G5 근본 해결)
- `compare.py`: 변경 최소화. 매크로/포트폴리오 검사는 그대로. detector가 compare 결과 + signals 결과를 병합.
- `jarvis.py`: 파이프라인에 detector 단계 추가.
- `notify.py`: 하이브리드 phase 라벨 + phase-aware dedup 키. C(reference)는 push 대상에서 제외.

## 5. Data Model

### 5.1 `monitored_universe.json`
```json
{
  "_comment": "지수 견인 대형주 감시 유니버스 — DCA universe.json과 별개, alert 전용",
  "tickers": [
    {"symbol": "005930", "market": "KR", "sector": "semiconductor", "name": "삼성전자"},
    {"symbol": "000660", "market": "KR", "sector": "semiconductor", "name": "SK하이닉스"},
    {"symbol": "NVDA",   "market": "US", "sector": "semiconductor", "name": "NVIDIA"}
  ]
}
```
sector 값: semiconductor, battery, bio, auto, internet, steel, shipbuilding, defense, bigtech, finance, pharma 등.

### 5.2 config.json 추가 (`alert_thresholds`)
```json
"stock_pct_change": {"default": 5.0, "overrides": {"005930": 4.0}},
"sector_basket_pct": {"semiconductor": 3.0, "shipbuilding": 4.0, "defense": 4.0, "_default": 4.0},
"leading": {"rs_min_pct": 2.0, "momentum_days": 3, "momentum_min_pct": 8.0}
```

### 5.3 Alert phase / dedup
- 기존 Alert에 `phase: "provisional" | "confirmed"` 필드 추가.
- dedup 키: `category::metric::phase` (기존 severity 대신 phase). 잠정은 cooldown 짧게, 확정은 1일 1회.

### 5.4 `state/reference_signals.json` (C)
```json
{"generated_at": "...", "signals": [
  {"kr_symbol": "000660", "us_driver": "semiconductor", "us_move_pct": 4.2,
   "catalyst": "...", "note": "美 반도체 야간 강세 → 韓 반도체 개장 주목 (참고)"}
]}
```

## 6. Sector Basket 계산
- 바스켓 % = 해당 sector 구성 종목 등락률 **동일가중 평균**.
- 예: 삼성 +7.02%, 하이닉스 +10.23% → semiconductor 바스켓 ≈ +8.6% → 임계 3% 초과 → alert.
- **검증 케이스**: 이 값이 KOSPI +2.55%에 묻히던 신호를 복원 (G1 해결).
- 시총 가중은 데이터 부재로 후순위. 동일가중의 한계(소형주 왜곡)는 universe를 대형주로 한정해 완화.

## 7. Severity
기존 `compare._classify_severity(value, threshold)` 재사용. 임계 대비 배수:
- 1.0x ~ 1.5x → medium
- 1.5x ~ 2.0x → high
- ≥ 2.0x → critical

## 8. Rollout (단계적)
1. **A** — universe_monitor (종목별 + 섹터 바스켓). 즉효·저위험. 오늘 케이스 즉시 포착.
2. **B1** — leading (RS + 모멘텀). 기존 데이터로 가능.
3. **B2** — leading (거래량 + 52주 신고가). `quote_provider` volume/historical 확장 선행.
4. **C** — cross_market (참고 전용). narrative.db / 지정학 MCP 연동.

각 단계는 독립 PR + TDD.

## 9. Testing (TDD, 80%+)
- **unit**: 임계 분류, RS 계산, 모멘텀 계산, 섹터 바스켓 집계, phase dedup, 하이브리드 라벨 할당.
- **integration**: 2026-05-27 09:13 스냅샷(삼성 +7.02 / 하이닉스 +10.23)을 fixture로 → semiconductor 바스켓 alert + 개별 종목 alert 발화 검증.
- **regression**: 012450식 장중 미완성봉 → `🟡 잠정`만 생성되고 `✅ 확정`은 생성되지 않음 보장 ([[feedback_dca_signal_timing]]).
- Korean 종목/지수 데이터는 pykrx + FinanceDataReader만 (yfinance 금지, [[feedback_no_yfinance_kr]]).

## 10. Open Questions / Risks
- **R1 (데이터)**: `quote_provider.Quote`에 volume/historical high 부재 → B2는 데이터 레이어 확장 필요. B2 이전에 quote_provider 확장 범위를 별도 설계.
- **R2 (universe 크기)**: ~20종 초안. 너무 많으면 노이즈, 적으면 사각. 운영하며 가감.
- **R3 (KR 장중 데이터 신뢰성)**: 개장 직후 미완성봉의 변동성. 하이브리드 잠정 라벨로 완화하되, 잠정 임계를 확정보다 보수적으로 둘지 검토.
- **R4 (cron)**: 비활성화된 hourly cron 재가동 여부는 별도 결정 (이 spec 범위 밖, 사용자 결정 대기).
