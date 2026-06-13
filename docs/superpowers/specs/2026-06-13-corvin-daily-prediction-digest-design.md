# Corvin 매일 예측 다이제스트 — 설계서 (Phase 1: 시스템 1~5)

- **created**: 2026-06-13
- **status**: design approved (사용자 "진행해" 2026-06-13), spec 검토 대기
- **owner**: Corvin
- **phase**: Phase 1 of 2 (Phase 2 = ML 계열 6~10, 별도 spec)

---

## 1. 목적 (Why)

매일 한국장 개장 직후, **여러 예측 방법론의 결과를 한 통의 텔레그램 다이제스트**로 받는다.
단일 지표가 아닌 **다각도 예측**(통계·확률·모멘텀·지정학·벡터 analog)을 교차로 보고
"오늘 시장/보유종목이 과거 어떤 국면과 닮았고, 어디가 위험/기회인가"를 한눈에 파악한다.

핵심 제약(Corvin 철칙):
- **추측 금지** — 데이터 부족 모듈은 가짜 % 대신 "보류" 표기
- **의미있는 지표** — confidence + 근거수치 동반, 거짓 정밀도 금지
- **실주문 0** — 전부 advisory only

## 2. 범위 (Scope)

**대상 universe** (사용자 선택 ⓑ):
- 개별종목 예측(시스템 1·2) = **보유종목 + 감시 대형주**
  - 보유: `portfolio.json` (런타임 read, 가정 금지)
  - 감시: `corvin_jarvis/monitored_universe.json` (삼성·하이닉스·LG엔솔 등)
- 시장 레벨 예측(시스템 4·5) = 지수/매크로 전체 (항상 포함)

**In scope**: 시스템 1~5 + 데이터 backfill + 다이제스트 조립 + 09:36 KST launchd + 텔레그램 전송
**Out of scope (Phase 2)**: 6 ML분류기, 7 시계열(Prophet/LSTM), 8 센티먼트, 9 몬테카를로, 10 앙상블

## 3. 아키텍처

```
                         ┌──────────────────────────┐
 launchd 09:36 KST  ──▶  │  run_prediction_digest.py │  (오케스트레이터)
                         └────────────┬─────────────┘
                                      │ 1) 증분 backfill 갱신
                         ┌────────────▼─────────────┐
                         │   daily_history (SQLite)  │  ← FinanceDataReader 5~10yr 일봉
                         └────────────┬─────────────┘
                                      │ 2) 5개 모듈 호출 (동일 출력 계약)
        ┌───────────┬───────────┬────┴──────┬───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
    1 VELOCITY  2 확률      3 모멘텀     4 지정학    5 벡터analog
   (engine)    (predict)  (backtest)   (geo MCP)   (신규 모듈)
        └───────────┴───────────┴────┬──────┴───────────┴───────────┘
                                      │ 3) PredictionResult[] 수집
                         ┌────────────▼─────────────┐
                         │   digest_assembler.py     │  스캔가능 5섹션 1통
                         └────────────┬─────────────┘
                                      │ 4) channels.send_telegram (달리아봇 8479910478)
                                      ▼
                                  Telegram
```

## 4. 컴포넌트 설계

### 4.1 데이터 레이어 — `prediction/backfill.py`
- 새 SQLite 테이블 **`daily_history`** (기존 intraday `quote_history`와 분리)
  - schema: `(symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, source TEXT, PRIMARY KEY(symbol, date))`
- **소스 규칙** (memory: KR은 yfinance 금지):
  - KR 종목/지수 → `FinanceDataReader` + `pykrx` (`scripts/kr_data.py` 재사용)
  - US 지수/원자재/환율 → `FinanceDataReader`
- **초기 backfill**: 5~10년 일봉 1회 적재 (지수·VIX·환율·원자재 + universe 종목)
- **증분 갱신**: 매 실행 시 마지막 date 이후만 fetch (idempotent, UPSERT)
- 함수: `backfill_symbol(db, symbol, market, years)`, `incremental_update(db, symbols)`, `read_daily(db, symbol, lookback)`

### 4.2 출력 계약 — `prediction/contract.py`
모든 모듈이 반환하는 단일 dataclass:
```python
@dataclass(frozen=True)
class PredictionResult:
    system: str          # "velocity" | "probability" | "momentum" | "geopolitical" | "vector_analog"
    scope: str           # "market" | symbol
    verdict: str         # 한 줄 결론 (한국어)
    confidence: float    # 0-100
    evidence: dict       # 근거 수치 (모듈별 자유 schema)
    data_ok: bool        # False면 데이터 부족 → 다이제스트에서 "보류" 표기
```
`data_ok=False` 강제 규칙: lookback 데이터가 모듈 최소요건 미만이면 무조건 False + verdict="데이터 부족·보류".

### 4.3 예측 모듈 5개 (각각 순수함수 + 어댑터)
어댑터가 `daily_history`/MCP에서 입력을 모아 기존 순수함수에 주입하고 `PredictionResult`로 변환.

| # | system | 재사용 | 어댑터 | 최소 데이터 |
|---|--------|--------|--------|------------|
| 1 | velocity | `predictive_engine.evaluate_velocity` | `prediction/m_velocity.py` | ≥10 거래일 |
| 2 | probability | `predict.probability_below` + `log_returns` | `prediction/m_probability.py` | ≥20 거래일 |
| 3 | momentum | `scripts/backtest_momentum.py` 로직 | `prediction/m_momentum.py` | ≥120 거래일 |
| 4 | geopolitical | Geopolitical MCP (`gdelt_risk_score` 등) | `prediction/m_geopolitical.py` | MCP 응답 |
| 5 | vector_analog | **신규** | `prediction/m_vector.py` | ≥250 거래일 |

### 4.4 벡터 analog 모듈 — `prediction/m_vector.py` (신규 핵심)
- **feature 벡터**: 각 거래일 = `[kospi_ret, nasdaq_ret, vix_level, usd_krw_ret, gold_ret, copper_ret, dxy_ret, ...]` (정규화: z-score)
- **유사도**: 오늘 벡터 vs 과거 모든 날 → 코사인 유사도 (numpy, 외부 의존 없음)
- **예측**: 상위 K개 analog 날의 **forward N일 수익률 분포**(평균·중앙·승률) → verdict
- **정직성**: analog 표본 < K_min 또는 유사도 < 임계면 `data_ok=False`
- 순수함수: `build_feature_matrix(daily)`, `top_k_analogs(today_vec, matrix, k)`, `forward_distribution(analogs, horizon)`

### 4.5 다이제스트 조립 — `prediction/digest_assembler.py`
- 입력: `list[PredictionResult]` → 출력: 텔레그램 Markdown 문자열 1통
- **스캔가능 포맷** (memory): 구분선·여백·상태아이콘·종목당 한 줄
- 섹션 순서: ① 시장 방향(4·5) → ② 보유종목 리스크(1·2) → ③ 감시종목 → ④ 모멘텀 신호
- `data_ok=False` 모듈은 "⏸ 보류(데이터 부족)" 한 줄로 축약
- **헤더 라벨**: "📅 YYYY-MM-DD 장초반 스냅샷 (09:36 KST)" — 미완성봉 명시 (memory: 가격 시점 라벨링)

### 4.6 오케스트레이터 — `prediction/run_prediction_digest.py`
1. 증분 backfill → 2. portfolio.json + monitored_universe read → 3. 5모듈 호출(예외격리: 한 모듈 실패해도 나머지 진행) → 4. assemble → 5. `channels.send_telegram(digest)`
- `--dry-run`: 전송 대신 stdout 출력 (테스트/검증용)

### 4.7 스케줄러 — `com.corvin.prediction-digest.plist`
- launchd, 매일 **09:36 KST** 1회
- `portfolio-watchdog.plist` 패턴 복제
- 전송 라우팅: config `notification.channels=['telegram','log_only']` (기존), `channels.send_telegram` → chat_id `8479910478` (달리아봇)

## 5. 데이터 흐름 (예: 5번 벡터)
`daily_history` → `build_feature_matrix` → 오늘 벡터 추출 → `top_k_analogs` (cosine) → `forward_distribution` → `PredictionResult(system="vector_analog", scope="market", verdict="현재 국면과 87% 닮은 과거 12개 중 다음 5일 평균 +1.8%, 승률 67%", confidence=…, data_ok=True)`

## 6. 에러 처리
- 모듈별 try/except 격리 — 한 모듈 예외 → 해당 섹션 "⚠️ 오류·생략", 나머지 다이제스트 정상 전송
- backfill 네트워크 실패 → 기존 `daily_history` 캐시로 진행 + "데이터 갱신 실패" 경고 1줄
- 텔레그램 전송 실패 → `log_only` fallback + 재시도 1회 (channels.py 기존 동작)

## 7. 테스트 (memory: 실제 전송 0)
- **단위**: 각 순수함수 — 주입식 fixture로 검증 (특히 5번: 알려진 행렬로 cosine/분포 정확성)
- **계약**: 모든 모듈이 `PredictionResult` 반환 + `data_ok` 게이트 동작 (데이터 부족 시 보류)
- **조립**: 5개 result → 다이제스트 문자열 스냅샷 (스캔가능 포맷 검증)
- **안전망**: conftest autouse — 실제 텔레그램/네트워크 전송 차단 (memory: tests_no_real_sends)
- backfill은 fdr 호출 mock

## 8. 마일스톤 (구현 순서)
1. `daily_history` 스키마 + `backfill.py` (초기 적재 + 증분)
2. `contract.py` (PredictionResult + data_ok 게이트)
3. 어댑터 1~4 (기존 로직 재사용, 빠름)
4. **5 벡터 모듈** (신규, TDD)
5. `digest_assembler.py` (스캔 포맷)
6. `run_prediction_digest.py` + `--dry-run` 검증
7. launchd plist + 설치 + dry-run 전송 1회 확인 → 폐하 승인 후 실가동

## 9. 미해결/가정
- 벡터 feature 구성 최종 목록은 구현 중 backfill 가능 심볼로 확정 (없는 심볼 제외)
- analog horizon N·K 기본값(예: N=5, K=12)은 구현 후 간이 백테스트로 튜닝 (Phase 2 auto-tuner와 연계 가능)
- 09:36 장초반 미완성봉 — "스냅샷" 라벨로 한계 명시, 정밀 신호 아님

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
