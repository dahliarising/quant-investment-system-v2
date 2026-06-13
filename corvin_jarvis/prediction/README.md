# prediction/ — 매일 예측 다이제스트 (시스템 1~10)

매일 09:36 KST → 텔레그램 달리아봇으로 예측 방법론 다이제스트 전송.

## 모듈
- contract.py — PredictionResult 출력 계약 (data_ok 게이트)
- backfill.py — daily_history (FDR 일봉)
- **Phase 1 (1~5)**: m_velocity / m_probability / m_momentum / m_geopolitical / m_vector
- **Phase 2 (6~10)**: m_logistic / m_band / m_sentiment / m_montecarlo / m_ensemble (numpy·scipy 전용, 의존성 0)
- backtest.py — Phase 2 walk-forward 검증 게이트 (**통과분만 합류**)
- digest_assembler.py — 스캔 포맷 조립 (종목당 한 줄 병합)
- run_prediction_digest.py — 오케스트레이터 (--dry-run)

## Phase 2 백테스트 게이트 (안전 척추)
각 방법론을 daily_history로 walk-forward 검증 → `state/phase2_backtest.json`. 오케스트레이터가 읽어 **통과 모델만** 다이제스트에 표시. 미통과는 조용히 제외(가짜 정밀도 금지).
- 방향계(logistic): directional hit-rate > 다수클래스 baseline + margin
- 분포/밴드계(montecarlo·band): 실현값이 예측구간에 든 coverage ≈ target ±tol
- 앙상블: ≥2표일 때만 '합의' (1표는 과신)
- 게이트 생성: `python3 -m corvin_jarvis.prediction.seed_phase2_backtest`

> 2026-06-13 실측: logistic 탈락(hit 0.568 < base 0.580), band·montecarlo 통과.
> geo/sentiment 페이로드는 운영 진입점에서 MCP fetcher 주입 필요(Phase 1과 동일 패턴).

## 실행
- 초기 적재: `python3 -m corvin_jarvis.prediction.seed_backfill`
- 수동 미리보기: `python3 -m corvin_jarvis.prediction.run_prediction_digest --dry-run`
- 스케줄 설치(폐하 승인 후): plist를 ~/Library/LaunchAgents에 복사 후 launchctl load

  ```bash
  sed "s|__HOME__|$HOME|g" corvin_jarvis/com.corvin.prediction-digest.plist > ~/Library/LaunchAgents/com.corvin.prediction-digest.plist
  launchctl load ~/Library/LaunchAgents/com.corvin.prediction-digest.plist
  ```

## FDR 심볼 매핑 (2026-06-13 실측 검증)
`seed_backfill._FDR_SYMBOL` — 내부 feature 키 → FinanceDataReader 심볼. 9개 코어 feature 모두 ~10년(2016~) 일봉 확보:

| feature | FDR 심볼 | 비고 |
|---------|---------|------|
| kospi | KS11 | |
| kosdaq | KQ11 | |
| nasdaq | IXIC | |
| sp500 | US500 | |
| vix | VIX | Volume=NaN 행 존재 → `_num()`으로 정제 |
| usd_krw | USD/KRW | Volume=NaN |
| gold | GC=F | 플랜 추정 `ZG=F`는 404 → **GC=F로 교정** |
| copper | HG=F | Volume=NaN |
| dxy | DX-Y.NYB | Volume=NaN |

> 지수·FX·선물은 Volume 컬럼에 NaN이 섞여 있어 `int(NaN)` 폭발 방지용 `_num()` 헬퍼로 정제한다.

## Phase 3 (예정)
auto-tuner (horizon/k/임계값 튜닝) · geo·sentiment MCP 운영 fetcher 배선 · band 종목별 가격타깃.
