# prediction/ — 매일 예측 다이제스트 (Phase 1: 시스템 1~5)

매일 09:36 KST → 텔레그램 달리아봇으로 5개 예측 방법론 다이제스트 전송.

## 모듈
- contract.py — PredictionResult 출력 계약 (data_ok 게이트)
- backfill.py — daily_history (FDR 일봉)
- m_velocity / m_probability / m_momentum / m_geopolitical / m_vector — 5개 예측
- digest_assembler.py — 스캔 포맷 조립
- run_prediction_digest.py — 오케스트레이터 (--dry-run)

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

## Phase 2 (예정)
6 ML분류기 · 7 시계열 · 8 센티먼트 · 9 몬테카를로 · 10 앙상블 — 백테스트 검증 통과분만 다이제스트 합류.
