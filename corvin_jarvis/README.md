# 🦅 Corvin Jarvis

자율 자산관리 에이전트. 매시간 시장+포트폴리오를 스캔하고 임계값 breach 시 자발적 알림.

## 빠른 시작

```bash
# 1회 실행 (전체 파이프라인)
bash corvin_jarvis/run_jarvis.sh

# 또는 개별 phase
python3 corvin_jarvis/pulse.py        # 데이터 ingest만
python3 corvin_jarvis/compare.py      # threshold 감지
python3 corvin_jarvis/narrate.py      # 전략 연속성 컨텍스트
python3 corvin_jarvis/jarvis.py       # 위 셋 + briefing.md 생성
python3 corvin_jarvis/discord_push.py # alert push

# 포트폴리오 업데이트 (advisory only — 실주문 X)
python3 corvin_jarvis/reconcile.py buy META 3 620.50 USD
python3 corvin_jarvis/reconcile.py sell 005930.KS 22 270500 KRW
python3 corvin_jarvis/reconcile.py refresh
```

## 파일

| 파일 | 역할 |
|---|---|
| `config.json` | alert thresholds + notification |
| `pulse.py` | 시장/원자재/FX/portfolio quote ingest |
| `compare.py` | threshold breach 감지 + severity 분류 |
| `narrate.py` | wiki 세션 기반 continuity context |
| `jarvis.py` | orchestrator (pulse + compare + narrate + briefing.md) |
| `discord_push.py` | webhook push + dedup + cooldown |
| `reconcile.py` | portfolio.json safe update |
| `run_jarvis.sh` | cron entry script |
| `state/*` | runtime state (snapshots, alerts, briefings, logs) |
| `timeseries.py` | SQLite 누적 저장 (quote_history) — Phase 5 |
| `earnings.py` | 어닝 캘린더 수집 + D-7/D-3/D-1 alert — Tier 1.3 |
| `narrative.py` | narrative-shift-detector readonly adapter — Tier 1.4 |
| `qa.py` | Q&A retrieval toolkit (RAG) — Tier 1.5 |
| `whatif.py` | 가상 거래 시뮬레이터 (advisory only) — Tier 2.1 |

## What-if 시뮬레이터 (Tier 2.1)

자연어 가상 거래 → portfolio 비중/HHI 변화 정량화. `portfolio.json`은 **건드리지 않음**.

```python
from corvin_jarvis import whatif

sim = whatif.simulate(
    "buy TSLA 5 @ 426",
    holdings,       # portfolio.json["holdings"]
    market_prices,  # {sym: price}
)
print(sim["delta"])
# {'positions': 1, 'total_value': 2130, 'top_weight_pct': -11.0, 'hhi': -0.1068}
```

명령 형식: `{buy|sell} SYMBOL SHARES [@ PRICE]` (price 없으면 market_prices에서 lookup).

지표:
- **HHI** (0~1): 0.18 미만 = unconcentrated, 0.25 이상 = highly concentrated
- **top_weight_pct**: 최대 비중 종목 비율
- **currency_mix_pct**: USD/KRW USD-equivalent 분포

## Q&A Retrieval Toolkit (Tier 1.5)

Discord 질문에 답하기 위한 retrieval-augmented context builder. Corvin LLM이 답변을 합성하기 전 raw context를 모은다.

- `recent_history_summary(symbol, days)`: timeseries.db N일 통계 (start/end/min/max/pct)
- `relative_strength(symbol, benchmark, days)`: 벤치마크 대비 상대 성과 + verdict
- `wiki_search(query, top_k)`: corvin-sessions/*.md grep + snippet
- `explain_move(symbol)`: 위 셋 + earnings + narrative 종합 dict

사용 예 ("왜 META 떨어졌어?" 답변 준비):
```python
from corvin_jarvis import qa
from pathlib import Path
out = qa.explain_move(
    Path("corvin_jarvis/state/timeseries.db"),
    symbol="META",
    days=7,
)
```

## Narrative Z-score (Tier 1.4)

`narrative-shift-detector/data/signals.db`에 **readonly**로 접근하여 KR 시장 레벨 narrative shift 계산.

- `sentiment_tone`, `foreign_net_buy`: 30일 이동 Z-score
- |Z| ≥ 1.5: medium · 2.0: high · 3.0: critical
- 별도 alert로 alerts.json에 merge

⚠️ **종목별 narrative 미지원**: signals.db는 KR 시장 통합 데이터만 보유. 종목별 narrative는 별도 source (GDELT per-ticker, FNSPID 등) 필요.

수동 사용:
```python
from corvin_jarvis import narrative
print(narrative.latest_signal(narrative.DEFAULT_SIGNALS_DB, market="KR"))
print(narrative.compute_zscore(narrative.DEFAULT_SIGNALS_DB, metric="sentiment_tone"))
```

## 어닝 캘린더 (Tier 1.3)

매 jarvis 사이클에 holdings + watchlist의 US 종목 어닝일을 yfinance에서 fetch하여 `state/timeseries.db`의 `earnings_calendar` 테이블에 upsert.

- **D-7**: severity = medium
- **D-3 / D-1**: severity = high
- 한국 종목 (숫자 코드, `.KS`, `.KQ`)은 yfinance.calendar 부적합 → 자동 skip

수동 조회:
```python
from corvin_jarvis import earnings
from datetime import date
from pathlib import Path

db = Path("corvin_jarvis/state/timeseries.db")
earnings.refresh_earnings_calendar(db, ["META", "MSFT", "NVDA"])
alerts = earnings.build_earnings_alerts(db, today=date.today())
```

## 시계열 DB

매 pulse마다 `state/timeseries.db` (SQLite)에 quote를 누적 저장 — predictive/what-if/attribution 기능의 토대.

테이블 `quote_history`:
- `category`: index / commodity / fx / portfolio / watchlist
- `symbol`, `price`, `pct_change`
- portfolio 전용: `pnl_pct`, `market_value`, `shares`
- `ts_utc`, `ts_kst`, `source`, `error`

조회 예시:
```python
from corvin_jarvis import timeseries
from pathlib import Path

rows = timeseries.read_history(
    Path("corvin_jarvis/state/timeseries.db"),
    symbol="META",
    limit=30,
)
```

## Watchlist

`config.json`의 `watchlist` 리스트에 추가한 종목은 보유 외에도 매 pulse마다 가격 추적됩니다. 한국 종목은 자동 감지하여 pykrx/FDR 사용.

예시:
```json
{
  "watchlist": ["TSLA", "AAPL", "005930"]
}
```

## 활성화 체크리스트

- [ ] **Discord webhook**: `CORVIN_DISCORD_WEBHOOK` env var 또는 config.json
- [ ] **portfolio.json 동기화**: 실제 보유와 일치 확인
- [ ] **cron 등록**: `run_jarvis.sh`를 `crontab -e` 또는 Claude Code schedule에
- [ ] **임계값 튜닝**: config.json `alert_thresholds` 폐하 선호 반영

## Cron 예시

```bash
# crontab -e
# KST 08-23, 매시간 정각
0 8-23 * * * /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_jarvis.sh

# 또는 Claude Code schedule (daily_briefing 패턴):
#   taskName: corvin-jarvis-hourly
#   cron: "0 8-23 * * *"
#   command: bash /Users/thethethe/Claude/quant_investment_system_v2/corvin_jarvis/run_jarvis.sh
```

## 정책 (CLAUDE.md 준수)

- 실제 매매 주문 **금지** — advisory only
- 모의 portfolio 업데이트는 `reconcile.py`로 명시적 실행만
- 모든 매매 기록은 `state/trades.json`에 append-only
- portfolio.json 변경 시 `state/portfolio_backups/`에 자동 백업
