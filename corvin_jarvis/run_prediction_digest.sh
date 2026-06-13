#!/bin/sh
cd "$HOME/Claude/quant_investment_system_v2" || exit 1
set -a
[ -f .env ] && . ./.env
set +a
LOG="$HOME/Claude/quant_investment_system_v2/corvin_jarvis/state/prediction_digest.log"
# 1) 데이터 신선화 — incremental(last_date 이후만 fetch). FDR 실패는 내부 swallow.
python3 -m corvin_jarvis.prediction.seed_backfill >> "$LOG" 2>&1
# 2) 다이제스트 생성·전송
exec python3 -m corvin_jarvis.prediction.run_prediction_digest >> "$LOG" 2>&1
