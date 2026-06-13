#!/bin/sh
cd "$HOME/Claude/quant_investment_system_v2" || exit 1
set -a
[ -f .env ] && . ./.env
set +a
exec python3 -m corvin_jarvis.prediction.run_prediction_digest >> "$HOME/Claude/quant_investment_system_v2/corvin_jarvis/state/prediction_digest.log" 2>&1
