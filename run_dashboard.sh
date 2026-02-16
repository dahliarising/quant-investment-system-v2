#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="${ROOT_DIR}/ui"
PORT="${1:-8000}"

if [ ! -f "${UI_DIR}/dashboard.html" ]; then
  echo "dashboard.html not found: ${UI_DIR}/dashboard.html" >&2
  exit 1
fi

echo "Starting dashboard server..."
echo "Open: http://localhost:${PORT}/dashboard.html"
cd "${UI_DIR}"
python -m http.server "${PORT}"
