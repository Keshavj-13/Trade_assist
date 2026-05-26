#!/usr/bin/env bash
# run_research.sh — Shell-agnostic launcher for run_strategy_research.py
# Works from any shell (bash, fish, zsh, etc.)
# Usage: ./run_research.sh [all args for run_strategy_research.py]
# Example:
#   ./run_research.sh --universe donchian --debug-strategy donchian_s1_20_10 --debug-symbol RELIANCE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] venv python not found at: $PYTHON"
  echo "  Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec "$PYTHON" "${SCRIPT_DIR}/scripts/run_strategy_research.py" "$@"
