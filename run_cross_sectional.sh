#!/usr/bin/env bash
# run_cross_sectional.sh — Shell-agnostic launcher for run_cross_sectional_research.py
# Works from any shell (bash, fish, zsh, etc.)
# Usage: ./run_cross_sectional.sh [args]
# Example:
#   ./run_cross_sectional.sh --universe factors --symbols TCS,INFY,RELIANCE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] venv python not found at: $PYTHON"
  echo "  Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec "$PYTHON" "${SCRIPT_DIR}/scripts/run_cross_sectional_research.py" "$@"
