#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_BACKEND=0
for arg in "$@"; do
    case "$arg" in
        --with-backend)
            RUN_BACKEND=1
            ;;
        *)
            ;;
    esac
done

cleanup() {
    if [[ -n "${DAEMON_PID:-}" ]]; then
        kill "$DAEMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$RUN_BACKEND" -eq 1 ]]; then
    python market_assistant.py daemon &
    DAEMON_PID=$!
fi

npm --prefix frontend_node install >/dev/null
npm --prefix frontend_node run dev
