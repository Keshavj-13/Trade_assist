#!/usr/bin/env bash
# Start Oracle if needed, then run predictor once in foreground and exit.

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"
MAIN="$APP_DIR/main.py"

is_oracle_running() {
  if [ -z "$ORACLE_DB_DSN" ]; then
    return 0
  fi
  timeout 2 bash -c "</dev/tcp/localhost/1521" 2>/dev/null && return 0
  return 1
}

start_oracle() {
  if is_oracle_running; then
    return 0
  fi
  echo "Starting Oracle DB..."
  systemctl start oracle 2>/dev/null || systemctl start oracledb 2>/dev/null || \
  docker start $(docker ps -a | grep oracle | awk '{print $1}') 2>/dev/null || \
  /u01/app/oracle/product/*/bin/lsnrctl start 2>/dev/null
  sleep 2
}

# Only enable persistence if Oracle credentials are set
if [ -n "$ORACLE_DB_USER" ] && [ -n "$ORACLE_DB_PASSWORD" ] && [ -n "$ORACLE_DB_DSN" ]; then
  start_oracle
  export FIN_ASSIST_ENABLE_PERSISTENCE=1
else
  echo "Oracle credentials not set; running predictor without persistence"
  export FIN_ASSIST_ENABLE_PERSISTENCE=0
fi

exec "$VENV_PY" "$MAIN"
