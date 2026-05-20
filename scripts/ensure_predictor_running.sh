#!/usr/bin/env bash
# Ensure Oracle DB is running, then start predictor main.py if not already running.
# Usage: scripts/ensure_predictor_running.sh [start|stop|status]

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$APP_DIR/.venv/bin/python"
MAIN="$APP_DIR/main.py"
PIDFILE="$APP_DIR/run/predictor.pid"
LOGFILE="$APP_DIR/logs/predictor-supervisor.log"

mkdir -p "$APP_DIR/run" "$APP_DIR/logs"

is_oracle_running() {
  # Try to connect to Oracle using env vars or return 0 if not configured
  if [ -z "$ORACLE_DB_DSN" ]; then
    return 0  # Oracle config not set; assume it's ok or will be handled elsewhere
  fi
  # Try simple TCP connect to Oracle listener (port 1521 by default)
  timeout 2 bash -c "</dev/tcp/localhost/1521" 2>/dev/null && return 0
  return 1
}

start_oracle() {
  echo "Checking Oracle DB..."
  if is_oracle_running; then
    echo "Oracle DB is running"
    return 0
  fi
  echo "Oracle DB not responding; attempting to start..."
  # Try systemd service first
  if systemctl is-active --quiet oracle 2>/dev/null || systemctl is-active --quiet oracledb 2>/dev/null; then
    echo "Oracle systemd service already active"
    return 0
  fi
  if systemctl start oracle 2>/dev/null || systemctl start oracledb 2>/dev/null; then
    echo "Started Oracle DB via systemd"
    sleep 5
    return 0
  fi
  # Try Docker container
  if docker ps | grep -q oracle; then
    echo "Oracle Docker container found and running"
    return 0
  fi
  if docker ps -a | grep -q oracle; then
    echo "Starting Oracle Docker container..."
    docker start $(docker ps -a | grep oracle | awk '{print $1}') >/dev/null 2>&1
    sleep 5
    return 0
  fi
  # Try local startup script or listener
  if [ -x /u01/app/oracle/product/*/bin/lsnrctl ]; then
    echo "Starting Oracle listener..."
    /u01/app/oracle/product/*/bin/lsnrctl start >/dev/null 2>&1
    sleep 3
    return 0
  fi
  echo "Warning: could not start Oracle DB; predictor may fail if DB is required"
  return 0  # don't block predictor start if Oracle start attempt fails
}

is_running() {
  if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      return 0
    else
      rm -f "$PIDFILE"
    fi
  fi
  # fallback: check for python process with main.py in args
  pgrep -f "python.*main.py" >/dev/null 2>&1
}

start() {
  if is_running; then
    echo "Predictor already running"
    exit 0
  fi
  if [ ! -x "$VENV_PY" ]; then
    echo "Warning: venv python not found at $VENV_PY, using system python"
    VENV_PY="$(command -v python || command -v python3)"
  fi
  # Predictor-first default: persistence is OFF unless explicitly enabled
  # and complete Oracle credentials are provided.
  export FIN_ASSIST_ENABLE_PERSISTENCE="${FIN_ASSIST_ENABLE_PERSISTENCE:-0}"
  if [ "$FIN_ASSIST_ENABLE_PERSISTENCE" = "1" ]; then
    if [ -n "$ORACLE_DB_USER" ] && [ -n "$ORACLE_DB_PASSWORD" ] && [ -n "$ORACLE_DB_DSN" ]; then
      start_oracle
    else
      echo "Persistence requested but Oracle credentials are incomplete; forcing predictor-only mode."
      export FIN_ASSIST_ENABLE_PERSISTENCE=0
    fi
  fi

  nohup "$VENV_PY" "$MAIN" >>"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  echo "Started predictor (pid $(cat $PIDFILE))"
}

stop() {
  if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    if kill "$pid" 2>/dev/null; then
      echo "Stopped predictor (pid $pid)"
      rm -f "$PIDFILE"
      return 0
    fi
  fi
  echo "No predictor pidfile found or process not running"
}

status() {
  if is_running; then
    pid=$(cat "$PIDFILE" 2>/dev/null || pgrep -f "python.*main.py")
    echo "Predictor running (pid $pid)"
  else
    echo "Predictor not running"
  fi
}

case "$1" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  status)
    status
    ;;
  restart)
    stop
    sleep 1
    start
    ;;
  *)
    # default: start if not running
    start
    ;;
esac
