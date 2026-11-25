#!/usr/bin/env bash
# One-command background service control for the MuJoCo WASM checkpoint preview.
#   ./start.sh            start (background, tmux when available)
#   ./start.sh status     show status
#   ./start.sh restart    stop + start
#   ./start.sh stop       stop
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

SESSION="${SESSION:-mujoco_preview}"
PORT="${PORT:-8099}"
HOST="${HOST:-0.0.0.0}"
PYTHON="${PYTHON:-/home/ubuntu/.local/share/mamba/envs/instinctlab/bin/python}"
LOG="$APP_DIR/server.log"
URL="http://127.0.0.1:${PORT}/"
PID_FILE="$APP_DIR/server.pid"

usage() {
  echo "usage: $0 [start|stop|restart|status]"
}

is_healthy() {
  curl -fsS --max-time 1 "${URL}api/info" >/dev/null 2>&1
}

session_running() {
  command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null
}

server_pid() {
  if session_running; then
    tmux list-panes -t "$SESSION" -F '#{pane_pid}' 2>/dev/null | head -1
  elif [ -f "$PID_FILE" ]; then
    cat "$PID_FILE"
  fi
}

print_status() {
  if is_healthy; then
    echo "status:  running"
    echo "url:     $URL"
    if session_running; then
      echo "tmux:    $SESSION  (attach: tmux attach -t $SESSION)"
    fi
    local pid
    pid="$(server_pid || true)"
    [ -n "$pid" ] && echo "pid:     $pid"
    echo "log:     tail -f $LOG"
    return 0
  fi
  echo "status:  not running (port $PORT)"
  return 1
}

start() {
  command -v node >/dev/null 2>&1 || { echo "error: node not found"; exit 1; }
  [ -f "$APP_DIR/server.mjs" ] || { echo "error: server.mjs not found in $APP_DIR"; exit 1; }
  [ -d "$APP_DIR/node_modules" ] || echo "warning: node_modules missing; run 'npm install' if boot fails"
  [ -x "$PYTHON" ] || echo "warning: exporter python not found at $PYTHON (checkpoint export will fail)"
  [ -f "$APP_DIR/public/config.json" ] || echo "warning: public/config.json missing; run prepare.py to create the default export"

  mkdir -p "$APP_DIR/uploads" "$APP_DIR/public/exports" "$APP_DIR/export-logs"

  if is_healthy; then
    echo "already running."
    print_status
    return 0
  fi
  if session_running; then
    echo "tmux session '$SESSION' exists but the HTTP check failed; replacing it."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.3
  fi

  echo "starting preview on port $PORT ..."
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$SESSION" -n server \
      "cd '$APP_DIR' && PORT='$PORT' HOST='$HOST' PYTHON='$PYTHON' node server.mjs >> '$LOG' 2>&1"
  else
    PORT="$PORT" HOST="$HOST" PYTHON="$PYTHON" nohup node server.mjs >> "$LOG" 2>&1 &
    echo $! > "$PID_FILE"
  fi

  for _ in $(seq 1 40); do
    if is_healthy; then
      echo "started."
      print_status
      return 0
    fi
    sleep 0.25
  done

  echo "error: server did not become healthy within 10 s"
  echo "---- last log lines ----"
  tail -40 "$LOG" 2>/dev/null || true
  exit 1
}

stop() {
  local stopped=0
  if session_running; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "stopped tmux session '$SESSION'"
    stopped=1
  fi
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "stopped pid $pid"
    stopped=1
  fi
  for _ in $(seq 1 20); do
    is_healthy || break
    sleep 0.2
  done
  if is_healthy; then
    echo "warning: $URL is still responding; the server may have been started outside this script."
    echo "         inspect with: ss -ltnp | grep $PORT"
    exit 1
  fi
  [ "$stopped" -eq 0 ] && echo "not running."
  return 0
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop || true; sleep 0.4; start ;;
  status)  print_status ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
