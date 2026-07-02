#!/usr/bin/env bash
# Clinic Codex dev launcher — backend on :7117, frontend on :7118.
# Defensive: fails fast with clear errors if env not set up.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '[run-dev] %s\n' "$*"; }
fail() { printf '[run-dev] ERROR: %s\n' "$*" >&2; exit 1; }

is_port() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }

# --- Pre-flight: venv ---
PY="backend/.venv/bin/python"
[ -x "$PY" ] || fail "backend/.venv missing. Run: bash scripts/install.sh"
"$PY" -c "import flask, torch, mobile_sam" 2>/dev/null \
  || fail "venv incomplete. Run: bash scripts/install.sh"


# --- Pre-flight: prototypes (auto-export if missing) ---
PROTO_DERIVED="backend/codex_model/weights/prototypes.pt"
PROTO_SOURCE="backend/prototypes/prototypes.pt"
if [ ! -f "$PROTO_DERIVED" ]; then
  [ -f "$PROTO_SOURCE" ] || fail "Model artefacts missing: $PROTO_SOURCE not found. See backend/README.md."
  log "prototypes.pt missing — running bootstrap export_model with explicit runtime-write opt-in"
  PY_ABS="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
  (cd backend && "$PY_ABS" -m codex_pipeline.scripts.export_model \
    --allow-runtime-write \
    --prototypes prototypes/prototypes.pt \
    --weights-dir codex_model/weights \
    --config-template codex_model/config.json \
    --config-out codex_model/config.json) \
    || fail "export_model failed — see error above"
  [ -f "$PROTO_DERIVED" ] || fail "export_model ran but $PROTO_DERIVED still missing"
fi

# --- Pre-flight: frontend deps ---
[ -d frontend/node_modules ] || fail "frontend/node_modules missing. Run: bash scripts/install.sh"

# --- Pre-flight: ports ---
BACKEND_PORT="${BACKEND_PORT:-7117}"
FRONTEND_PORT="${FRONTEND_PORT:-7118}"
is_port "$BACKEND_PORT" || fail "BACKEND_PORT must be a TCP port number (1-65535), got: $BACKEND_PORT"
is_port "$FRONTEND_PORT" || fail "FRONTEND_PORT must be a TCP port number (1-65535), got: $FRONTEND_PORT"
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    fail "Port $port already in use. Stop other process or override BACKEND_PORT/FRONTEND_PORT env."
  fi
done
BACKEND_URL="http://localhost:$BACKEND_PORT"
FRONTEND_URL="http://localhost:$FRONTEND_PORT"
FRONTEND_API_BASE_URL="${VITE_API_BASE_URL:-$BACKEND_URL}"
BACKEND_CORS_ORIGINS="${CORS_ORIGINS:-$FRONTEND_URL,http://127.0.0.1:$FRONTEND_PORT}"

# --- Cleanup on exit ---
PIDS=()
terminate_tree() {
  local pid="$1"
  local signal="${2:-TERM}"
  local child
  if command -v pgrep >/dev/null 2>&1; then
    while read -r child; do
      [ -n "$child" ] && terminate_tree "$child" "$signal"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi
  kill "-$signal" "$pid" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM
  log "shutting down (pids: ${PIDS[*]:-none})"
  for pid in "${PIDS[@]:-}"; do terminate_tree "$pid" TERM; done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill -0 "$pid" 2>/dev/null && terminate_tree "$pid" KILL
  done
  wait 2>/dev/null || true
}
signal_exit() {
  local status="$1"
  cleanup
  exit "$status"
}
trap cleanup EXIT
trap 'signal_exit 130' INT
trap 'signal_exit 143' TERM

ensure_children_running() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -0 "$pid" 2>/dev/null || fail "Child process $pid exited before services became ready — see logs above"
  done
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local timeout_seconds="${3:-30}"
  local i
  for i in $(seq 1 "$timeout_seconds"); do
    sleep 1
    if curl -sf "$url" >/dev/null 2>&1; then
      log "$label ready: $url"
      return 0
    fi
    ensure_children_running
  done
  fail "$label did not start within ${timeout_seconds}s — see logs above"
}

# --- Launch backend (call python directly, no source activate) ---
log "starting backend on :$BACKEND_PORT"
PORT="$BACKEND_PORT" CORS_ORIGINS="$BACKEND_CORS_ORIGINS" "$PY" -m flask --app backend.wsgi run --host 127.0.0.1 --port "$BACKEND_PORT" &
PIDS+=($!)

# --- Launch frontend ---
log "starting frontend on :$FRONTEND_PORT"
(cd frontend && VITE_API_BASE_URL="$FRONTEND_API_BASE_URL" npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort) &
PIDS+=($!)

# --- Wait for services ready ---
wait_for_url "backend" "http://127.0.0.1:$BACKEND_PORT/classes"
wait_for_url "frontend" "http://127.0.0.1:$FRONTEND_PORT/"

printf '\n[run-dev] both services up.\n'
printf '[run-dev]   backend:  http://localhost:%s\n' "$BACKEND_PORT"
printf '[run-dev]   frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf '[run-dev] Ctrl-C to stop.\n\n'

wait
