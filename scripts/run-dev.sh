#!/usr/bin/env bash
# Clinic Codex dev launcher — backend on :7117, frontend on :7118.
# Defensive: fails fast with clear errors if env not set up.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '[run-dev] %s\n' "$*"; }
fail() { printf '[run-dev] ERROR: %s\n' "$*" >&2; exit 1; }

SMOKE=false
SMOKE_TIMEOUT_SECONDS=30
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke)
      SMOKE=true
      shift
      ;;
    --smoke-timeout-seconds)
      [ "$#" -ge 2 ] || fail "--smoke-timeout-seconds requires a positive integer"
      [[ "$2" =~ ^[0-9]+$ ]] && [ "$2" -ge 1 ] \
        || fail "--smoke-timeout-seconds requires a positive integer, got: $2"
      SMOKE_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: bash scripts/run-dev.sh [--smoke] [--smoke-timeout-seconds N]\n'
      exit 0
      ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

is_port() { [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }
load_backend_env() {
  local env_file="backend/.env"
  local line key value
  [ -f "$env_file" ] || return 0
  log "loading backend env from $env_file"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    if [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$env_file"
}

load_backend_env

authentication_enabled() {
  case "${AUTH_REQUIRED:-true}" in
    0|false|FALSE|False|no|NO|No|off|OFF|Off|'') return 1 ;;
    *) return 0 ;;
  esac
}

if authentication_enabled && [ -z "${AUTH_SECRET_KEY:-}" ]; then
  fail "Authentication is enabled but AUTH_SECRET_KEY is missing. Run: bash scripts/install.sh"
fi
if [ -z "${AUTH_COOKIE_SECURE+x}" ]; then
  log "AUTH_COOKIE_SECURE is not configured — using false for this loopback HTTP development session"
  export AUTH_COOKIE_SECURE=false
fi

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
command -v node >/dev/null 2>&1 || fail "Need Node.js 22.22 or newer. Run: bash scripts/install.sh"
command -v npm >/dev/null 2>&1 || fail "Need npm on PATH. Run: bash scripts/install.sh"
NODE_VERSION=$(node -p 'process.versions.node' 2>/dev/null) || fail "Unable to read the Node.js version."
"$PY" - "$NODE_VERSION" <<'PY' || fail "Need Node.js 22.22 or newer. Found $NODE_VERSION."
import sys

parts = tuple(int(part) for part in sys.argv[1].split(".")[:3])
raise SystemExit(0 if parts >= (22, 22, 0) else 1)
PY

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
  local i status
  for i in $(seq 1 "$timeout_seconds"); do
    sleep 1
    status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$url" 2>/dev/null || true)"
    case "$status" in
      2??)
        log "$label ready: $url (HTTP $status)"
        return 0
        ;;
    esac
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
wait_for_url "backend and model assets" "http://127.0.0.1:$BACKEND_PORT/ready" "$SMOKE_TIMEOUT_SECONDS"
wait_for_url "frontend" "http://127.0.0.1:$FRONTEND_PORT/" "$SMOKE_TIMEOUT_SECONDS"

if [ "$SMOKE" = true ]; then
  printf '\n[run-dev] smoke PASS: backend and frontend responded successfully.\n'
  exit 0
fi

printf '\n[run-dev] both services up.\n'
printf '[run-dev]   backend:  http://localhost:%s\n' "$BACKEND_PORT"
printf '[run-dev]   frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf '[run-dev] Ctrl-C to stop.\n\n'

wait
