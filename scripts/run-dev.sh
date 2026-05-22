#!/usr/bin/env bash
# Clinic Codex dev launcher — backend on :7117, frontend on :7118.
# Defensive: fails fast with clear errors if env not set up.

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '[run-dev] %s\n' "$*"; }
fail() { printf '[run-dev] ERROR: %s\n' "$*" >&2; exit 1; }

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
  log "prototypes.pt missing — running export_model"
  PY_ABS="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
  (cd backend && "$PY_ABS" -m codex_pipeline.scripts.export_model) \
    || fail "export_model failed — see error above"
  [ -f "$PROTO_DERIVED" ] || fail "export_model ran but $PROTO_DERIVED still missing"
fi

# --- Pre-flight: frontend deps ---
[ -d frontend/node_modules ] || fail "frontend/node_modules missing. Run: bash scripts/install.sh"

# --- Pre-flight: ports ---
BACKEND_PORT="${BACKEND_PORT:-7117}"
FRONTEND_PORT="${FRONTEND_PORT:-7118}"
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    fail "Port $port already in use. Stop other process or override BACKEND_PORT/FRONTEND_PORT env."
  fi
done

# --- Cleanup on exit ---
PIDS=()
cleanup() {
  log "shutting down (pids: ${PIDS[*]:-none})"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Launch backend (call python directly, no source activate) ---
log "starting backend on :$BACKEND_PORT"
PORT="$BACKEND_PORT" "$PY" -m flask --app backend.wsgi run --host 0.0.0.0 --port "$BACKEND_PORT" &
PIDS+=($!)

# --- Launch frontend ---
log "starting frontend on :$FRONTEND_PORT"
(cd frontend && PORT="$FRONTEND_PORT" npm run dev) &
PIDS+=($!)

# --- Wait for backend ready ---
for i in $(seq 1 30); do
  sleep 1
  if curl -sf "http://127.0.0.1:$BACKEND_PORT/classes" >/dev/null 2>&1; then
    log "backend ready: http://localhost:$BACKEND_PORT"
    break
  fi
  [ "$i" = 30 ] && fail "backend did not start within 30s — see logs above"
done

printf '\n[run-dev] both services up.\n'
printf '[run-dev]   backend:  http://localhost:%s\n' "$BACKEND_PORT"
printf '[run-dev]   frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf '[run-dev] Ctrl-C to stop.\n\n'

wait
