#!/usr/bin/env bash
# Clinic Codex installer — staged, CPU-only, WSL-safe.
# Each stage is checkpointed. If interrupted, re-run is safe (idempotent).

set -u  # do NOT enable -e; explicit error handling per stage
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\n[install] %s\n' "$*"; }
fail() { printf '\n[install] ERROR: %s\n' "$*" >&2; exit 1; }

# --- Stage 0: Pre-flight checks ---
log "Stage 0/5: pre-flight checks"

# Find Python 3.10 or 3.11 (refuse 3.12+, 3.14, system fallbacks)
PYTHON=""
for candidate in python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"; break
  fi
done
[ -n "$PYTHON" ] || fail "Need python3.10 or python3.11. Install: sudo apt install python3.11 python3.11-venv"

# Verify version (defense against shims)
PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
  3.10|3.11) ;;
  *) fail "Found $PYTHON but reports version $PYVER. Need 3.10 or 3.11." ;;
esac
log "Using $PYTHON ($PYVER)"

# Disk space check (need ~2GB)
AVAIL_KB=$(df -k . | awk 'NR==2 {print $4}')
[ "$AVAIL_KB" -gt 2000000 ] || fail "Need >=2GB free disk. Have $((AVAIL_KB/1024))MB."

# --- Stage 1: venv ---
log "Stage 1/5: venv at backend/.venv"
if [ ! -x backend/.venv/bin/python ]; then
  "$PYTHON" -m venv backend/.venv || fail "venv creation failed"
else
  log "  reusing existing venv"
fi
PIP="backend/.venv/bin/pip"
PY="backend/.venv/bin/python"

# Best-effort pip upgrade (don't fail install if this fails)
"$PIP" install --quiet --upgrade pip 2>/dev/null || log "  pip upgrade skipped (network or permission)"

# --- Stage 2: core utils (small, fast, low risk) ---
log "Stage 2/5: core utils (numpy, pillow, scipy, pandas, pyyaml, tqdm)"
"$PIP" install --no-cache-dir --prefer-binary \
  "numpy>=1.24,<2.5" "pillow>=10.0" "pyyaml>=6.0" "scipy>=1.11" "pandas>=2.0" "tqdm>=4.66" \
  || fail "Stage 2 failed — core utils"

# --- Stage 3: web (flask) ---
log "Stage 3/5: flask + flask-cors"
"$PIP" install --no-cache-dir --prefer-binary \
  "flask>=3.0,<4.0" "flask-cors>=4.0" \
  || fail "Stage 3 failed — flask"

# --- Stage 4: torch CPU (the big one — ~200MB, this is where WSL crashed before) ---
log "Stage 4/5: torch CPU wheels (~200MB download — be patient)"
"$PIP" install --no-cache-dir --prefer-binary \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.1,<2.6" "torchvision>=0.16,<0.21" \
  || fail "Stage 4 failed — torch. Re-run script to resume."

# Sanity: confirm CPU not CUDA
HAS_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "import-failed")
[ "$HAS_CUDA" = "None" ] || fail "torch installed with CUDA ($HAS_CUDA) — should be CPU. Wipe venv and retry."

# --- Stage 5: SAM family + augmentation + timm ---
log "Stage 5/5: segment-anything, mobile-sam, albumentations, timm"
# mobile-sam is not on PyPI — install from GitHub source
"$PIP" install --no-cache-dir --prefer-binary \
  "segment-anything==1.0" \
  "git+https://github.com/ChaoningZhang/MobileSAM.git" \
  "albumentations>=1.4,<2.0" "timm>=0.9" \
  || fail "Stage 5 failed — SAM/augmentation"

# --- Frontend (idempotent npm) ---
log "Frontend: npm install (idempotent)"
if [ ! -d frontend/node_modules ] || [ -z "$(ls -A frontend/node_modules 2>/dev/null)" ]; then
  (cd frontend && npm install) || fail "npm install failed"
else
  log "  frontend/node_modules present — skipping (delete it to force reinstall)"
fi

# --- Final import sanity ---
log "Sanity: importing all critical modules"
"$PY" -c "import flask, torch, torchvision, pandas, mobile_sam, segment_anything, albumentations, timm; print('all imports OK')" \
  || fail "Sanity import failed — see error above"


# --- Final: Export model prototypes (idempotent) ---
log "Exporting model prototypes (idempotent)"
PROTO_DERIVED="backend/codex_model/weights/prototypes.pt"
PROTO_SOURCE="backend/prototypes/prototypes.pt"
if [ -f "$PROTO_DERIVED" ]; then
  log "  prototypes.pt present — skipping export"
elif [ ! -f "$PROTO_SOURCE" ]; then
  fail "Model artefacts missing: $PROTO_SOURCE not found. See backend/README.md."
else
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

log "DONE. Launch with: bash scripts/run-dev.sh (Linux/macOS) or powershell -ExecutionPolicy Bypass -File .\\scripts\\run-dev.ps1 (Windows)"
