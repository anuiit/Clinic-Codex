#!/usr/bin/env bash
# CPU E2E smoke for deterministic projection-head training and candidate export.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
PIPELINE="$BACKEND_DIR/codex_pipeline/scripts"
PYTHON="${PYTHON:-$BACKEND_DIR/.venv/bin/python3}"
CONFIG="$BACKEND_DIR/codex_pipeline/config/baseline-smoke.yaml"
FEATURES="$BACKEND_DIR/model_registry/versions/20260711T220000Z-external286-weak-v3/training_data/precomputed/features.pt"
OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_baseline_recipe.sh [--features <features.pt>] [--output-root <dir>]

Runs train -> evaluate/prototype export -> runtime candidate export twice on the
trusted cached-DINO snapshot and compares artifacts byte-for-byte. The full
baseline launcher additionally exercises metadata and DINO feature precompute.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --features) FEATURES="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$FEATURES" ]] || { echo "ERROR: features file not found: $FEATURES" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: smoke config not found: $CONFIG" >&2; exit 2; }
SEED="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["seed"])' "$CONFIG")"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$(mktemp -d)"
  trap 'rm -rf "$OUTPUT_ROOT"' EXIT
else
  mkdir -p "$OUTPUT_ROOT"
fi

run_trial() {
  local name="$1"
  local root="$OUTPUT_ROOT/$name"
  mkdir -p "$root"/{checkpoints,prototypes,runtime/weights}

  "$PYTHON" "$PIPELINE/train.py" --config "$CONFIG" --features "$FEATURES" \
    --checkpoint-dir "$root/checkpoints" --noise-std 0 --mixup-prob 0
  "$PYTHON" "$PIPELINE/evaluate.py" --checkpoint "$root/checkpoints/best.pt" \
    --features "$FEATURES" --export-prototypes --prototype-dir "$root/prototypes" --num-episodes 2
  "$PYTHON" "$PIPELINE/export_model.py" --prototypes "$root/prototypes/prototypes.pt" \
    --weights-dir "$root/runtime/weights" --config-template "$BACKEND_DIR/codex_model/config.json" \
    --config-out "$root/runtime/config.json" --manifest-out "$root/export_model_manifest.json"
}

run_trial first
run_trial second

for artifact in runtime/weights/projection.pt runtime/weights/prototypes.pt runtime/config.json checkpoints/training_manifest.json; do
  cmp -- "$OUTPUT_ROOT/first/$artifact" "$OUTPUT_ROOT/second/$artifact"
done

"$PYTHON" - "$OUTPUT_ROOT/first" "$OUTPUT_ROOT/second" <<'PY'
import json
import sys
from pathlib import Path

first, second = map(Path, sys.argv[1:])
for relative in ("prototypes/provenance.json", "export_model_manifest.json"):
    left = json.loads((first / relative).read_text(encoding="utf-8"))
    right = json.loads((second / relative).read_text(encoding="utf-8"))
    if relative.endswith("provenance.json"):
        for value in (left, right):
            for key in ("checkpoint_path", "prototypes_path"):
                value.pop(key, None)
    else:
        left = {name: data["sha256"] for name, data in left["artifacts"].items()}
        right = {name: data["sha256"] for name, data in right["artifacts"].items()}
    if left != right:
        raise SystemExit(f"non-deterministic manifest: {relative}")
PY

echo "PASS: deterministic E2E smoke succeeded ($OUTPUT_ROOT)"
