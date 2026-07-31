#!/usr/bin/env bash
# Rebuild the frozen-DINO baseline from an approved, non-annotation corpus.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
PIPELINE="$BACKEND_DIR/codex_pipeline/scripts"
PYTHON="${PYTHON:-$BACKEND_DIR/.venv/bin/python3}"

CONFIG="$BACKEND_DIR/codex_pipeline/config/baseline.yaml"
RUNTIME_CONFIG="$BACKEND_DIR/codex_model/config.json"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ELEMENTS_DIR=""
APPROVED_MANIFEST=""
BACKBONE_MANIFEST=""
OUTPUT_ROOT="${OUTPUT_ROOT:-$BACKEND_DIR/training_runs}"
RUN_ID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_baseline_recipe.sh \
  --elements-dir <Elements> --approved-manifest <import_snapshot.json> \
  --backbone-manifest <dinov2-local-pin.json> [options]

Runs the reproducible, pre-annotation baseline: metadata -> frozen DINO
features -> leakage-free episodic head training -> prototype export -> runtime
candidate package. It never writes backend/codex_model.

Options:
  --config <yaml>       Baseline config (default: codex_pipeline/config/baseline.yaml)
  --runtime-config <json> Runtime class-order contract (default: backend/codex_model/config.json)
  --output-root <dir>   Parent for immutable run directory (default: backend/training_runs)
  --run-id <id>         Optional stable suffix; otherwise derived from input hashes
  --device <device>     cuda by default; set cpu deliberately for a slow fallback
  --batch-size <n>      DINO feature batch size (default: 16)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --elements-dir) ELEMENTS_DIR="$2"; shift 2 ;;
    --approved-manifest) APPROVED_MANIFEST="$2"; shift 2 ;;
    --backbone-manifest) BACKBONE_MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --runtime-config) RUNTIME_CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in "$ELEMENTS_DIR" "$APPROVED_MANIFEST" "$BACKBONE_MANIFEST" "$CONFIG" "$RUNTIME_CONFIG"; do
  [[ -n "$required" && -e "$required" ]] || { echo "ERROR: required input is missing: $required" >&2; exit 2; }
done
[[ -d "$ELEMENTS_DIR" ]] || { echo "ERROR: --elements-dir must be a directory" >&2; exit 2; }

ELEMENTS_DIR="$(cd "$ELEMENTS_DIR" && pwd -P)"
APPROVED_MANIFEST="$(cd "$(dirname "$APPROVED_MANIFEST")" && pwd -P)/$(basename "$APPROVED_MANIFEST")"
BACKBONE_MANIFEST="$(cd "$(dirname "$BACKBONE_MANIFEST")" && pwd -P)/$(basename "$BACKBONE_MANIFEST")"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd -P)/$(basename "$CONFIG")"
RUNTIME_CONFIG="$(cd "$(dirname "$RUNTIME_CONFIG")" && pwd -P)/$(basename "$RUNTIME_CONFIG")"
SEED="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["seed"])' "$CONFIG")"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

RUN_FINGERPRINT="$({
  printf '%s\n' "device=$DEVICE" "batch_size=$BATCH_SIZE"
  sha256sum "$CONFIG" "$RUNTIME_CONFIG" "$APPROVED_MANIFEST" "$BACKBONE_MANIFEST" \
    "$0" "$PIPELINE/build_metadata.py" "$PIPELINE/precompute_embeddings.py" "$PIPELINE/train.py" "$PIPELINE/evaluate.py" "$PIPELINE/export_model.py"
} | sha256sum | cut -c1-16)"
RUN_ID="${RUN_ID:-baseline-${RUN_FINGERPRINT}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: invalid --run-id" >&2; exit 2; }
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "ERROR: immutable run directory already exists: $RUN_DIR" >&2; exit 2; }

if [[ "$DEVICE" =~ ^cuda(:[0-9]+)?$ ]]; then
  "$PYTHON" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA was requested but is unavailable; explicitly pass --device cpu for a fallback run.")
print(f"CUDA baseline device: {torch.cuda.get_device_name(0)}")
PY
fi

mkdir -p "$RUN_DIR"/{precomputed,checkpoints,prototypes,runtime/weights}
METADATA="$RUN_DIR/metadata.csv"
FEATURES="$RUN_DIR/precomputed/features.pt"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

run "$PYTHON" "$PIPELINE/build_metadata.py" --elements-dir "$ELEMENTS_DIR" --output "$METADATA" --absolute-paths
run "$PYTHON" "$PIPELINE/precompute_embeddings.py" --config "$CONFIG" --runtime-config "$RUNTIME_CONFIG" --metadata-csv "$METADATA" --backbone-manifest "$BACKBONE_MANIFEST" --batch-size "$BATCH_SIZE" --device "$DEVICE" --output-dir "$RUN_DIR/precomputed"
run "$PYTHON" "$PIPELINE/train.py" --config "$CONFIG" --features "$FEATURES" --checkpoint-dir "$RUN_DIR/checkpoints" --noise-std 0 --mixup-prob 0
run "$PYTHON" "$PIPELINE/evaluate.py" --checkpoint "$RUN_DIR/checkpoints/best.pt" --features "$FEATURES" --export-prototypes --prototype-dir "$RUN_DIR/prototypes" --num-episodes "$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["evaluation"]["num_eval_episodes"])' "$CONFIG")"
run "$PYTHON" "$PIPELINE/export_model.py" --prototypes "$RUN_DIR/prototypes/prototypes.pt" --weights-dir "$RUN_DIR/runtime/weights" --config-template "$RUNTIME_CONFIG" --config-out "$RUN_DIR/runtime/config.json" --manifest-out "$RUN_DIR/export_model_manifest.json"

"$PYTHON" - "$RUN_DIR/run_manifest.json" "$RUN_ID" "$CONFIG" "$RUNTIME_CONFIG" "$APPROVED_MANIFEST" "$BACKBONE_MANIFEST" "$ELEMENTS_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_id = sys.argv[2]
config, runtime, approved, backbone, elements = map(Path, sys.argv[3:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def artifact(path):
    return {"path": str(path), "sha256": sha(path)}
run_dir = out.parent
outputs = {
    name: artifact(path)
    for name, path in {
        "features": run_dir / "precomputed/features.pt",
        "features_provenance": run_dir / "precomputed/features.pt.prov.json",
        "checkpoint": run_dir / "checkpoints/best.pt",
        "training_provenance": run_dir / "checkpoints/training_manifest.json",
        "prototypes": run_dir / "prototypes/prototypes.pt",
        "prototype_provenance": run_dir / "prototypes/provenance.json",
        "runtime_projection": run_dir / "runtime/weights/projection.pt",
        "runtime_prototypes": run_dir / "runtime/weights/prototypes.pt",
        "runtime_config": run_dir / "runtime/config.json",
    }.items()
}
out.write_text(json.dumps({
    "schema_version": "baseline-run.v1",
    "run_id": run_id,
    "inputs": {
        "config": {"path": str(config), "sha256": sha(config)},
        "runtime_config": {"path": str(runtime), "sha256": sha(runtime)},
        "approved_manifest": {"path": str(approved), "sha256": sha(approved)},
        "backbone_manifest": {"path": str(backbone), "sha256": sha(backbone)},
        "elements_dir": str(elements),
    },
    "outputs": outputs,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Baseline candidate created: $RUN_DIR"
