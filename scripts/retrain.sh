#!/usr/bin/env bash
# retrain.sh — approved-only classifier retraining with explicit repo-root paths.
# Usage: bash scripts/retrain.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
LOCKFILE="$BACKEND_DIR/.retrain.lock"
PYTHON="${PYTHON:-$BACKEND_DIR/.venv/bin/python3}"
PIPELINE="$BACKEND_DIR/codex_pipeline/scripts"

ANNOTATIONS_DIR="$BACKEND_DIR/annotations"
APPROVED_ROOT="$BACKEND_DIR/training_data/approved"
ELEMENTS_DIR="$APPROVED_ROOT/Elements"
METADATA_CSV="$APPROVED_ROOT/metadata.csv"
PRECOMPUTED_DIR="$APPROVED_ROOT/precomputed"
FEATURES_FILE="$PRECOMPUTED_DIR/features.pt"
CHECKPOINT_DIR="$BACKEND_DIR/checkpoints"
PROTOTYPE_DIR="$BACKEND_DIR/prototypes"
PROTOTYPE_FILE="$PROTOTYPE_DIR/prototypes.pt"
MODEL_REGISTRY_DIR="${MODEL_REGISTRY_DIR:-$BACKEND_DIR/model_registry}"
GIT_SHORT="${GIT_SHORT:-$(git -C "$REPO_ROOT" rev-parse --short=8 HEAD 2>/dev/null || printf 'nogit')}"
RUN8_SOURCE="${MODEL_VERSION_RUN_ID:-$(date -u +%H%M%S)-$$}"
MODEL_VERSION_ID="${MODEL_VERSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$GIT_SHORT-${RUN8_SOURCE:0:8}}"
if [[ -z "$MODEL_VERSION_ID" || "$MODEL_VERSION_ID" == *"/"* || "$MODEL_VERSION_ID" == *"\\"* || "$MODEL_VERSION_ID" == *".."* || ! "$MODEL_VERSION_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ERROR: invalid MODEL_VERSION_ID: $MODEL_VERSION_ID" >&2
  echo "MODEL_VERSION_ID may contain only letters, numbers, underscore, dot, and dash; path separators and '..' are forbidden." >&2
  exit 2
fi
VERSION_DIR="$MODEL_REGISTRY_DIR/versions/$MODEL_VERSION_ID"
CHECKPOINT_DIR="$VERSION_DIR/checkpoints"
PROTOTYPE_DIR="$VERSION_DIR/prototypes"
PROTOTYPE_FILE="$PROTOTYPE_DIR/prototypes.pt"
WEIGHTS_DIR="$VERSION_DIR/runtime/weights"
CLASSIFIER_CONFIG_TEMPLATE="$BACKEND_DIR/codex_model/config.json"
CLASSIFIER_CONFIG="$VERSION_DIR/runtime/config.json"
EXPORT_MANIFEST="$VERSION_DIR/export_model_manifest.json"
CONFIG="$BACKEND_DIR/codex_pipeline/config/default.yaml"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-auto}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/retrain.sh [--dry-run]

Runs approved-only classifier/prototype retraining:
  1. export approved annotation crops to backend/training_data/approved/Elements
  2. build metadata CSV from that generated Elements directory
  3. precompute DINOv2 embeddings to backend/training_data/approved/precomputed/features.pt
  4. train projection/classifier checkpoints from that explicit features file
  5. evaluate/export prototypes to backend/model_registry/versions/<id>/prototypes/prototypes.pt
  6. export backend-loadable candidate artifacts/config under backend/model_registry/versions/<id>

No segmentation/MobileSAM retraining is performed.
No runtime backend/codex_model artifacts are modified; run scripts/promote_model.py to activate a candidate.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_step() {
  local label="$1"
  shift
  echo "=== $label ==="
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ -f "$LOCKFILE" ]]; then
    OLD_PID="$(cat "$LOCKFILE")"
    if kill -0 "$OLD_PID" 2>/dev/null; then
      echo "ERROR: retrain already running (PID $OLD_PID). Aborting." >&2
      exit 1
    fi
    echo "WARNING: stale lockfile (PID $OLD_PID not running). Removing." >&2
    rm -f "$LOCKFILE"
  fi

  echo $$ > "$LOCKFILE"
  trap 'rm -f "$LOCKFILE"' EXIT INT TERM
fi

run_step "[1/6] export_approved_annotations" \
  "$PYTHON" "$REPO_ROOT/scripts/export_approved_annotations.py" \
  --annotations-dir "$ANNOTATIONS_DIR" \
  --output "$ELEMENTS_DIR"

run_step "[2/6] build_metadata" \
  "$PYTHON" "$PIPELINE/build_metadata.py" \
  --elements-dir "$ELEMENTS_DIR" \
  --output "$METADATA_CSV"

run_step "[3/6] precompute_embeddings" \
  "$PYTHON" "$PIPELINE/precompute_embeddings.py" \
  --config "$CONFIG" \
  --metadata-csv "$METADATA_CSV" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$PRECOMPUTED_DIR"

run_step "[4/6] train" \
  "$PYTHON" "$PIPELINE/train.py" \
  --config "$CONFIG" \
  --features "$FEATURES_FILE" \
  --checkpoint-dir "$CHECKPOINT_DIR"

run_step "[5/6] evaluate_export_prototypes" \
  "$PYTHON" "$PIPELINE/evaluate.py" \
  --checkpoint "$CHECKPOINT_DIR/best.pt" \
  --features "$FEATURES_FILE" \
  --export-prototypes \
  --prototype-dir "$PROTOTYPE_DIR"

run_step "[6/6] export_model" \
  "$PYTHON" "$PIPELINE/export_model.py" \
  --prototypes "$PROTOTYPE_FILE" \
  --weights-dir "$WEIGHTS_DIR" \
  --config-template "$CLASSIFIER_CONFIG_TEMPLATE" \
  --config-out "$CLASSIFIER_CONFIG" \
  --manifest-out "$EXPORT_MANIFEST" \
  --registry-dir "$MODEL_REGISTRY_DIR" \
  --version-id "$MODEL_VERSION_ID" \
  --metadata-csv "$METADATA_CSV" \
  --approved-manifest "$ELEMENTS_DIR/_approved_export_manifest.json"

echo "=== Candidate model version created: $MODEL_VERSION_ID ==="
echo "=== Inspect: $VERSION_DIR ==="
echo "=== Promote explicitly: $PYTHON scripts/promote_model.py $MODEL_VERSION_ID --dry-run ==="
