#!/usr/bin/env bash
# retrain.sh — classifier retraining from approved data or a cumulative snapshot.
# Usage: bash scripts/retrain.sh [options]

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
# Training is GPU-first. Set DEVICE=cpu only for an intentional CPU fallback.
DEVICE="${DEVICE:-cuda}"
DRY_RUN=0
ELEMENTS_DIR_OVERRIDE=""
APPROVED_MANIFEST_OVERRIDE=""
METADATA_CSV_OVERRIDE=""
BACKBONE_MANIFEST_OVERRIDE=""
CONFIG_OVERRIDE=""
INIT_PROJECTION=""
EVAL_ONLY=0
UPDATE_ANNOTATED_PROTOTYPES=0
CHECKPOINT_SELECTION_CLI=""

usage() {
  cat <<'EOF'
Usage: bash scripts/retrain.sh [--dry-run] [--elements-dir <Elements>] [--approved-manifest <snapshot.json>] [--metadata-csv <metadata.csv>] [--backbone-manifest <pin.json>] [--config <training.yaml>] [--init-projection <projection.pt>] [--eval-only] [--checkpoint-selection best|latest]

Runs classifier/prototype retraining. Without snapshot arguments it:
  1. export approved annotation crops to backend/training_data/approved/Elements
  2. build metadata CSV from that generated Elements directory
  3. precompute DINOv2 embeddings to backend/training_data/approved/precomputed/features.pt
  4. train projection/classifier checkpoints from that explicit features file
  5. evaluate/export prototypes to backend/model_registry/versions/<id>/prototypes/prototypes.pt
  6. export backend-loadable candidate artifacts/config under backend/model_registry/versions/<id>

No segmentation/MobileSAM retraining is performed.
No runtime backend/codex_model artifacts are modified; run scripts/promote_model.py to activate a candidate.

CUDA is required by default. Set DEVICE=cpu only for an intentional CPU run.

--elements-dir uses an already materialized training snapshot instead of
exporting admin annotations. It requires --approved-manifest and --metadata-csv
from the same immutable training-snapshot.v2 directory. The default recipe for
this path is config/snapshot.yaml (persisted train/dev/locked_test split), and
--backbone-manifest must bind the local DINOv2-S/14 source and weights.
--eval-only requires --init-projection and performs descriptive dev evaluation
without fitting. --checkpoint-selection defaults to best for compatibility.
--update-annotated-prototypes keeps the active projection and unannotated class
prototypes; requires a prepared snapshot and the active weights/projection.pt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --elements-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --elements-dir requires a path" >&2; exit 2; }
      ELEMENTS_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --approved-manifest)
      [[ $# -ge 2 ]] || { echo "ERROR: --approved-manifest requires a path" >&2; exit 2; }
      APPROVED_MANIFEST_OVERRIDE="$2"
      shift 2
      ;;
    --metadata-csv)
      [[ $# -ge 2 ]] || { echo "ERROR: --metadata-csv requires a path" >&2; exit 2; }
      METADATA_CSV_OVERRIDE="$2"
      shift 2
      ;;
    --backbone-manifest)
      [[ $# -ge 2 ]] || { echo "ERROR: --backbone-manifest requires a path" >&2; exit 2; }
      BACKBONE_MANIFEST_OVERRIDE="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 2; }
      CONFIG_OVERRIDE="$2"
      shift 2
      ;;
    --init-projection)
      [[ $# -ge 2 ]] || { echo "ERROR: --init-projection requires a path" >&2; exit 2; }
      INIT_PROJECTION="$2"
      shift 2
      ;;
    --eval-only)
      EVAL_ONLY=1
      shift
      ;;
    --update-annotated-prototypes)
      UPDATE_ANNOTATED_PROTOTYPES=1
      EVAL_ONLY=1
      shift
      ;;
    --checkpoint-selection)
      [[ $# -ge 2 ]] || { echo "ERROR: --checkpoint-selection requires best or latest" >&2; exit 2; }
      CHECKPOINT_SELECTION_CLI="$2"
      shift 2
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

if [[ -n "$CHECKPOINT_SELECTION_CLI" && "$CHECKPOINT_SELECTION_CLI" != "best" && "$CHECKPOINT_SELECTION_CLI" != "latest" ]]; then
  echo "ERROR: --checkpoint-selection must be best or latest" >&2
  exit 2
fi
if [[ "$EVAL_ONLY" -eq 1 && -z "$INIT_PROJECTION" ]]; then
  echo "ERROR: --eval-only requires --init-projection" >&2
  exit 2
fi
if [[ "$UPDATE_ANNOTATED_PROTOTYPES" -eq 1 && -z "$ELEMENTS_DIR_OVERRIDE" ]]; then
  echo "ERROR: --update-annotated-prototypes requires --elements-dir" >&2
  exit 2
fi

if [[ ( -n "$APPROVED_MANIFEST_OVERRIDE" || -n "$METADATA_CSV_OVERRIDE" || -n "$BACKBONE_MANIFEST_OVERRIDE" ) && -z "$ELEMENTS_DIR_OVERRIDE" ]]; then
  echo "ERROR: snapshot provenance options require --elements-dir" >&2
  exit 2
fi

if [[ -n "$CONFIG_OVERRIDE" ]]; then
  CONFIG_DIR="$(cd "$(dirname "$CONFIG_OVERRIDE")" 2>/dev/null && pwd -P || printf '%s' "$(dirname "$CONFIG_OVERRIDE")")"
  CONFIG="$CONFIG_DIR/$(basename "$CONFIG_OVERRIDE")"
fi

if [[ -n "$ELEMENTS_DIR_OVERRIDE" ]]; then
  ELEMENTS_DIR="$(cd "$ELEMENTS_DIR_OVERRIDE" 2>/dev/null && pwd -P || printf '%s' "$ELEMENTS_DIR_OVERRIDE")"
  APPROVED_MANIFEST="$APPROVED_MANIFEST_OVERRIDE"
  if [[ -z "$APPROVED_MANIFEST" ]]; then
    echo "ERROR: --elements-dir requires --approved-manifest for candidate provenance" >&2
    exit 2
  fi
  if [[ -z "$METADATA_CSV_OVERRIDE" ]]; then
    echo "ERROR: --elements-dir requires --metadata-csv from the immutable snapshot" >&2
    exit 2
  fi
  if [[ -z "$BACKBONE_MANIFEST_OVERRIDE" ]]; then
    echo "ERROR: --elements-dir requires --backbone-manifest for reproducible DINOv2 features" >&2
    exit 2
  fi
  BACKBONE_MANIFEST="$(cd "$(dirname "$BACKBONE_MANIFEST_OVERRIDE")" 2>/dev/null && pwd -P || printf '%s' "$(dirname "$BACKBONE_MANIFEST_OVERRIDE")")/$(basename "$BACKBONE_MANIFEST_OVERRIDE")"
  TRAINING_WORK_DIR="$VERSION_DIR/training_data"
  METADATA_CSV="$(cd "$(dirname "$METADATA_CSV_OVERRIDE")" 2>/dev/null && pwd -P || printf '%s' "$(dirname "$METADATA_CSV_OVERRIDE")")/$(basename "$METADATA_CSV_OVERRIDE")"
  PRECOMPUTED_DIR="$TRAINING_WORK_DIR/precomputed"
  FEATURES_FILE="$PRECOMPUTED_DIR/features.pt"
  if [[ -z "$CONFIG_OVERRIDE" ]]; then
    CONFIG="$BACKEND_DIR/codex_pipeline/config/snapshot.yaml"
  fi
else
  APPROVED_MANIFEST="$ELEMENTS_DIR/_approved_export_manifest.json"
  TRAINING_WORK_DIR="$APPROVED_ROOT"
fi

resolve_checkpoint_selection() {
  "$PYTHON" - "$CONFIG" "$CHECKPOINT_SELECTION_CLI" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cli_selection = sys.argv[2] or None
try:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
except (OSError, yaml.YAMLError) as exc:
    raise SystemExit(f"ERROR: could not read training config {config_path}: {exc}") from exc
if not isinstance(config, dict):
    raise SystemExit(f"ERROR: training config must be a YAML mapping: {config_path}")
training = config.get("training", {})
if not isinstance(training, dict):
    raise SystemExit(f"ERROR: training config 'training' must be a mapping: {config_path}")
config_selection = training.get("checkpoint_selection")
if config_selection is not None and config_selection not in {"best", "latest"}:
    raise SystemExit(
        "ERROR: training.checkpoint_selection must be best or latest, "
        f"got {config_selection!r}"
    )
if cli_selection and config_selection and cli_selection != config_selection:
    raise SystemExit(
        "ERROR: --checkpoint-selection conflicts with preregistered "
        f"training.checkpoint_selection={config_selection}"
    )
print(cli_selection or config_selection or "best")
PY
}

CHECKPOINT_SELECTION="$(resolve_checkpoint_selection)"
SELECTED_CHECKPOINT="$CHECKPOINT_DIR/$CHECKPOINT_SELECTION.pt"

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

require_cuda() {
  [[ "$DEVICE" =~ ^cuda(:[0-9]+)?$ ]] || return 0
  "$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "ERROR: DEVICE=cuda was requested, but this Python environment cannot use CUDA. "
        "Install the CUDA PyTorch build with scripts/install_gpu_training.sh, "
        "or explicitly override with DEVICE=cpu."
    )

print(f"CUDA training device: {torch.cuda.get_device_name(0)}")
PY
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  require_cuda
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
  mkdir -p "$TRAINING_WORK_DIR"
fi

if [[ -n "$ELEMENTS_DIR_OVERRIDE" ]]; then
  echo "=== [1/6] use_prepared_elements_snapshot ==="
  printf '+ prepared Elements: %q\n' "$ELEMENTS_DIR"
  printf '+ approved manifest: %q\n' "$APPROVED_MANIFEST"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ -d "$ELEMENTS_DIR" ]] || { echo "ERROR: Elements directory does not exist: $ELEMENTS_DIR" >&2; exit 2; }
    [[ -f "$APPROVED_MANIFEST" ]] || { echo "ERROR: approved manifest does not exist: $APPROVED_MANIFEST" >&2; exit 2; }
    [[ -f "$METADATA_CSV" ]] || { echo "ERROR: snapshot metadata does not exist: $METADATA_CSV" >&2; exit 2; }
    [[ -f "$BACKBONE_MANIFEST" ]] || { echo "ERROR: backbone manifest does not exist: $BACKBONE_MANIFEST" >&2; exit 2; }
    [[ -f "$CONFIG" ]] || { echo "ERROR: training config does not exist: $CONFIG" >&2; exit 2; }
  fi
else
  run_step "[1/6] export_approved_annotations" \
    "$PYTHON" "$REPO_ROOT/scripts/export_approved_annotations.py" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --output "$ELEMENTS_DIR"
fi

if [[ -n "$ELEMENTS_DIR_OVERRIDE" ]]; then
  echo "=== [2/6] use_snapshot_metadata ==="
  printf '+ snapshot metadata: %q\n' "$METADATA_CSV"
else
  run_step "[2/6] build_metadata" \
    "$PYTHON" "$PIPELINE/build_metadata.py" \
    --elements-dir "$ELEMENTS_DIR" \
    --output "$METADATA_CSV"
fi

run_step "[3/6] precompute_embeddings" \
  "$PYTHON" "$PIPELINE/precompute_embeddings.py" \
  --config "$CONFIG" \
  --metadata-csv "$METADATA_CSV" \
  --runtime-config "$CLASSIFIER_CONFIG_TEMPLATE" \
  ${ELEMENTS_DIR_OVERRIDE:+--snapshot-manifest "$APPROVED_MANIFEST"} \
  ${ELEMENTS_DIR_OVERRIDE:+--backbone-manifest "$BACKBONE_MANIFEST"} \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$PRECOMPUTED_DIR"

TRAIN_COMMAND=(
  "$PYTHON" "$PIPELINE/train.py"
  --config "$CONFIG"
  --features "$FEATURES_FILE"
  --checkpoint-dir "$CHECKPOINT_DIR"
)
if [[ -n "$INIT_PROJECTION" ]]; then
  TRAIN_COMMAND+=(--init-projection "$INIT_PROJECTION")
fi
if [[ "$EVAL_ONLY" -eq 1 ]]; then
  TRAIN_COMMAND+=(--eval-only)
fi
run_step "[4/6] train" "${TRAIN_COMMAND[@]}"

if [[ -n "$ELEMENTS_DIR_OVERRIDE" ]]; then
  UPDATE_ARGUMENTS=()
  if [[ "$UPDATE_ANNOTATED_PROTOTYPES" -eq 1 ]]; then
    UPDATE_ARGUMENTS=(--base-prototypes "$(dirname "$INIT_PROJECTION")/prototypes.pt" --snapshot-manifest "$APPROVED_MANIFEST")
  fi
  run_step "[5/6] evaluate_export_prototypes" \
    "$PYTHON" "$PIPELINE/evaluate.py" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --features "$FEATURES_FILE" \
    --split-strategy persisted \
    --prototype-split train \
    --skip-few-shot \
    --export-prototypes \
    --prototype-dir "$PROTOTYPE_DIR" "${UPDATE_ARGUMENTS[@]}"
else
  run_step "[5/6] evaluate_export_prototypes" \
    "$PYTHON" "$PIPELINE/evaluate.py" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --features "$FEATURES_FILE" \
    --export-prototypes \
    --prototype-dir "$PROTOTYPE_DIR"
fi

EXPORT_COMMAND=(
  "$PYTHON" "$PIPELINE/export_model.py"
  --prototypes "$PROTOTYPE_FILE"
  --weights-dir "$WEIGHTS_DIR"
  --config-template "$CLASSIFIER_CONFIG_TEMPLATE"
  --config-out "$CLASSIFIER_CONFIG"
  --manifest-out "$EXPORT_MANIFEST"
  --registry-dir "$MODEL_REGISTRY_DIR"
  --version-id "$MODEL_VERSION_ID"
  --metadata-csv "$METADATA_CSV"
  --approved-manifest "$APPROVED_MANIFEST"
  --training-config "$CONFIG"
  --features "$FEATURES_FILE"
  --features-provenance "${FEATURES_FILE}.prov.json"
  --checkpoint "$SELECTED_CHECKPOINT"
  --training-manifest "$CHECKPOINT_DIR/training_manifest.json"
  --checkpoint-selection "$CHECKPOINT_SELECTION"
)
if [[ -n "$INIT_PROJECTION" ]]; then
  EXPORT_COMMAND+=(--init-projection "$INIT_PROJECTION")
  EXPORT_COMMAND+=(--base-model-dir "$(dirname "$(dirname "$INIT_PROJECTION")")")
fi
if [[ "$UPDATE_ANNOTATED_PROTOTYPES" -eq 1 ]]; then
  EXPORT_COMMAND+=(--evaluate-candidate)
fi
run_step "[6/6] export_model" "${EXPORT_COMMAND[@]}"

echo "=== Candidate model version created: $MODEL_VERSION_ID ==="
echo "=== Inspect: $VERSION_DIR ==="
echo "=== Promote explicitly: $PYTHON scripts/promote_model.py $MODEL_VERSION_ID --dry-run ==="
