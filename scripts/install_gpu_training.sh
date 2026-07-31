#!/usr/bin/env bash
# Install the CUDA-enabled PyTorch wheels required for RTX GPU training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/backend/.venv/bin/python}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver tooling (nvidia-smi) is unavailable." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python virtual environment not found: $PYTHON" >&2
  echo "Run scripts/install.sh first, then re-run this script." >&2
  exit 1
fi

echo "Installing PyTorch 2.5.1 CUDA 12.4 wheels into: $PYTHON"
"$PYTHON" -m pip install --upgrade --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124

"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA PyTorch installation completed but no CUDA device is usable. "
        f"torch.version.cuda={torch.version.cuda!r}"
    )

print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
PY
