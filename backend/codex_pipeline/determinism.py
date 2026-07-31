"""Deterministic execution controls shared by baseline pipeline stages."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_determinism(seed: int, *, strict: bool = True) -> None:
    """Seed every RNG used by the pipeline before work starts.

    Strict mode is intended for reproducible baseline runs.  It deliberately
    rejects unsupported nondeterministic kernels instead of silently falling
    back to a best-effort result.
    """

    # PYTHONHASHSEED must be exported by the parent process before Python starts.
    # The baseline launchers do this before spawning every pipeline stage.
    if os.environ.get("PYTHONHASHSEED") not in (None, str(seed)):
        raise ValueError("PYTHONHASHSEED must equal the configured training seed")
    # Must be set before the first CUDA BLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    torch.use_deterministic_algorithms(strict)
