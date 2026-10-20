"""Seeding helpers. Determinism is best-effort: some CUDA kernels have no deterministic version."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> dict:
    """Seed Python, NumPy and (if installed) PyTorch. Returns what was set, for the run record."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    record = {"seed": seed, "deterministic": deterministic, "torch_seeded": False}
    try:
        import torch
    except ImportError:
        return record

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    record["torch_seeded"] = True

    if deterministic:
        # Needed by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: ops without a deterministic kernel warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)
    return record
