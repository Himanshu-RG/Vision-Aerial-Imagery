"""Locate the repo root and the large-file store (datasets, checkpoints) outside version control.

Resolution order for the store:
  1. AOVD_STORE environment variable
  2. `store:` key in configs/paths.local.yaml (gitignored, per machine)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_PATHS_FILE = REPO_ROOT / "configs" / "paths.local.yaml"


class StoreNotConfigured(RuntimeError):
    pass


def get_store(must_exist: bool = True) -> Path:
    raw = os.environ.get("AOVD_STORE")
    source = "AOVD_STORE"
    if not raw and LOCAL_PATHS_FILE.is_file():
        from aerial_ovd.utils.config import load_yaml

        raw = load_yaml(LOCAL_PATHS_FILE).get("store")
        source = str(LOCAL_PATHS_FILE)
    if not raw:
        raise StoreNotConfigured(
            "Large-file store not configured. Set AOVD_STORE or create configs/paths.local.yaml "
            "(copy configs/paths.example.yaml)."
        )
    store = Path(raw).expanduser()
    if must_exist and not store.is_dir():
        raise StoreNotConfigured(f"store from {source} does not exist: {store}")
    return store


def store_subdir(name: str, create: bool = True) -> Path:
    path = get_store() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
