"""Immutable run identifiers: the same config always maps to the same ID, any change gives a new one."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


def canonical_json(obj: Any) -> str:
    """Key-order-independent JSON encoding used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    if not 4 <= length <= 64:
        raise ValueError("length must be in [4, 64]")
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:length]


def make_run_id(name: str, config: Mapping[str, Any]) -> str:
    """`<name>-<hash>`; the name is sanitised so it is safe as a directory name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"
    return f"{safe}-{config_hash(config)}"
