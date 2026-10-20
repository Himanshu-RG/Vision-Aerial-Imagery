"""YAML config loading with pydantic validation.

Only the fields needed so far are modelled. Later modules extend `RunConfig` rather than
reading raw dicts, so typos in config keys fail loudly (extra="forbid").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    """Fields every experiment run must declare."""

    name: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    deterministic: bool = False
    device: str = "cpu"  # "cpu", "cuda", "cuda:0", ...
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    output_root: str = "outputs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path, model: type[T]) -> T:
    """Load a YAML file and validate it against `model` (raises pydantic.ValidationError)."""
    return model.model_validate(load_yaml(path))
