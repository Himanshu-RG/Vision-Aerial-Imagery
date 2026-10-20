import pytest
from pydantic import ValidationError

from aerial_ovd.utils.config import RunConfig, load_config, load_yaml


def test_valid_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: smoke\nseed: 3\nprecision: fp16\ndevice: cuda\n", encoding="utf-8")
    cfg = load_config(p, RunConfig)
    assert cfg.name == "smoke" and cfg.seed == 3 and cfg.precision == "fp16"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: smoke\nbox_treshold: 0.3\n", encoding="utf-8")  # typo must fail loudly
    with pytest.raises(ValidationError):
        load_config(p, RunConfig)


def test_bad_values_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: smoke\nprecision: fp8\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p, RunConfig)


def test_config_is_frozen():
    cfg = RunConfig(name="x")
    with pytest.raises(ValidationError):
        cfg.seed = 5  # type: ignore[misc]


def test_missing_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(p)
