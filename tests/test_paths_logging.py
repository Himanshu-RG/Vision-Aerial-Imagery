import json

import pytest

from aerial_ovd.utils import paths
from aerial_ovd.utils.logging import get_logger


def test_store_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AOVD_STORE", str(tmp_path))
    assert paths.get_store() == tmp_path


def test_store_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AOVD_STORE", str(tmp_path / "does_not_exist"))
    with pytest.raises(paths.StoreNotConfigured):
        paths.get_store()


def test_store_unconfigured_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("AOVD_STORE", raising=False)
    monkeypatch.setattr(paths, "LOCAL_PATHS_FILE", tmp_path / "absent.yaml")
    with pytest.raises(paths.StoreNotConfigured):
        paths.get_store()


def test_jsonl_logging(tmp_path):
    log_file = tmp_path / "run.jsonl"
    logger = get_logger("test_jsonl", log_file)
    logger.info("image failed", extra={"image_id": 42, "reason": "oom"})
    for h in logger.handlers:
        h.flush()
    rec = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["msg"] == "image failed" and rec["image_id"] == 42 and rec["reason"] == "oom"
