from aerial_ovd.utils.run_id import config_hash, make_run_id


def test_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"x": 2, "y": 3}}) == config_hash({"b": {"y": 3, "x": 2}, "a": 1})


def test_hash_changes_with_value():
    assert config_hash({"thr": 0.35}) != config_hash({"thr": 0.36})


def test_run_id_is_filesystem_safe_and_stable():
    rid = make_run_id("E1 full vocab/dev", {"k": 80})
    assert rid == make_run_id("E1 full vocab/dev", {"k": 80})
    assert " " not in rid and "/" not in rid
    assert rid.startswith("E1_full_vocab_dev-") and len(rid.split("-")[-1]) == 12
