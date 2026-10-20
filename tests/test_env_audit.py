import json

import env_audit


def test_collect_has_required_fields_and_is_json():
    info = env_audit.collect()
    for key in ("hostname", "os", "python", "packages", "torch", "disk_free_gb", "git_commit"):
        assert key in info
    assert "torch" in info["packages"]
    json.dumps(info)  # must be serialisable
