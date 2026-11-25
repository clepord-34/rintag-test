import copy

import pytest

from hitl.common import HitlError, read_json, write_json_atomic
from hitl.registry import append_version, get_entry, next_version, set_active


def test_registry_append_and_promote(tmp_repo):
    paths, _, _ = tmp_repo
    registry = read_json(paths.registry)
    original = copy.deepcopy(registry["versions"])
    assert next_version(registry) == 2
    new = {"version": 2, "test_accuracy": 0.1}
    append_version(paths, new)
    updated = read_json(paths.registry)
    assert updated["versions"][:1] == original
    assert updated["active_version"] == 1
    with pytest.raises(HitlError):
        append_version(paths, new)
    set_active(paths, 2)
    updated = read_json(paths.registry)
    assert updated["active_version"] == 2
    assert updated["promotion_log"][-1]["from_version"] == 1
    assert updated["promotion_log"][-1]["to_version"] == 2
    assert get_entry(updated, 1) == original[0]