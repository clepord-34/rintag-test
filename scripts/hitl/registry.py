"""Version registry operations."""

import copy

from hitl.common import HitlError, read_json, utc_now_iso, write_json_atomic


def load_registry(paths):
    if not paths.registry.exists():
        raise HitlError(f"model registry not found: {paths.registry}")
    return read_json(paths.registry)


def save_registry(paths, registry):
    write_json_atomic(paths.registry, registry)


def get_entry(registry, version):
    for entry in registry.get("versions", []):
        if entry.get("version") == version:
            return entry
    raise HitlError(f"unknown model version: {version}")


def all_versions(registry):
    return sorted(entry["version"] for entry in registry.get("versions", []))


def baseline_version(registry):
    versions = all_versions(registry)
    if not versions:
        raise HitlError("model registry contains no versions")
    return versions[0]


def latest_version(registry):
    versions = all_versions(registry)
    if not versions:
        raise HitlError("model registry contains no versions")
    return versions[-1]


def next_version(registry):
    versions = all_versions(registry) + [registry.get("active_version", 0)]
    return max(versions) + 1


def append_version(paths, entry):
    registry = load_registry(paths)
    old_entries = copy.deepcopy(registry.get("versions", []))
    if any(old.get("version") == entry.get("version") for old in old_entries):
        raise HitlError(f"version {entry.get('version')} already exists in registry")
    registry.setdefault("versions", []).append(copy.deepcopy(entry))
    if registry["versions"][:-1] != old_entries:
        raise AssertionError("append_version modified an existing registry entry")
    save_registry(paths, registry)


def set_active(paths, version):
    registry = load_registry(paths)
    get_entry(registry, version)
    previous = registry["active_version"]
    registry.setdefault("promotion_log", []).append({
        "timestamp": utc_now_iso(), "from_version": previous, "to_version": version,
    })
    registry["active_version"] = version
    save_registry(paths, registry)