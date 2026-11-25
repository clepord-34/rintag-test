import copy
import shutil

import pytest

from hitl.common import HitlError, read_json, sha256_file, write_json_atomic
from hitl.promote import promote


def _add_version(paths, experimental=False):
    shutil.copy2(paths.model_file(1), paths.model_file(2))
    shutil.copy2(paths.vocab_file(1), paths.vocab_file(2))
    registry = read_json(paths.registry)
    entry = copy.deepcopy(registry["versions"][0])
    entry.update({"version": 2, "experimental": experimental,
                  "artifact_sha256": {"crf_model": sha256_file(paths.model_file(2)), "vocab": sha256_file(paths.vocab_file(2))}})
    registry["versions"].append(entry)
    write_json_atomic(paths.registry, registry)


def test_promote_and_rollback_preserve_archived_bytes(tmp_repo):
    paths, _, _ = tmp_repo
    _add_version(paths)
    promote(paths, 2, assume_yes=True, skip_tests=True)
    assert (paths.bundled / "crf_model.pkl").read_bytes() == paths.model_file(2).read_bytes()
    assert (paths.bundled / "vocab.pkl").read_bytes() == paths.vocab_file(2).read_bytes()
    assert read_json(paths.registry)["active_version"] == 2
    assert len(read_json(paths.registry)["promotion_log"]) == 1
    promote(paths, 1, assume_yes=True, skip_tests=True)
    assert (paths.bundled / "crf_model.pkl").read_bytes() == paths.model_file(1).read_bytes()


def test_experimental_refusal_and_hash_mismatch(tmp_repo):
    paths, _, _ = tmp_repo
    _add_version(paths, experimental=True)
    originals = [(paths.bundled / name).read_bytes() for name in ("crf_model.pkl", "vocab.pkl")]
    registry = read_json(paths.registry)
    with pytest.raises(HitlError, match="--force"):
        promote(paths, 2, assume_yes=True, skip_tests=True)
    assert originals == [(paths.bundled / name).read_bytes() for name in ("crf_model.pkl", "vocab.pkl")]
    promote(paths, 2, assume_yes=True, force=True, skip_tests=True)
    registry = read_json(paths.registry)
    registry["active_version"] = 1
    registry["versions"][1]["artifact_sha256"]["crf_model"] = "bad-hash"
    write_json_atomic(paths.registry, registry)
    with pytest.raises(HitlError, match="SHA-256"):
        promote(paths, 2, assume_yes=True, force=True, skip_tests=True)