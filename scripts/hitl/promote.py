"""Guarded promotion and rollback of archived model versions."""

import shutil
import subprocess
import sys

from hitl.common import HitlError, sha256_file
from hitl.evaluate import evaluate_version
from hitl.registry import get_entry, load_registry, set_active


def promote(paths, version, assume_yes=False, force=False, skip_tests=False):
    registry = load_registry(paths)
    entry = get_entry(registry, version)
    if entry.get("experimental") and not force:
        raise HitlError(f"version {version} is experimental (trained without validator-only labels); use --force to promote it anyway")
    for label, artifact, key in (
        ("model", paths.model_file(version), "crf_model"),
        ("vocabulary", paths.vocab_file(version), "vocab"),
    ):
        if not artifact.exists() or sha256_file(artifact) != entry.get("artifact_sha256", {}).get(key):
            raise HitlError(f"version {version} {label} artifact is missing or its SHA-256 does not match the registry")

    active = registry["active_version"]
    print(f"Promotion comparison: v{version} vs active v{active}")
    for name in ("test", "hitl_holdout", "hitl_holdout_user"):
        target = evaluate_version(paths, version, name)
        current = evaluate_version(paths, active, name)
        if target is None or current is None:
            print(f"  {name}: v{active}=n/a, v{version}=n/a" if target is None and current is None else f"  {name}: comparison n/a")
        else:
            print(f"  {name}: v{active} acc={current['accuracy']:.4f} macro_f1={current['macro_f1']:.4f}; v{version} acc={target['accuracy']:.4f} macro_f1={target['macro_f1']:.4f}")
    if not assume_yes and input(f"Promote v{version} to the bundled model? [y/N] ").strip().lower() != "y":
        print("cancelled")
        return

    paths.bundled.mkdir(parents=True, exist_ok=True)
    model_tmp = paths.bundled / "crf_model.pkl.tmp"
    vocab_tmp = paths.bundled / "vocab.pkl.tmp"
    shutil.copyfile(paths.model_file(version), model_tmp)
    shutil.copyfile(paths.vocab_file(version), vocab_tmp)
    model_tmp.replace(paths.bundled / "crf_model.pkl")
    vocab_tmp.replace(paths.bundled / "vocab.pkl")
    set_active(paths, version)
    if not skip_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=paths.root)
        if result.returncode:
            print(f"Tests failed after promotion. Roll back with: python scripts/06_hitl.py promote --version {active} --yes")
            raise HitlError(f"post-promotion tests failed with exit code {result.returncode}")