import json

import pytest

from hitl.common import HitlError, read_json
from hitl.ingest import run_ingest
from hitl.retrain import retrain


def test_user_retrain_archives_candidate_and_preserves_baseline(tmp_repo):
    paths, write_feedback, candidates = tmp_repo
    input_path = paths.root / "prototype_feedback.csv"
    write_feedback(input_path, candidates(20), validator=False, user=True)
    original_registry = read_json(paths.registry)
    bundled_before = [(paths.bundled / name).read_bytes() for name in ("crf_model.pkl", "vocab.pkl")]
    version = retrain(paths, [input_path.name], "user")
    assert version == 2
    registry = read_json(paths.registry)
    entry = registry["versions"][1]
    assert len(registry["versions"]) == 2
    assert registry["active_version"] == 1
    assert registry["versions"][0] == original_registry["versions"][0]
    assert entry["parent_version"] == 1
    assert entry["best_c1"] == entry["best_c2"] == 0.01
    assert entry["experimental"] is True
    assert entry["label_policy"] == "user"
    assert entry["notes"].startswith("EXPERIMENTAL")
    assert paths.model_file(2).exists() and paths.vocab_file(2).exists()
    manifest = read_json(paths.version_results(2) / "train_manifest.json")
    assert len(manifest["feedback"]["train_uids"]) == entry["n_feedback_train_sentences"]
    report = read_json(paths.snapshot(2) / "ingest_report.json")
    assert entry["n_feedback_train_sentences"] == report["n_trainable"]
    assert "n_feedback_holdout_sentences" not in entry
    assert "n_holdout_sentences" not in manifest["feedback"]
    with open(paths.vocab_file(2), "rb") as file:
        import pickle
        assert "mamumuya" in pickle.load(file)
    assert bundled_before == [(paths.bundled / name).read_bytes() for name in ("crf_model.pkl", "vocab.pkl")]


def test_validator_without_validator_tags_fails_before_candidate_creation(tmp_repo):
    paths, write_feedback, candidates = tmp_repo
    input_path = paths.root / "prototype_feedback.csv"
    write_feedback(input_path, candidates(4), validator=False)
    with pytest.raises(HitlError, match="--label-policy user"):
        retrain(paths, [input_path.name], "validator")
    assert not paths.model_file(2).exists()
    assert not paths.vocab_file(2).exists()
    assert not paths.version_results(2).exists()
    assert not paths.snapshot(2).exists()


def test_evaluation_failure_cleans_candidate_outputs(tmp_repo, monkeypatch):
    paths, write_feedback, candidates = tmp_repo
    input_path = paths.root / "prototype_feedback.csv"
    write_feedback(input_path, candidates(8), validator=False)
    original = read_json(paths.registry)
    monkeypatch.setattr("hitl.evaluate.evaluate_version", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated evaluation failure")))
    with pytest.raises(RuntimeError, match="simulated evaluation failure"):
        retrain(paths, [input_path.name], "user")
    assert not paths.model_file(2).exists()
    assert not paths.vocab_file(2).exists()
    assert not paths.version_results(2).exists()
    assert not paths.snapshot(2).exists()
    assert read_json(paths.registry) == original


def test_unknown_label_policy_is_rejected(tmp_repo):
    paths, _, _ = tmp_repo
    with pytest.raises(HitlError, match="unknown label policy"):
        retrain(paths, label_policy="guess")