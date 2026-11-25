from hitl.common import read_json, sha256_file, write_csv, write_json_atomic
from hitl.evalsets import load_evalset
from hitl.evaluate import evaluate_version, verify_version


def test_evaluate_caches_and_verify_reproduces_registry(tmp_repo, monkeypatch):
    paths, _, _ = tmp_repo
    result = evaluate_version(paths, 1, "test")
    directory = paths.version_results(1)
    metrics_path = next(directory.glob("eval_test_*.json"))
    predictions_path = next(directory.glob("pred_test_*.csv"))
    modified = predictions_path.stat().st_mtime_ns

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cached prediction should be reused")

    monkeypatch.setattr("hitl.evaluate.predict_with_confidence", fail_if_called)
    assert evaluate_version(paths, 1, "test") == result
    assert predictions_path.stat().st_mtime_ns == modified
    monkeypatch.undo()
    dev = evaluate_version(paths, 1, "dev")
    registry = read_json(paths.registry)
    entry = registry["versions"][0]
    entry.update({
        "test_accuracy": round(result["accuracy"], 4),
        "test_macro_f1": round(result["macro_f1"], 4),
        "test_weighted_f1": round(result["weighted_f1"], 4),
        "test_tokens": result["n_tokens"],
        "oov_correct": result["oov_own_vocab"]["correct"],
        "oov_total": result["oov_own_vocab"]["total"],
        "dev_macro_f1": round(dev["macro_f1"], 4),
        "artifact_sha256": {"crf_model": sha256_file(paths.model_file(1)), "vocab": sha256_file(paths.vocab_file(1))},
    })
    write_json_atomic(paths.registry, registry)
    assert verify_version(paths, 1) == []
    entry["test_accuracy"] += 0.01
    write_json_atomic(paths.registry, registry)
    assert verify_version(paths, 1)
    assert evaluate_version(paths, 1, "hitl_holdout") is None
    assert metrics_path.exists()


def test_feedback_evalset_gold_is_policy_independent(tmp_repo):
    paths, _, _ = tmp_repo
    sentence_rows = [
        {"uid": "prototype:hold", "status": "ok", "split": "holdout", "validated": "true"},
        {"uid": "prototype:unvalidated", "status": "ok", "split": "holdout", "validated": "false"},
        {"uid": "prototype:train", "status": "ok", "split": "train", "validated": "true"},
        {"uid": "prototype:rejected", "status": "rejected", "split": "", "validated": ""},
    ]
    token_rows = [
        {"uid": row["uid"], "token_index": 0, "token": "mamumuya", "validator_tag": "VERB", "user_effective_tag": "NOUN", "final_tag": "ADJ"}
        for row in sentence_rows
    ]
    write_csv(paths.feedback_validated / "feedback_sentences.csv", list(sentence_rows[0]), sentence_rows)
    write_csv(paths.feedback_validated / "feedback_tokens.csv", list(token_rows[0]), token_rows)
    validated = load_evalset(paths, "hitl_holdout")
    user = load_evalset(paths, "hitl_holdout_user")
    assert validated.uids == ["prototype:hold"]
    assert validated.gold == [["VERB"]]
    assert user.uids == ["prototype:hold", "prototype:unvalidated"]
    assert user.gold == [["NOUN"], ["NOUN"]]
    for row in token_rows:
        row["final_tag"] = "PUNCT"
    write_csv(paths.feedback_validated / "feedback_tokens.csv", list(token_rows[0]), token_rows)
    assert load_evalset(paths, "hitl_holdout").id == validated.id
    assert load_evalset(paths, "hitl_holdout_user").id == user.id


def test_feedback_evalsets_empty_without_files(tmp_repo):
    paths, _, _ = tmp_repo
    assert load_evalset(paths, "hitl_holdout").n_sentences == 0
    assert load_evalset(paths, "hitl_holdout_user").n_sentences == 0