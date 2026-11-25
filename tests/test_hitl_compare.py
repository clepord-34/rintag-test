import copy
import shutil

from hitl.common import TAGSET, read_csv_rows, read_json, sha256_file, write_csv, write_json_atomic
from hitl.compare import LEGEND, compare
from hitl.evalsets import load_evalset
from hitl.evaluate import evaluate_version


def _add_identical_v2(paths):
    shutil.copy2(paths.model_file(1), paths.model_file(2))
    shutil.copy2(paths.vocab_file(1), paths.vocab_file(2))
    registry = read_json(paths.registry)
    entry = copy.deepcopy(registry["versions"][0])
    entry.update({
        "version": 2, "parent_version": 1, "label_policy": "user",
        "experimental": True, "n_feedback_train_sentences": 0,
        "artifact_sha256": {"crf_model": sha256_file(paths.model_file(2)), "vocab": sha256_file(paths.vocab_file(2))},
    })
    registry["versions"].append(entry)
    write_json_atomic(paths.registry, registry)


def test_comparison_and_token_regression_reporting(tmp_repo):
    paths, _, _ = tmp_repo
    _add_identical_v2(paths)
    test = load_evalset(paths, "test")
    evaluate_version(paths, 1, "test")
    evaluate_version(paths, 2, "test")
    meta = compare(paths, [2, 1], resamples=20)
    assert meta["versions"] == [1, 2]
    assert meta["leakage_audit_ok"] == {}
    _, pair_rows = read_csv_rows(paths.compare_out / "pairwise_test.csv")
    assert pair_rows[0]["n_fixed"] == pair_rows[0]["n_regressed"] == "0"
    assert float(pair_rows[0]["mcnemar_p"]) == 1.0
    assert float(pair_rows[0]["delta_accuracy"]) == 0.0
    assert (paths.compare_out / "version_comparison.txt").read_text(encoding="utf-8").rstrip().endswith(LEGEND)
    comparison = read_csv_rows(paths.compare_out / "version_comparison.csv")[1]
    assert len(comparison) == 2
    assert comparison[1]["experimental"] == "True"

    pred_path = next(paths.version_results(2).glob("pred_test_*.csv"))
    _, rows = read_csv_rows(pred_path)
    correct = next(row for row in rows if row["pred"] == row["gold"])
    correct["pred"] = next(tag for tag in TAGSET if tag != correct["gold"])
    write_csv(pred_path, ["uid", "token_index", "token", "gold", "pred", "confidence"], rows)
    compare(paths, [1, 2], resamples=20)
    _, pair_rows = read_csv_rows(paths.compare_out / "pairwise_test.csv")
    assert pair_rows[0]["n_regressed"] == "1"
    assert pair_rows[0]["n_fixed"] == "0"
    assert pair_rows[0]["net_fixed"] == "-1"
    change_path = paths.compare_out / "changes_test_v1_to_v2.csv"
    _, changes = read_csv_rows(change_path)
    assert len(changes) == 1
    assert changes[0]["kind"] == "regressed"


def test_single_version_comparison_writes_no_pairwise_files(tmp_repo):
    paths, _, _ = tmp_repo
    meta = compare(paths, [1], resamples=5)
    assert meta["versions"] == [1]
    assert (paths.compare_out / "version_comparison.csv").exists()
    assert not list(paths.compare_out.glob("pairwise_*.csv"))