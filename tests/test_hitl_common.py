from pathlib import Path

from hitl import common


def test_sentence_keys():
    assert common.sentence_key("...") == ""
    assert common.sentence_key("Nag-dalagan  SU igin!") == "nag dalagan su igin"


def test_jaccard():
    assert common.jaccard(common.token_set("a b c d e"), common.token_set("a b c d e f")) > 0.8
    assert common.jaccard(common.token_set("a b c"), common.token_set("a b c d")) == 0.75
    assert common.jaccard(frozenset(), frozenset({"a"})) == 0.0


def test_constants_match_original_trainer():
    trainer = (common.REPO_ROOT / "scripts" / "05_train_model.py").read_text(encoding="utf-8")
    assert 'ALGORITHM = "lbfgs"' in trainer
    assert "MAX_ITERATIONS = 200" in trainer
    assert common.ALGORITHM == "lbfgs"
    assert common.MAX_ITERATIONS == 200


def test_json_atomic_and_nan_csv_round_trip(tmp_path):
    path = tmp_path / "data.json"
    common.write_json_atomic(path, {"ok": True})
    assert common.read_json(path) == {"ok": True}
    assert not Path(str(path) + ".tmp").exists()
    csv_path = tmp_path / "tokens.csv"
    common.write_csv(csv_path, ["token"], [{"token": "nan"}])
    assert common.read_csv_rows(csv_path)[1][0]["token"] == "nan"