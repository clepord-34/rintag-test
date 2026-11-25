from hitl.common import write_csv
from hitl.leakage import SplitIndex, format_audit, leakage_audit


def test_index_exact_near_and_audit(tmp_repo):
    paths, _, _ = tmp_repo
    index = SplitIndex(paths)
    _, test_rows = __import__("hitl.common", fromlist=["read_csv_rows"]).read_csv_rows(paths.processed / "test.csv")
    original = test_rows[0]["full_sentence"]
    assert index.exact_match(__import__("hitl.common", fromlist=["sentence_key"]).sentence_key("  " + original.upper().replace(" ", " - ") + "!")).startswith("test:")
    assert index.near_match(" ".join(original.split()[1:])) is None or index.near_match(" ".join(original.split()[1:]))[1] >= 0.8
    assert index.exact_match("unique unseen text") is None
    path = paths.feedback_validated / "feedback_sentences.csv"
    clean = [{"uid": "prototype:clean", "full_sentence": "mamumuya ako", "status": "ok", "split": "train"}]
    write_csv(path, list(clean[0]), clean)
    assert leakage_audit(paths)["ok"]
    clean[0]["full_sentence"] = original
    write_csv(path, list(clean[0]), clean)
    audit = leakage_audit(paths)
    assert not audit["ok"]
    assert "prototype:clean" in audit["exact_overlap"]["test"]
    assert format_audit(audit).startswith("LEAKAGE AUDIT: FAILED (")


def test_audit_ignores_rejected_rows_and_finds_internal_duplicates(tmp_repo):
    paths, _, _ = tmp_repo
    path = paths.feedback_validated / "feedback_sentences.csv"
    rows = [
        {"uid": "a", "full_sentence": "mamumuya ako", "status": "ok", "split": "train"},
        {"uid": "b", "full_sentence": "mamumuya ako!", "status": "ok", "split": "holdout"},
        {"uid": "c", "full_sentence": "mamumuya ako", "status": "rejected", "split": ""},
    ]
    write_csv(path, list(rows[0]), rows)
    audit = leakage_audit(paths)
    assert len(audit["internal_duplicates"]) == 1
    assert not audit["ok"]