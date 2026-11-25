from hitl.common import read_csv_rows
from hitl.ingest import run_ingest


def test_policy_labels_and_blank_user_defaults_to_model(tmp_repo):
    paths, write_feedback, candidates = tmp_repo
    sentence = candidates(1)
    input_path = paths.root / "prototype_feedback.csv"
    write_feedback(input_path, sentence, validator=False, user=False)
    report = run_ingest(paths, [input_path.name], "validator")
    assert report["n_validated"] == 0
    assert report["n_trainable"] == 0
    _, tokens = read_csv_rows(paths.feedback_validated / "feedback_tokens.csv")
    assert all(row["user_effective_tag"] == row["model_tag"] for row in tokens)
    run_ingest(paths, [input_path.name], "validator_then_user")
    _, tokens = read_csv_rows(paths.feedback_validated / "feedback_tokens.csv")
    assert all(row["final_tag"] == row["model_tag"] for row in tokens)
    report = run_ingest(paths, [input_path.name], "user")
    assert report["n_trainable"] == 1
    assert report["experimental"] is True


def test_all_accepted_trainable_feedback_is_assigned_to_training(tmp_repo):
    paths, write_feedback, candidates = tmp_repo
    input_path = paths.root / "prototype_feedback.csv"
    write_feedback(input_path, candidates(1), validator=False, user=True)

    report = run_ingest(paths, [input_path.name], "user")
    _, sentences = read_csv_rows(paths.feedback_validated / "feedback_sentences.csv")

    assert report["n_train"] == report["n_trainable"] == report["n_ok"] == 1
    assert sentences[0]["split"] == "train"
    assert not any(key.startswith("n_holdout") for key in report)


def test_duplicate_ids_are_namespaced_and_validated_copy_wins(tmp_repo):
    paths, write_feedback, candidates = tmp_repo
    sentence = candidates(1)
    prototype = paths.root / "prototype_feedback.csv"
    library = paths.root / "library_feedback.csv"
    write_feedback(prototype, sentence, validator=False)
    write_feedback(library, sentence, validator=True)
    report = run_ingest(paths, [prototype.name, library.name], "user")
    assert report["n_candidates"] == 2
    assert report["n_ok"] == 1
    _, rows = read_csv_rows(paths.feedback_validated / "feedback_sentences.csv")
    assert {row["uid"] for row in rows} == {"prototype:1", "library:1"}
    kept = next(row for row in rows if row["status"] == "ok")
    rejected = next(row for row in rows if row["status"] == "rejected")
    assert kept["uid"] == "library:1"
    assert rejected["reason"] == "duplicate_of:library:1"
    assert rejected["validated"] == rejected["trainable"] == ""


def test_corpus_exact_and_near_duplicates_are_rejected(tmp_repo):
    paths, write_feedback, _ = tmp_repo
    _, corpus_rows = read_csv_rows(paths.processed / "test.csv")
    text = corpus_rows[0]["full_sentence"]
    exact_path = paths.root / "prototype_feedback.csv"
    write_feedback(exact_path, [("testleak", text.upper(), {})])
    run_ingest(paths, [exact_path.name], "user")
    _, rows = read_csv_rows(paths.feedback_validated / "feedback_sentences.csv")
    assert rows[0]["reason"] == f"in_test:{corpus_rows[0]['sentence_id']}"

    near_text = " ".join(text.split()[1:])
    write_feedback(exact_path, [("near", near_text, {})])
    run_ingest(paths, [exact_path.name], "user")
    _, rows = read_csv_rows(paths.feedback_validated / "feedback_sentences.csv")
    assert rows[0]["reason"].startswith("near_dup_test:")