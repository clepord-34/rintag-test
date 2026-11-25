"""Train and archive a new CRF version from corpus plus HITL feedback."""

import copy
import importlib.metadata
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import sklearn_crfsuite

from hitl.common import (
    ALGORITHM, LABEL_POLICIES, MAX_ITERATIONS, TAGSET, HitlError,
    read_csv_rows, sha256_file, utc_now_iso, write_json_atomic,
)
from hitl.evalsets import load_corpus_train
from hitl.ingest import run_ingest
from hitl.leakage import format_audit, leakage_audit
from hitl.registry import append_version, get_entry, latest_version, load_registry, next_version, save_registry
from rintag.features import DEFAULT_GROUPS, sent2features


def make_crf(**params):
    return sklearn_crfsuite.CRF(
        algorithm=ALGORITHM,
        max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True,
        **params,
    )


def _library_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(root):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _cleanup(paths, version, original_registry):
    for path in (paths.model_file(version), paths.vocab_file(version), paths.version_results(version), paths.snapshot(version)):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    save_registry(paths, original_registry)


def _feedback_training(paths):
    _, sentence_rows = read_csv_rows(paths.feedback_validated / "feedback_sentences.csv")
    _, token_rows = read_csv_rows(paths.feedback_validated / "feedback_tokens.csv")
    train_uids = [row["uid"] for row in sentence_rows if row["status"] == "ok" and row["trainable"] == "true"]
    selected = set(train_uids)
    grouped = {uid: [] for uid in train_uids}
    for row in token_rows:
        if row["uid"] in selected:
            grouped[row["uid"]].append(row)
    sentences = []
    for uid in train_uids:
        ordered = sorted(grouped[uid], key=lambda row: int(row["token_index"]))
        sentences.append([(row["token"], row["final_tag"]) for row in ordered])
    return sentences, train_uids


def retrain(paths, inputs=None, label_policy="validator", parent=None, c1=None, c2=None, min_sentences=1):
    if label_policy not in LABEL_POLICIES:
        raise HitlError(f"unknown label policy: {label_policy}")
    report = run_ingest(paths, inputs, label_policy)
    audit = leakage_audit(paths)
    if not audit["ok"]:
        raise HitlError(format_audit(audit))
    n_train = report["n_train"]
    if n_train < min_sentences:
        message = f"only {n_train} trainable feedback sentences; need at least {min_sentences}"
        if label_policy == "validator":
            message += ". Validators have not tagged enough sentences yet. For an experiment without validator tags use: retrain --label-policy user"
        raise HitlError(message)

    original_registry = copy.deepcopy(load_registry(paths))
    version = next_version(original_registry)
    parent_version = parent or latest_version(original_registry)
    parent_entry = get_entry(original_registry, parent_version)
    c1 = parent_entry["best_c1"] if c1 is None else c1
    c2 = parent_entry["best_c2"] if c2 is None else c2
    output_paths = [paths.model_file(version), paths.vocab_file(version), paths.version_results(version), paths.snapshot(version)]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise HitlError("refusing to overwrite existing candidate paths: " + ", ".join(existing))

    run_started = time.perf_counter()
    try:
        corpus = load_corpus_train(paths)
        feedback, train_uids = _feedback_training(paths)
        sentences = corpus + feedback
        X = [sent2features([word for word, _ in sentence]) for sentence in sentences]
        y = [[tag for _, tag in sentence] for sentence in sentences]
        invalid = sorted({tag for tags in y for tag in tags if tag not in TAGSET})
        if invalid:
            raise HitlError(f"training labels outside TAGSET: {invalid}")
        model = make_crf(c1=c1, c2=c2)
        model.fit(X, y)
        vocab = {word.lower() for sentence in sentences for word, _ in sentence}

        with open(paths.model_file(version), "wb") as file:
            pickle.dump(model, file)
        with open(paths.vocab_file(version), "wb") as file:
            pickle.dump(vocab, file)

        snapshot = paths.snapshot(version)
        snapshot.mkdir(parents=True)
        for name in ("feedback_sentences.csv", "feedback_tokens.csv", "ingest_report.json"):
            shutil.copy2(paths.feedback_validated / name, snapshot / name)
        raw_dir = snapshot / "raw"
        raw_dir.mkdir()
        for item in report["inputs"]:
            shutil.copy2(item["path"], raw_dir / Path(item["path"]).name)

        artifact_hashes = {"crf_model": sha256_file(paths.model_file(version)), "vocab": sha256_file(paths.vocab_file(version))}
        placeholder = {
            "version": version, "dev_macro_f1": 0.0, "test_accuracy": 0.0,
            "test_macro_f1": 0.0, "test_weighted_f1": 0.0, "test_tokens": 0,
            "oov_accuracy": None, "oov_correct": 0, "oov_total": 0,
            "artifact_sha256": artifact_hashes,
        }
        temporary_registry = copy.deepcopy(original_registry)
        temporary_registry.setdefault("versions", []).append(placeholder)
        save_registry(paths, temporary_registry)

        from hitl.evaluate import evaluate_version
        evaluations = {}
        for name in ("test", "dev"):
            evaluations[name] = evaluate_version(paths, version, name)
        save_registry(paths, original_registry)

        test = evaluations["test"]
        dev = evaluations["dev"]
        experimental = label_policy != "validator"
        prefix = f"EXPERIMENTAL (policy={label_policy}, unvalidated labels): " if experimental else ""
        notes = prefix + f"CRF retrained on corpus train + {len(feedback)} feedback sentences (policy={label_policy})"
        entry = {
            "version": version, "timestamp": utc_now_iso(), "dev_macro_f1": round(dev["macro_f1"], 4),
            "test_accuracy": round(test["accuracy"], 4), "test_macro_f1": round(test["macro_f1"], 4),
            "test_weighted_f1": round(test["weighted_f1"], 4), "test_tokens": test["n_tokens"],
            "oov_accuracy": round(test["oov_own_vocab"]["accuracy"], 4) if test["oov_own_vocab"]["accuracy"] is not None else None,
            "oov_correct": test["oov_own_vocab"]["correct"], "oov_total": test["oov_own_vocab"]["total"],
            "cv_macro_f1": None, "cv_macro_f1_std": None, "best_c1": c1, "best_c2": c2,
            "cv_folds": None, "artifact_sha256": artifact_hashes, "source": "hitl_retrain", "notes": notes,
            "parent_version": parent_version, "label_policy": label_policy, "experimental": experimental,
            "n_feedback_train_sentences": len(feedback),
            "n_feedback_train_tokens": sum(len(sentence) for sentence in feedback),
            "leakage_audit_ok": True, "snapshot_dir": f"data/feedback/snapshots/v{version}",
            "results_dir": f"models/results/v{version}",
        }
        training_seconds = time.perf_counter() - run_started
        manifest = {
            "version": version, "created_utc": utc_now_iso(), "parent_version": parent_version,
            "label_policy": label_policy, "experimental": experimental,
            "hyperparameters": {"algorithm": ALGORITHM, "max_iterations": MAX_ITERATIONS,
                                "all_possible_transitions": True, "c1": c1, "c2": c2, "tuned": False},
            "feature_groups": dict(DEFAULT_GROUPS),
            "features_py_sha256": sha256_file(paths.root / "src" / "rintag" / "features.py"),
            "corpus": {
                "train_csv_sha256": sha256_file(paths.processed / "train.csv"),
                "dev_csv_sha256": sha256_file(paths.processed / "dev.csv"),
                "test_csv_sha256": sha256_file(paths.processed / "test.csv"),
                "n_train_sentences": len(corpus),
            },
            "feedback": {
                "inputs": [{"path": item["path"], "sha256": item["sha256"]} for item in report["inputs"]],
                "snapshot_dir": entry["snapshot_dir"], "n_train_sentences": len(feedback),
                "n_train_tokens": entry["n_feedback_train_tokens"], "train_uids": train_uids,
            },
            "leakage_audit": audit,
            "libraries": {"python": sys.version.split()[0], "scikit_learn": _library_version("scikit-learn"),
                          "sklearn_crfsuite": _library_version("sklearn-crfsuite"),
                          "python_crfsuite": _library_version("python-crfsuite")},
            "git_commit": _git_commit(paths.root), "training_seconds": training_seconds,
        }
        write_json_atomic(paths.version_results(version) / "train_manifest.json", manifest)
        append_version(paths, entry)
    except Exception:
        _cleanup(paths, version, original_registry)
        raise

    print(f"v{parent_version} -> v{version}: test acc={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}")
    return version