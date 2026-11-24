import pickle
import time
import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
from sklearn.model_selection import ParameterGrid, cross_val_score
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from rintag.features import sent2features, sent2labels, resolve_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "src" / "rintag" / "models" / "crf_model.pkl"
VOCAB_PATH = MODEL_PATH.parent / "vocab.pkl"
REGISTRY_PATH = PROJECT_ROOT / "models" / "model_registry.json"
MODEL_ARCHIVE_DIR = PROJECT_ROOT / "models"

# ── Training configuration (record these in the thesis) ──
ALGORITHM = "lbfgs"
MAX_ITERATIONS = 200
PARAMS_SPACE = {
    "c1": [0.01, 0.1, 0.5, 1.0],
    "c2": [0.01, 0.1, 0.5, 1.0],
}
CV_FOLDS = 5
N_JOBS = -1


def fmt_time(seconds):
    """Format seconds as e.g. '42.3s', '5m 07s' or '1h 02m 05s'."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_model_registry(
    dev_f1,
    test_f1,
    test_accuracy,
    test_weighted_f1,
    oov_accuracy,
    oov_correct,
    oov_total,
    test_tokens,
    cv_score,
    cv_std,
    best_params,
    source=None,
    notes=None,
    extra_metadata=None,
):
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            registry = json.load(file)
    else:
        registry = {"active_version": 0, "versions": []}

    versions = registry.get("versions", [])
    version = max(
        [entry["version"] for entry in versions] + [registry.get("active_version", 0)]
    ) + 1

    archived_model = MODEL_ARCHIVE_DIR / f"crf_model_v{version}.pkl"
    archived_vocab = MODEL_ARCHIVE_DIR / f"vocab_v{version}.pkl"
    shutil.copy2(MODEL_PATH, archived_model)
    shutil.copy2(VOCAB_PATH, archived_vocab)

    entry = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dev_macro_f1": round(float(dev_f1), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_weighted_f1": round(float(test_weighted_f1), 4),
        "test_tokens": int(test_tokens),
        "oov_accuracy": round(float(oov_accuracy), 4) if oov_total else None,
        "oov_correct": int(oov_correct),
        "oov_total": int(oov_total),
        "cv_macro_f1": round(float(cv_score), 12),
        "cv_macro_f1_std": round(float(cv_std), 12),
        "best_c1": best_params["c1"],
        "best_c2": best_params["c2"],
        "cv_folds": CV_FOLDS,
        "artifact_sha256": {
            "crf_model": sha256(archived_model),
            "vocab": sha256(archived_vocab),
        },
        "source": source or ("initial_training" if version == 1 else "training"),
        "notes": notes or ("Original baseline model" if version == 1 else "Trained model"),
    }
    if extra_metadata:
        entry.update(extra_metadata)

    versions.append(entry)
    registry["active_version"] = version
    registry["versions"] = versions
    MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)
        file.write("\n")
    print(f"Model registry updated ({REGISTRY_PATH}, active version {version}).")


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_sentences(csv_path):
    """Read a split CSV and rebuild sentences as lists of (word, pos_tag).

    keep_default_na=False stops pandas from turning words such as "nan" or "NA"
    into missing values. The split CSVs were written after pandas had already
    done that to the real word "nan" (a CCONJ), leaving 3 empty word cells in
    train.csv, so empty cells are restored to "nan" here. (The cleaner fix is
    to read annotated_corpus.csv with keep_default_na=False in the script that
    creates train/dev/test and regenerate the splits.)
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["word"] = df["word"].astype(str).replace("", "nan")
    sentences = []
    for _, group in df.groupby("sentence_id"):
        sentences.append(list(zip(group["word"].astype(str), group["pos_tag"].astype(str))))
    return sentences


def make_crf(**params):
    return sklearn_crfsuite.CRF(
        algorithm=ALGORITHM,
        max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True,
        **params,
    )


def describe_features(X_train):
    """Print which feature groups are active and how large the feature space is."""
    groups = resolve_groups()
    n_tokens = sum(len(sent) for sent in X_train)

    templates = set()      # feature names, e.g. "word[-3:]"
    attributes = set()     # name=value pairs actually seen, e.g. ("word[-3:]", "gan")
    total_active = 0
    for sent in X_train:
        for feats in sent:
            total_active += len(feats)
            for name, value in feats.items():
                templates.add(name)
                attributes.add((name, value) if isinstance(value, str) else (name, None))

    banner("FEATURES")
    print("Feature groups (base features are always on):")
    for name, on in groups.items():
        print(f"  [{'x' if on else ' '}] {name}")
    print(f"Feature templates (names)      : {len(templates)}")
    print(f"Distinct feature attributes    : {len(attributes):,}")
    print(f"Avg active features per token  : {total_active / n_tokens:.1f}")


def train_model():
    script_start = time.perf_counter()

    banner("RINTAG CRF TRAINING")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Algorithm    : {ALGORITHM} | max_iterations={MAX_ITERATIONS} | "
          f"all_possible_transitions=True")
    print(f"Grid         : {PARAMS_SPACE} | {CV_FOLDS}-fold CV | n_jobs={N_JOBS}")

    print("\nLoading training data...")
    sentences = load_sentences(PROCESSED_DIR / "train.csv")

    t0 = time.perf_counter()
    X_train = [sent2features([word for word, _ in sent]) for sent in sentences]
    y_train = [sent2labels(sent) for sent in sentences]
    feature_time = time.perf_counter() - t0

    # All possible tags in training data
    labels = list(set([tag for tags in y_train for tag in tags]))
    n_tokens = sum(len(s) for s in sentences)
    print(f"Train: {len(sentences)} sentences, {n_tokens:,} tokens, {len(labels)} tags "
          f"(feature extraction took {fmt_time(feature_time)})")

    describe_features(X_train)

    crf = make_crf()
    f1_scorer = make_scorer(flat_f1_score, average='macro', labels=labels)

    param_grid = list(ParameterGrid(PARAMS_SPACE))
    total = len(param_grid)
    banner(f"PARAMETER SEARCH: {CV_FOLDS}-fold CV, {total} combinations, "
           f"max_iterations={MAX_ITERATIONS}")
    results = []
    search_start = time.perf_counter()
    for combination_number, params in enumerate(param_grid, start=1):
        print(f"\n[{combination_number}/{total}] Training parameters: {params}")
        candidate_crf = crf.set_params(**params)
        combo_start = time.perf_counter()
        scores = cross_val_score(
            candidate_crf,
            X_train,
            y_train,
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            scoring=f1_scorer,
        )
        combo_time = time.perf_counter() - combo_start
        mean_score = scores.mean()
        elapsed = time.perf_counter() - search_start
        print(
            f"[{combination_number}/{total}] Finished parameters: {params}; "
            f"fold scores: {scores}; mean macro F1: {mean_score:.4f} "
            f"(sd {scores.std(ddof=1):.4f})"
        )
        print(f"[{combination_number}/{total}] Time: {fmt_time(combo_time)} | "
              f"elapsed: {fmt_time(elapsed)} ")
        results.append((mean_score, scores.std(ddof=1), params))
    search_time = time.perf_counter() - search_start

    banner("SEARCH RESULTS (best first)")
    for rank, (score, std, params) in enumerate(sorted(results, key=lambda r: r[0], reverse=True), 1):
        print(f"{rank:>2}. macro F1 {score:.4f}  {params}")
    best_score, best_std, best_params = max(results, key=lambda result: result[0])
    print("\nBest parameters:", best_params)
    print(f"Best CV Macro F1: {best_score:.4f}")
    print(f"Parameter search took {fmt_time(search_time)}")

    # Warn if the best value sits on the edge of the grid: the optimum may lie outside it.
    for name, values in PARAMS_SPACE.items():
        if best_params[name] in (min(values), max(values)):
            print(f"Note: best {name}={best_params[name]} is at the edge of the grid {values}.")

    banner("FINAL MODEL")
    best_crf = make_crf(**best_params)
    print(f"Training final model with best parameters: {best_params}")
    fit_start = time.perf_counter()
    best_crf.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    print(f"Final training took {fmt_time(fit_time)}")

    # Did L-BFGS converge, or stop because it hit max_iterations?
    log = getattr(best_crf, "training_log_", None)
    iterations = getattr(log, "iterations", None)
    if iterations:
        used = len(iterations)
        status = ("hit max_iterations, so it may not have converged; consider raising it"
                  if used >= MAX_ITERATIONS else "stopped before the limit (converged)")
        print(f"Iterations used: {used} of {MAX_ITERATIONS} -> {status}")

    banner("DEV SET EVALUATION")
    dev_sentences = load_sentences(PROCESSED_DIR / "dev.csv")

    X_dev = [sent2features([word for word, _ in sent]) for sent in dev_sentences]
    y_dev = [sent2labels(sent) for sent in dev_sentences]

    eval_start = time.perf_counter()
    y_pred = best_crf.predict(X_dev)
    eval_time = time.perf_counter() - eval_start
    dev_f1 = flat_f1_score(y_dev, y_pred, average='macro', labels=labels)
    print(f"Dev: {len(dev_sentences)} sentences, {sum(len(s) for s in dev_sentences):,} tokens "
          f"(prediction took {fmt_time(eval_time)})")
    print(f"Dev Set Macro F1: {dev_f1:.4f}")
    print("\nPer-class report on dev:")
    print(flat_classification_report(y_dev, y_pred, labels=sorted(labels), digits=2))

    print(f"\nSaving model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_crf, f)
    print(f"Model saved successfully ({MODEL_PATH.stat().st_size / 1e6:.1f} MB).")

    # Save training vocabulary for OOV detection
    train_vocab = set(
        w.lower()
        for sent in sentences
        for w, _ in sent
    )
    vocab_path = MODEL_PATH.parent / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(train_vocab, f)
    print(f"Vocabulary saved ({len(train_vocab):,} unique words) to {vocab_path}.")

    banner("TEST SET EVALUATION")
    test_sentences = load_sentences(PROCESSED_DIR / "test.csv")
    X_test = [sent2features([word for word, _ in sent]) for sent in test_sentences]
    y_test = [sent2labels(sent) for sent in test_sentences]
    y_test_pred = best_crf.predict(X_test)

    y_test_flat = [tag for tags in y_test for tag in tags]
    y_test_pred_flat = [tag for tags in y_test_pred for tag in tags]
    test_accuracy = accuracy_score(y_test_flat, y_test_pred_flat)
    test_f1 = f1_score(y_test_flat, y_test_pred_flat, average="macro", labels=labels)
    test_weighted_f1 = f1_score(
        y_test_flat,
        y_test_pred_flat,
        average="weighted",
        labels=labels,
    )

    oov_correct = 0
    oov_total = 0
    for sentence, predictions in zip(test_sentences, y_test_pred):
        for (word, true_tag), predicted_tag in zip(sentence, predictions):
            if word.lower() not in train_vocab:
                oov_total += 1
                oov_correct += int(true_tag == predicted_tag)
    oov_accuracy = oov_correct / oov_total if oov_total else 0.0

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"OOV Accuracy: {oov_accuracy:.4f} ({oov_correct}/{oov_total})")

    update_model_registry(
        dev_f1=dev_f1,
        test_f1=test_f1,
        test_accuracy=test_accuracy,
        test_weighted_f1=test_weighted_f1,
        oov_accuracy=oov_accuracy,
        oov_correct=oov_correct,
        oov_total=oov_total,
        test_tokens=len(y_test_flat),
        cv_score=best_score,
        cv_std=best_std,
        best_params=best_params,
    )

    banner("TIMING SUMMARY")
    print(f"Feature extraction : {fmt_time(feature_time)}")
    print(f"Parameter search   : {fmt_time(search_time)} ({total} combinations x {CV_FOLDS} folds)")
    print(f"Final training     : {fmt_time(fit_time)}")
    print(f"Total              : {fmt_time(time.perf_counter() - script_start)}")


if __name__ == "__main__":
    train_model()