import pickle
import time
from pathlib import Path

import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
from sklearn.model_selection import ParameterGrid, cross_val_score
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from rintag.features import sent2features, sent2labels, resolve_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "src" / "rintag" / "models" / "crf_model.pkl"

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
        results.append((mean_score, params))
    search_time = time.perf_counter() - search_start

    banner("SEARCH RESULTS (best first)")
    for rank, (score, params) in enumerate(sorted(results, key=lambda r: r[0], reverse=True), 1):
        print(f"{rank:>2}. macro F1 {score:.4f}  {params}")
    best_score, best_params = max(results, key=lambda result: result[0])
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

    banner("TIMING SUMMARY")
    print(f"Feature extraction : {fmt_time(feature_time)}")
    print(f"Parameter search   : {fmt_time(search_time)} ({total} combinations x {CV_FOLDS} folds)")
    print(f"Final training     : {fmt_time(fit_time)}")
    print(f"Total              : {fmt_time(time.perf_counter() - script_start)}")


if __name__ == "__main__":
    train_model()