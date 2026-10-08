import pandas as pd
import sklearn_crfsuite
import hashlib
import json
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import make_scorer
from sklearn_crfsuite.metrics import flat_f1_score
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from rintag.features import sent2features, sent2labels


MAX_ITERATIONS = 200
PARAMS_SPACE = {
    "c1": [0.01, 0.1, 0.5, 1.0],
    "c2": [0.01, 0.1, 0.5, 1.0],
}


def _cache_metadata(train_path):
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    return {
        "training_data_sha256": train_hash,
        "params_space": PARAMS_SPACE,
        "max_iterations": MAX_ITERATIONS,
        "folds": 5,
        "algorithm": "lbfgs",
        "all_possible_transitions": True,
        "scoring": "flat_f1_macro",
    }


def _has_valid_cache(results_path, metadata_path, metadata):
    if not results_path.is_file() or results_path.stat().st_size == 0:
        return False
    try:
        with metadata_path.open(encoding="utf-8") as file:
            return json.load(file) == metadata
    except (OSError, json.JSONDecodeError):
        return False


def cross_validation():
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    results_dir = PROJECT_ROOT / "experiments" / "results"
    results_path = results_dir / "cv_results.csv"
    metadata_path = results_dir / "cv_results.meta.json"
    metadata = _cache_metadata(train_path)

    if _has_valid_cache(results_path, metadata_path, metadata):
        print("Training data and CV settings unchanged; reusing cv_results.csv.")
        return

    print("Loading training data...")
    train_df = pd.read_csv(
        train_path,
        keep_default_na=False,
    )
    
    sentences = []
    for _, group in train_df.groupby("sentence_id"):
        sentences.append(list(zip(group["word"].astype(str), group["pos_tag"].astype(str))))
    
    X_train = [sent2features([word for word, _ in sent]) for sent in sentences]
    y_train = [sent2labels(sent) for sent in sentences]
    labels = list(set([tag for tags in y_train for tag in tags]))
    
    f1_scorer = make_scorer(flat_f1_score, average='macro', labels=labels)

    print("Running 5-fold CV for the training grid...")
    rows = []
    for params in ParameterGrid(PARAMS_SPACE):
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            max_iterations=MAX_ITERATIONS,
            all_possible_transitions=True,
            **params,
        )
        scores = cross_val_score(
            crf, X_train, y_train, cv=5, scoring=f1_scorer, n_jobs=-1
        )
        print(
            f"c1={params['c1']}, c2={params['c2']}: "
            f"{scores.mean():.4f} +/- {scores.std():.4f}"
        )
        rows.append({
            "c1": params["c1"],
            "c2": params["c2"],
            "macro_f1_mean": scores.mean(),
            "macro_f1_std": scores.std(),
        })

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).to_csv(
        results_path, index=False
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

if __name__ == "__main__":
    cross_validation()
