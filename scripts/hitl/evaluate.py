"""Version evaluation and reproducibility verification."""

import pickle

from hitl.common import HitlError, read_csv_rows, sha256_file, utc_now_iso, write_csv, write_json_atomic
from hitl.evalsets import EvalSet, load_evalset
from hitl.metrics import compute_metrics, predict_with_confidence
from hitl.registry import baseline_version, get_entry, load_registry


def load_model_and_vocab(paths, version):
    model_path, vocab_path = paths.model_file(version), paths.vocab_file(version)
    if not model_path.exists() or not vocab_path.exists():
        raise HitlError(f"model artifacts missing for v{version}")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)
    return model, vocab


def _cache_paths(paths, version, evalset):
    directory = paths.version_results(version)
    short_id = evalset.id[:12]
    return directory / f"eval_{evalset.name}_{short_id}.json", directory / f"pred_{evalset.name}_{short_id}.csv"


def evaluate_version(paths, version, evalset_name, force=False):
    registry = load_registry(paths)
    get_entry(registry, version)
    if not paths.model_file(version).exists() or not paths.vocab_file(version).exists():
        raise HitlError(f"model artifacts missing for v{version}")
    evalset = load_evalset(paths, evalset_name)
    if evalset.n_sentences == 0:
        return None
    metrics_path, predictions_path = _cache_paths(paths, version, evalset)
    if metrics_path.exists() and not force:
        from hitl.common import read_json
        return read_json(metrics_path)
    model, vocab = load_model_and_vocab(paths, version)
    base_model, baseline_vocab = load_model_and_vocab(paths, baseline_version(registry))
    del base_model
    pred, conf = predict_with_confidence(model, evalset.tokens)
    metrics = compute_metrics(evalset, pred, conf, vocab, baseline_vocab)
    output = {
        "version": version, "evalset": evalset.name, "evalset_id": evalset.id,
        "n_sentences": evalset.n_sentences, "n_tokens": evalset.n_tokens,
        **metrics, "created_utc": utc_now_iso(),
    }
    rows = []
    for uid, tokens, gold, predictions, confidences in zip(evalset.uids, evalset.tokens, evalset.gold, pred, conf):
        for token_index, (token, gold_tag, pred_tag, confidence) in enumerate(zip(tokens, gold, predictions, confidences)):
            rows.append({"uid": uid, "token_index": token_index, "token": token, "gold": gold_tag,
                         "pred": pred_tag, "confidence": round(float(confidence), 6)})
    write_csv(predictions_path, ["uid", "token_index", "token", "gold", "pred", "confidence"], rows)
    write_json_atomic(metrics_path, output)
    return output


def load_predictions(paths, version, evalset: EvalSet):
    metrics_path, predictions_path = _cache_paths(paths, version, evalset)
    if not predictions_path.exists():
        raise HitlError(f"prediction cache not found: {predictions_path}")
    _, rows = read_csv_rows(predictions_path)
    grouped = {uid: [] for uid in evalset.uids}
    for row in rows:
        if row["uid"] not in grouped:
            raise HitlError(f"prediction cache contains unexpected uid: {row['uid']}")
        grouped[row["uid"]].append(row)
    predictions, confidences = [], []
    for uid, expected_tokens in zip(evalset.uids, evalset.tokens):
        sentence_rows = sorted(grouped[uid], key=lambda row: int(row["token_index"]))
        actual_tokens = [row["token"] for row in sentence_rows]
        if actual_tokens != expected_tokens:
            raise HitlError(f"prediction tokens do not match eval set for {uid}")
        predictions.append([row["pred"] for row in sentence_rows])
        confidences.append([float(row["confidence"]) for row in sentence_rows])
    return predictions, confidences


def verify_version(paths, version):
    registry = load_registry(paths)
    entry = get_entry(registry, version)
    problems = []
    for label, path, key in (
        ("crf_model", paths.model_file(version), "crf_model"),
        ("vocab", paths.vocab_file(version), "vocab"),
    ):
        if not path.exists():
            problems.append(f"missing {label} artifact: {path}")
        elif sha256_file(path) != entry.get("artifact_sha256", {}).get(key):
            problems.append(f"{label} artifact SHA-256 does not match registry")
    try:
        test = evaluate_version(paths, version, "test", force=True)
        if test is None:
            problems.append("test eval set is empty")
        else:
            for result_key, registry_key in (("accuracy", "test_accuracy"), ("macro_f1", "test_macro_f1"), ("weighted_f1", "test_weighted_f1")):
                if round(test[result_key], 4) != round(entry[registry_key], 4):
                    problems.append(f"{registry_key}: registry={entry[registry_key]} reproduced={test[result_key]}")
            if test["n_tokens"] != entry["test_tokens"]:
                problems.append(f"test_tokens: registry={entry['test_tokens']} reproduced={test['n_tokens']}")
            own = test["oov_own_vocab"]
            if own["correct"] != entry["oov_correct"] or own["total"] != entry["oov_total"]:
                problems.append(f"OOV counts: registry={entry['oov_correct']}/{entry['oov_total']} reproduced={own['correct']}/{own['total']}")
        dev = evaluate_version(paths, version, "dev", force=True)
        if dev is None or round(dev["macro_f1"], 4) != round(entry["dev_macro_f1"], 4):
            problems.append(f"dev_macro_f1: registry={entry['dev_macro_f1']} reproduced={None if dev is None else dev['macro_f1']}")
    except Exception as exc:
        problems.append(f"evaluation failed: {exc}")
    return problems