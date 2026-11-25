"""Prediction confidence, standard metrics, and paired uncertainty helpers."""

import math

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from hitl.common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, LOW_CONF_THRESHOLD
from rintag.features import sent2features


def predict_with_confidence(model, tokens_list):
    """-> (pred_tags, confidences), both list[list]. Empty input -> ([], [])."""
    if not tokens_list:
        return [], []
    X = [sent2features(t) for t in tokens_list]
    pred = model.predict(X)
    marg = model.predict_marginals(X)
    conf = [[m[i][p[i]] for i in range(len(p))] for m, p in zip(marg, pred)]
    return pred, conf


def sentence_counts(gold_sents, pred_sents, labels):
    """Per-sentence tp/fp/fn per label (labels outside `labels` are ignored, like sklearn's labels=)."""
    idx = {t: i for i, t in enumerate(labels)}
    n, L = len(gold_sents), len(labels)
    tp = np.zeros((n, L)); fp = np.zeros((n, L)); fn = np.zeros((n, L))
    correct = np.zeros(n); ntok = np.zeros(n)
    for s, (g, p) in enumerate(zip(gold_sents, pred_sents)):
        for gt, pt in zip(g, p):
            ntok[s] += 1
            gi, pi = idx.get(gt), idx.get(pt)
            if gt == pt:
                correct[s] += 1
                if gi is not None: tp[s, gi] += 1
            else:
                if pi is not None: fp[s, pi] += 1
                if gi is not None: fn[s, gi] += 1
    return {"tp": tp, "fp": fp, "fn": fn, "correct": correct, "ntok": ntok}


def metrics_from_counts(c, rows=None):
    """-> (accuracy, macro_f1). `rows` = optional array of sentence indices (bootstrap resample)."""
    sel = (lambda a: a) if rows is None else (lambda a: a[rows])
    tp, fp, fn = (sel(c[k]).sum(axis=0) for k in ("tp", "fp", "fn"))
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    total = sel(c["ntok"]).sum()
    accuracy = sel(c["correct"]).sum() / total if total else 0.0
    return float(accuracy), float(f1.mean()) if len(f1) else 0.0


def bootstrap_indices(n_sentences, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    return np.random.default_rng(seed).integers(0, n_sentences, size=(resamples, n_sentences))


def bootstrap_distribution(counts, idx):
    accs = np.empty(len(idx)); f1s = np.empty(len(idx))
    for b in range(len(idx)):
        accs[b], f1s[b] = metrics_from_counts(counts, idx[b])
    return accs, f1s


def ci95(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar p-value. Sentence-level paired bootstrap is the primary uncertainty measure because tokens are not independent."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compute_metrics(evalset, pred, conf, vocab_own, vocab_baseline):
    labels = sorted({tag for sentence in evalset.gold for tag in sentence})
    flat_gold = [tag for sentence in evalset.gold for tag in sentence]
    flat_pred = [tag for sentence in pred for tag in sentence]
    flat_conf = [value for sentence in conf for value in sentence]
    counts = sentence_counts(evalset.gold, pred, labels)
    accuracy, macro_f1 = metrics_from_counts(counts)
    weighted_f1 = f1_score(flat_gold, flat_pred, average="weighted", labels=labels, zero_division=0) if labels else 0.0
    precision, recall, f1, support = precision_recall_fscore_support(flat_gold, flat_pred, labels=labels, zero_division=0)
    per_tag = {
        tag: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, tag in enumerate(labels)
    }
    own_total = own_correct = baseline_total = baseline_correct = 0
    for token_s, gold_s, pred_s in zip(evalset.tokens, evalset.gold, pred):
        for token, gold_tag, pred_tag in zip(token_s, gold_s, pred_s):
            if token.lower() not in vocab_own:
                own_total += 1
                own_correct += gold_tag == pred_tag
            if token.lower() not in vocab_baseline:
                baseline_total += 1
                baseline_correct += gold_tag == pred_tag
    return {
        "labels": labels, "accuracy": accuracy, "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "low_conf_rate": sum(value < LOW_CONF_THRESHOLD for value in flat_conf) / len(flat_conf) if flat_conf else 0.0,
        "mean_confidence": float(np.mean(flat_conf)) if flat_conf else 0.0,
        "oov_own_vocab": {"correct": int(own_correct), "total": own_total, "accuracy": own_correct / own_total if own_total else None},
        "oov_baseline_vocab": {"correct": int(baseline_correct), "total": baseline_total, "accuracy": baseline_correct / baseline_total if baseline_total else None},
        "per_tag": per_tag,
    }