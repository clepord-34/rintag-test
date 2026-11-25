import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from hitl.evalsets import EvalSet, compute_evalset_id
from hitl.metrics import bootstrap_indices, mcnemar_exact_p, metrics_from_counts, sentence_counts


def test_counts_match_sklearn_with_subset_labels():
    gold = [["NOUN", "VERB"], ["ADJ", "NOUN"]]
    pred = [["NOUN", "ADJ"], ["VERB", "NOUN"]]
    labels = ["NOUN", "VERB"]
    accuracy, macro = metrics_from_counts(sentence_counts(gold, pred, labels))
    flat_gold = sum(gold, [])
    flat_pred = sum(pred, [])
    assert accuracy == accuracy_score(flat_gold, flat_pred)
    assert macro == f1_score(flat_gold, flat_pred, average="macro", labels=labels, zero_division=0)


def test_mcnemar_and_bootstrap_helpers():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert abs(mcnemar_exact_p(10, 2) - binomtest(2, 12, 0.5).pvalue) < 1e-9
    assert mcnemar_exact_p(10, 2) == mcnemar_exact_p(2, 10)
    first = bootstrap_indices(5, 10, 1)
    assert first.shape == (10, 5)
    assert np.array_equal(first, bootstrap_indices(5, 10, 1))


def test_evalset_id_changes_with_gold():
    one = compute_evalset_id(["x"], [["ako"]], [["PRON"]])
    two = compute_evalset_id(["x"], [["ako"]], [["NOUN"]])
    assert one != two