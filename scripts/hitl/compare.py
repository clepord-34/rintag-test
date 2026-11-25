"""Paired, version-wide HITL comparison outputs."""

import math
from collections import Counter

import pandas as pd

from hitl.common import HitlError, read_json, write_csv, write_json_atomic, utc_now_iso
from hitl.evalsets import load_evalset
from hitl.leakage import leakage_audit
from hitl.metrics import (
    bootstrap_distribution, bootstrap_indices, ci95, mcnemar_exact_p,
    sentence_counts,
)
from hitl.registry import get_entry, load_registry

LEGEND = (
    "gold_source: corpus_annotation = frozen corpus test/dev annotations.\n"
    "experimental=True: the version was trained WITHOUT validator-only labels (label_policy user / validator_then_user). Not thesis-grade; do not promote."
)
GOLD_SOURCE = {"test": "corpus_annotation", "dev": "corpus_annotation"}


def _rounded(value):
    return round(value, 4) if value is not None else None


def _write_frame(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def compare(paths, versions=None, resamples=1000):
    registry = load_registry(paths)
    selected = sorted(versions if versions is not None else [entry["version"] for entry in registry["versions"]])
    if not selected:
        raise HitlError("no versions selected for comparison")
    for version in selected:
        get_entry(registry, version)
    paths.compare_out.mkdir(parents=True, exist_ok=True)
    for path in paths.compare_out.glob("*hitl_holdout*.csv"):
        path.unlink()
    version_rows = []
    eval_context = {}
    metrics_by_pair = {}
    prediction_by_pair = {}
    counts_by_pair = {}
    distributions = {}
    for name in ("test", "dev"):
        evalset = load_evalset(paths, name)
        if not evalset.n_sentences:
            print(f"NOTE: {name} eval set is empty; skipped")
            continue
        idx = bootstrap_indices(evalset.n_sentences, resamples)
        eval_context[name] = (evalset, idx)
        for version in selected:
            metrics_path = paths.version_results(version) / f"eval_{name}_{evalset.id[:12]}.json"
            metrics = __import__("hitl.evaluate", fromlist=["evaluate_version"]).evaluate_version(paths, version, name)
            metrics_by_pair[(version, name)] = metrics
            predictions, _ = __import__("hitl.evaluate", fromlist=["load_predictions"]).load_predictions(paths, version, evalset)
            prediction_by_pair[(version, name)] = predictions
            labels = metrics["labels"]
            counts = sentence_counts(evalset.gold, predictions, labels)
            counts_by_pair[(version, name)] = counts
            accs, f1s = bootstrap_distribution(counts, idx)
            distributions[(version, name)] = (accs, f1s)
            entry = get_entry(registry, version)
            version_rows.append({
                "version": version, "evalset": name, "gold_source": GOLD_SOURCE[name],
                "evalset_id": evalset.id, "n_sentences": evalset.n_sentences,
                "n_tokens": evalset.n_tokens, "accuracy": _rounded(metrics["accuracy"]),
                "accuracy_ci_low": _rounded(ci95(accs)[0]), "accuracy_ci_high": _rounded(ci95(accs)[1]),
                "macro_f1": _rounded(metrics["macro_f1"]), "macro_f1_ci_low": _rounded(ci95(f1s)[0]),
                "macro_f1_ci_high": _rounded(ci95(f1s)[1]), "weighted_f1": _rounded(metrics["weighted_f1"]),
                "oov_baseline_vocab_accuracy": _rounded(metrics["oov_baseline_vocab"]["accuracy"]),
                "oov_baseline_vocab_total": metrics["oov_baseline_vocab"]["total"],
                "low_conf_rate": _rounded(metrics["low_conf_rate"]), "mean_confidence": _rounded(metrics["mean_confidence"]),
                "parent_version": entry.get("parent_version", ""), "label_policy": entry.get("label_policy", ""),
                "experimental": entry.get("experimental", False),
                "n_feedback_train_sentences": entry.get("n_feedback_train_sentences", ""),
            })

        tags = sorted({tag for sentence in evalset.gold for tag in sentence})
        gold_support = Counter(tag for sentence in evalset.gold for tag in sentence)
        per_tag_rows = []
        for tag in tags:
            row = {"tag": tag, "support": gold_support[tag]}
            values = {}
            for version in selected:
                metrics = metrics_by_pair[(version, name)]
                value = metrics["per_tag"].get(tag, {}).get("f1")
                row[f"f1_v{version}"] = _rounded(value)
                values[version] = value
            for i, version in enumerate(selected[1:], start=1):
                previous = selected[i - 1]
                row[f"delta_prev_v{version}"] = _rounded(values[version] - values[previous]) if values[version] is not None and values[previous] is not None else ""
                base = selected[0]
                row[f"delta_base_v{version}"] = _rounded(values[version] - values[base]) if values[version] is not None and values[base] is not None else ""
            per_tag_rows.append(row)
        _write_frame(paths.compare_out / f"per_tag_f1_{name}.csv", per_tag_rows)

    comparison_columns = ["version", "evalset", "gold_source", "evalset_id", "n_sentences", "n_tokens", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "weighted_f1", "oov_baseline_vocab_accuracy", "oov_baseline_vocab_total", "low_conf_rate", "mean_confidence", "parent_version", "label_policy", "experimental", "n_feedback_train_sentences"]
    comparison = pd.DataFrame(version_rows, columns=comparison_columns)
    comparison.to_csv(paths.compare_out / "version_comparison.csv", index=False)
    title = "RinTag HITL version comparison"
    table = "\n".join(line.rstrip() for line in comparison.to_string(index=False).splitlines())
    text = title + "\n" + "=" * len(title) + "\n\n" + table + "\n\n" + LEGEND + "\n"
    (paths.compare_out / "version_comparison.txt").write_text(text, encoding="utf-8")

    if len(selected) >= 2:
        for name, (evalset, idx) in eval_context.items():
            pairs = []
            pair_seen = set()
            for i, to_version in enumerate(selected[1:], start=1):
                candidates = [selected[i - 1], selected[0]]
                for from_version in candidates:
                    pair = (from_version, to_version)
                    if pair in pair_seen:
                        continue
                    pair_seen.add(pair)
                    pred_a = prediction_by_pair[(from_version, name)]
                    pred_b = prediction_by_pair[(to_version, name)]
                    flat_a = sum(pred_a, [])
                    flat_b = sum(pred_b, [])
                    flat_gold = sum(evalset.gold, [])
                    n_fixed = sum(a != gold and b == gold for a, b, gold in zip(flat_a, flat_b, flat_gold))
                    n_regressed = sum(a == gold and b != gold for a, b, gold in zip(flat_a, flat_b, flat_gold))
                    counts_a = counts_by_pair[(from_version, name)]
                    counts_b = counts_by_pair[(to_version, name)]
                    acc_a, f1_a = distributions[(from_version, name)]
                    acc_b, f1_b = distributions[(to_version, name)]
                    delta_acc = ci95(acc_b - acc_a)
                    delta_f1 = ci95(f1_b - f1_a)
                    acc_point_a = metrics_by_pair[(from_version, name)]["accuracy"]
                    acc_point_b = metrics_by_pair[(to_version, name)]["accuracy"]
                    f1_point_a = metrics_by_pair[(from_version, name)]["macro_f1"]
                    f1_point_b = metrics_by_pair[(to_version, name)]["macro_f1"]
                    pairs.append({
                        "from_version": from_version, "to_version": to_version,
                        "n_tokens": evalset.n_tokens, "n_fixed": n_fixed,
                        "n_regressed": n_regressed, "net_fixed": n_fixed - n_regressed,
                        "mcnemar_p": mcnemar_exact_p(n_regressed, n_fixed),
                        "delta_accuracy": acc_point_b - acc_point_a,
                        "delta_accuracy_ci_low": delta_acc[0], "delta_accuracy_ci_high": delta_acc[1],
                        "delta_macro_f1": f1_point_b - f1_point_a,
                        "delta_macro_f1_ci_low": delta_f1[0], "delta_macro_f1_ci_high": delta_f1[1],
                    })
                    change_rows = []
                    for uid, tokens, golds, preds_a, preds_b in zip(evalset.uids, evalset.tokens, evalset.gold, pred_a, pred_b):
                        for token_index, (token, gold, a, b) in enumerate(zip(tokens, golds, preds_a, preds_b)):
                            if a == b:
                                continue
                            kind = "fixed" if a != gold and b == gold else "regressed" if a == gold and b != gold else "changed_wrong"
                            change_rows.append({"uid": uid, "token_index": token_index, "token": token, "gold": gold, "pred_from": a, "pred_to": b, "kind": kind})
                    write_csv(paths.compare_out / f"changes_{name}_v{from_version}_to_v{to_version}.csv", ["uid", "token_index", "token", "gold", "pred_from", "pred_to", "kind"], change_rows)
            _write_frame(paths.compare_out / f"pairwise_{name}.csv", pairs)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(8, 5))
        for name, (evalset, _) in eval_context.items():
            y = [metrics_by_pair[(version, name)]["macro_f1"] for version in selected]
            low = [metrics_by_pair[(version, name)]["macro_f1"] - ci95(distributions[(version, name)][1])[0] for version in selected]
            high = [ci95(distributions[(version, name)][1])[1] - metrics_by_pair[(version, name)]["macro_f1"] for version in selected]
            axis.errorbar(selected, y, yerr=[low, high], marker="o", capsize=3, label=name)
            offset = 8
            for version, value in zip(selected, y):
                axis.annotate(f"{value:.4f}", (version, value), textcoords="offset points", xytext=(0, offset), ha="center", va="bottom" if offset > 0 else "top")
        axis.set_xticks(selected, [f"v{version}{'*' if get_entry(registry, version).get('experimental') else ''}" for version in selected])
        axis.set_xlabel("Model version")
        axis.set_ylabel("Macro-F1")
        axis.legend()
        fig.tight_layout()
        fig.savefig(paths.compare_out / "macro_f1_by_version.png", dpi=150)
        plt.close(fig)
    except ImportError as exc:
        print(f"WARNING: matplotlib unavailable; skipping plots ({exc})")

    audit_status = {}
    for version in selected:
        snapshot_file = paths.snapshot(version) / "feedback_sentences.csv"
        if paths.snapshot(version).exists():
            audit_status[str(version)] = leakage_audit(paths, snapshot_file)["ok"]
    meta = {
        "created_utc": utc_now_iso(), "versions": selected,
        "evalsets": {name: {"id": evalset.id, "gold_source": GOLD_SOURCE[name],
                            "n_sentences": evalset.n_sentences, "n_tokens": evalset.n_tokens}
                     for name, (evalset, _) in eval_context.items()},
        "bootstrap": {"resamples": resamples, "seed": 42},
        "leakage_audit_ok": audit_status,
    }
    write_json_atomic(paths.compare_out / "compare_meta.json", meta)
    return meta