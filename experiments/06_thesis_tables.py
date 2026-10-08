"""
06_thesis_tables.py
─────────────────────────────────────────────────────────────────────────────
Generates ALL tables and statistics needed for the Results and Discussion
section of the RinTag thesis.

This script consolidates outputs from all previous experiment scripts and
produces thesis-ready formatted tables. Each table is saved as both a CSV
file and a formatted text file suitable for direct copy-paste into the
thesis document.

Tables generated:
  Table 5  — Corpus Composition & Statistics
  Table 6  — Sentence Length Distribution by Stratum
  Table 7  — POS Tag Distribution in the Annotated Corpus
  Table 8  — Annotator-Expert Agreement per POS Tag (Cohen's Kappa)
  Table 9  — Hyperparameter Grid Search Results (5-Fold CV)
  Table 10 — Per-Class Classification Report on Held-Out Test Set
  Table 11 — Overall Model Performance Summary
  Table 12 — Out-of-Vocabulary (OOV) Accuracy
  Table 13 — Pre-Annotator vs. Final Annotation Comparison
  Table 14 — Top Discriminative Features per POS Tag

Output directory: experiments/results/thesis_tables/

Reference: Thesis Chapter 3, "Evaluation Metrics"
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import runpy
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rintag.features import sent2features, sent2labels

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "../data"
MODEL_PATH  = "../src/rintag/models/crf_model.pkl"
OUTPUT_DIR  = "results/thesis_tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPER: SAVE TABLE ───────────────────────────────────────────────────────
def save_table(df: pd.DataFrame, name: str, title: str) -> None:
    """Save a DataFrame as both CSV and formatted text."""
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    txt_path = os.path.join(OUTPUT_DIR, f"{name}.txt")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")

    print(f"  ✓ {title}")
    print(f"    → {csv_path}")


# ── TABLE 5: CORPUS COMPOSITION ──────────────────────────────────────────────
def table5_corpus_composition() -> None:
    """Generate Table 5: Corpus Composition & Statistics."""
    print("\n-- Table 5: Corpus Composition & Statistics --")

    corpus = pd.read_csv(os.path.join(DATA_DIR, "processed/annotated_corpus.csv"), keep_default_na=False)
    train  = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"), keep_default_na=False)
    dev    = pd.read_csv(os.path.join(DATA_DIR, "processed/dev.csv"), keep_default_na=False)
    test   = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)

    rows = []
    for name, df in [("Full Corpus", corpus), ("Train", train),
                      ("Dev", dev), ("Test", test)]:
        n_sents = df["sentence_id"].nunique()
        n_tokens = len(df)
        n_words = len(df[df["pos_tag"] != "PUNCT"])
        n_punct = len(df[df["pos_tag"] == "PUNCT"])
        vocab = df[df["pos_tag"] != "PUNCT"]["word"].str.lower().nunique()
        avg_len = df.groupby("sentence_id").size().mean()

        rows.append({
            "Partition":        name,
            "Sentences":        n_sents,
            "Total Tokens":     n_tokens,
            "Word Tokens":      n_words,
            "Punctuation":      n_punct,
            "Vocabulary Size":  vocab,
            "Avg Tokens/Sent":  round(avg_len, 1),
        })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table5_corpus_composition",
               "Table 5: Corpus Composition and Statistics")


# ── TABLE 6: SENTENCE LENGTH BY STRATUM ──────────────────────────────────────
def table6_length_distribution() -> None:
    """Generate Table 6: Sentence Length Distribution by Stratum."""
    print("\n-- Table 6: Sentence Length Distribution by Stratum --")

    sampled = pd.read_csv(os.path.join(DATA_DIR, "interim/sampled_sentences.csv"))

    rows = []
    for stratum in ["short", "medium", "long", "extended"]:
        sub = sampled[sampled["stratum"] == stratum]
        rows.append({
            "Stratum":      stratum.capitalize(),
            "Word Range":   {"short": "4–7", "medium": "8–15",
                             "long": "16–25", "extended": "26–40"}[stratum],
            "Count":        len(sub),
            "Min Words":    sub["word_count"].min(),
            "Max Words":    sub["word_count"].max(),
            "Mean Words":   round(sub["word_count"].mean(), 1),
            "Median Words": round(sub["word_count"].median(), 1),
        })

    rows.append({
        "Stratum":      "Total",
        "Word Range":   "4–40",
        "Count":        len(sampled),
        "Min Words":    sampled["word_count"].min(),
        "Max Words":    sampled["word_count"].max(),
        "Mean Words":   round(sampled["word_count"].mean(), 1),
        "Median Words": round(sampled["word_count"].median(), 1),
    })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table6_length_distribution",
               "Table 6: Sentence Length Distribution by Stratum")


# ── TABLE 7: POS TAG DISTRIBUTION ────────────────────────────────────────────
def table7_tag_distribution() -> None:
    """Generate Table 7: POS Tag Distribution across train/dev/test."""
    print("\n-- Table 7: POS Tag Distribution --")

    train = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"), keep_default_na=False)
    dev   = pd.read_csv(os.path.join(DATA_DIR, "processed/dev.csv"), keep_default_na=False)
    test  = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)
    full  = pd.read_csv(os.path.join(DATA_DIR, "processed/annotated_corpus.csv"), keep_default_na=False)

    all_tags = sorted(full["pos_tag"].unique())

    rows = []
    for tag in all_tags:
        train_count = (train["pos_tag"] == tag).sum()
        dev_count   = (dev["pos_tag"] == tag).sum()
        test_count  = (test["pos_tag"] == tag).sum()
        total       = (full["pos_tag"] == tag).sum()

        rows.append({
            "POS Tag":      tag,
            "Train":        train_count,
            "Dev":          dev_count,
            "Test":         test_count,
            "Total":        total,
            "Percentage":   f"{total / len(full):.1%}",
        })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table7_tag_distribution",
               "Table 7: POS Tag Distribution in the Annotated Corpus")


# ── TABLE 8: EXPERT AGREEMENT ────────────────────────────────────────────────
def table8_expert_agreement() -> None:
    """Generate Table 8: Annotator-Expert Agreement (Cohen's Kappa)."""
    print("\n-- Table 8: Annotator-Expert Agreement --")

    df = pd.read_csv(os.path.join(DATA_DIR, "validation/validator_annotations.csv"))
    df["validator_tag_filled"] = df["validator_tag"].fillna(df["pos_tag"])

    researcher = df["pos_tag"].astype(str)
    expert     = df["validator_tag_filled"].astype(str)

    kappa = cohen_kappa_score(researcher, expert)
    total = len(df)
    blanks = df["validator_tag"].isna().sum()
    matches = (researcher == expert).sum()
    po = matches / total

    rows = []
    for tag in sorted(researcher.unique()):
        tag_df = df[df["pos_tag"] == tag]
        tag_matches = (tag_df["pos_tag"].astype(str) ==
                       tag_df["validator_tag_filled"].astype(str)).sum()
        rows.append({
            "POS Tag":          tag,
            "Total Annotated":  len(tag_df),
            "Expert Agreed":    tag_matches,
            "Agreement Rate":   f"{tag_matches / len(tag_df):.1%}" if len(tag_df) > 0 else "N/A",
        })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table8_expert_agreement",
               "Table 8: Annotator-Expert Agreement per POS Tag")

    # Save overall Kappa summary
    summary_path = os.path.join(OUTPUT_DIR, "kappa_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Cohen's Kappa Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total tokens in validation set : {total:,}\n")
        f.write(f"Blank validator tags (=agree)  : {blanks:,}\n")
        f.write(f"Observed agreement (Po)        : {po:.4f}\n")
        f.write(f"Cohen's Kappa (κ)              : {kappa:.4f}\n")

        if kappa < 0.00:   interp = "Poor"
        elif kappa <= 0.20: interp = "Slight"
        elif kappa <= 0.40: interp = "Fair"
        elif kappa <= 0.60: interp = "Moderate"
        elif kappa <= 0.80: interp = "Substantial"
        else:               interp = "Almost Perfect"

        f.write(f"Landis-Koch interpretation     : {interp} Agreement\n")
        f.write(f"\nThreshold met (κ ≥ 0.61)?      : "
                f"{'YES ✓' if kappa >= 0.61 else 'NO ✗'}\n")

    print(f"    → {summary_path}")


# ── TABLE 9: GRID SEARCH RESULTS ─────────────────────────────────────────────
def table9_grid_search() -> None:
    """Generate Table 9: Hyperparameter Grid Search Results.

    Reuse the cross-validation experiment's cached per-combination results.
    """
    print("\n-- Table 9: Hyperparameter Grid Search Results --")

    cross_validation = runpy.run_path(
        os.path.join(os.path.dirname(__file__), "02_cross_validation.py")
    )["cross_validation"]
    cross_validation()

    results = pd.read_csv("results/cv_results.csv")
    df_table = results.rename(columns={
        "macro_f1_mean": "Mean F1",
        "macro_f1_std": "Std F1",
    })[["c1", "c2", "Mean F1", "Std F1"]]
    df_table["Mean F1"] = df_table["Mean F1"].round(4)
    df_table["Std F1"] = df_table["Std F1"].round(4)
    df_table["Rank"] = df_table["Mean F1"].rank(
        method="min", ascending=False
    ).astype(int)
    df_table = df_table.sort_values("Rank")
    save_table(df_table, "table9_grid_search",
               "Table 9: Hyperparameter Grid Search Results (5-Fold CV)")

    best = df_table.iloc[0]
    print(f"  Best: c1={best['c1']}, c2={best['c2']}, "
          f"F1={best['Mean F1']:.4f}")


# ── TABLE 10: CLASSIFICATION REPORT ──────────────────────────────────────────
def table10_classification_report() -> None:
    """Generate Table 10: Per-Class Classification Report on Test Set."""
    print("\n-- Table 10: Classification Report on Test Set --")

    test_df  = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"), keep_default_na=False)

    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    sentences = []
    for _, group in test_df.groupby("sentence_id"):
        sentences.append(list(zip(
            group["word"].astype(str), group["pos_tag"].astype(str)
        )))

    X_test = [sent2features([w for w, _ in s]) for s in sentences]
    y_test = [sent2labels(s) for s in sentences]

    y_test_flat = [tag for tags in y_test for tag in tags]
    y_pred = crf.predict(X_test)
    y_pred_flat = [tag for tags in y_pred for tag in tags]

    # Classification report as dict
    report = classification_report(y_test_flat, y_pred_flat, output_dict=True)
    rows = []
    for tag in sorted(set(y_test_flat)):
        if tag in report:
            rows.append({
                "POS Tag":    tag,
                "Precision":  round(report[tag]["precision"], 4),
                "Recall":     round(report[tag]["recall"], 4),
                "F1-Score":   round(report[tag]["f1-score"], 4),
                "Support":    int(report[tag]["support"]),
            })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table10_classification_report",
               "Table 10: Per-Class Classification Report on Held-Out Test Set")

    # Also save the full text report
    report_text = classification_report(y_test_flat, y_pred_flat)
    txt_path = os.path.join(OUTPUT_DIR, "classification_report_full.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)


# ── TABLE 11: OVERALL PERFORMANCE SUMMARY ────────────────────────────────────
def table11_overall_performance() -> None:
    """Generate Table 11: Overall Model Performance Summary."""
    print("\n-- Table 11: Overall Model Performance Summary --")

    test_df  = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"), keep_default_na=False)

    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    sentences = []
    for _, group in test_df.groupby("sentence_id"):
        sentences.append(list(zip(
            group["word"].astype(str), group["pos_tag"].astype(str)
        )))

    X_test = [sent2features([w for w, _ in s]) for s in sentences]
    y_test = [sent2labels(s) for s in sentences]

    y_test_flat = [tag for tags in y_test for tag in tags]
    y_pred = crf.predict(X_test)
    y_pred_flat = [tag for tags in y_pred for tag in tags]

    accuracy  = sum(1 for t, p in zip(y_test_flat, y_pred_flat) if t == p) / len(y_test_flat)
    macro_f1  = f1_score(y_test_flat, y_pred_flat, average="macro")
    weighted_f1 = f1_score(y_test_flat, y_pred_flat, average="weighted")

    # OOV accuracy
    train_vocab = set(train_df["word"].astype(str).str.lower())
    test_tokens = []
    for i, sent in enumerate(sentences):
        for j, (word, true_tag) in enumerate(sent):
            test_tokens.append((word, true_tag, y_pred[i][j]))

    oov_tokens = [t for t in test_tokens if t[0].lower() not in train_vocab]
    oov_correct = sum(1 for t in oov_tokens if t[1] == t[2])
    oov_acc = oov_correct / len(oov_tokens) if oov_tokens else 0.0

    rows = [{
        "Metric":           "Accuracy",
        "Value":            round(accuracy, 4),
        "Threshold":        "—",
        "Met?":             "—",
    }, {
        "Metric":           "Macro F1",
        "Value":            round(macro_f1, 4),
        "Threshold":        "≥ 0.75",
        "Met?":             "YES ✓" if macro_f1 >= 0.75 else "NO ✗",
    }, {
        "Metric":           "Weighted F1",
        "Value":            round(weighted_f1, 4),
        "Threshold":        "—",
        "Met?":             "—",
    }, {
        "Metric":           "OOV Accuracy",
        "Value":            round(oov_acc, 4),
        "Threshold":        "—",
        "Met?":             "—",
    }, {
        "Metric":           "OOV Tokens",
        "Value":            f"{oov_correct}/{len(oov_tokens)}",
        "Threshold":        "—",
        "Met?":             "—",
    }, {
        "Metric":           "Total Test Tokens",
        "Value":            len(y_test_flat),
        "Threshold":        "—",
        "Met?":             "—",
    }]

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table11_overall_performance",
               "Table 11: Overall Model Performance Summary")


# ── TABLE 12: OOV BREAKDOWN ──────────────────────────────────────────────────
def table12_oov_breakdown() -> None:
    """Generate Table 12: OOV Accuracy Breakdown by POS Tag."""
    print("\n-- Table 12: OOV Accuracy Breakdown --")

    test_df  = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train.csv"), keep_default_na=False)

    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    sentences = []
    for _, group in test_df.groupby("sentence_id"):
        sentences.append(list(zip(
            group["word"].astype(str), group["pos_tag"].astype(str)
        )))

    X_test = [sent2features([w for w, _ in s]) for s in sentences]
    y_test = [sent2labels(s) for s in sentences]
    y_pred = crf.predict(X_test)

    train_vocab = set(train_df["word"].astype(str).str.lower())

    # Collect OOV tokens with their tags
    oov_data = []
    for i, sent in enumerate(sentences):
        for j, (word, true_tag) in enumerate(sent):
            if word.lower() not in train_vocab:
                oov_data.append({
                    "word":       word,
                    "true_tag":   true_tag,
                    "pred_tag":   y_pred[i][j],
                    "correct":    true_tag == y_pred[i][j],
                })

    if not oov_data:
        print("  No OOV tokens found.")
        return

    oov_df = pd.DataFrame(oov_data)

    rows = []
    for tag in sorted(oov_df["true_tag"].unique()):
        tag_oov = oov_df[oov_df["true_tag"] == tag]
        correct = tag_oov["correct"].sum()
        total = len(tag_oov)
        rows.append({
            "POS Tag":      tag,
            "OOV Count":    total,
            "Correct":      correct,
            "OOV Accuracy": f"{correct / total:.1%}" if total > 0 else "N/A",
        })

    # Add total row
    total_correct = oov_df["correct"].sum()
    total_count = len(oov_df)
    rows.append({
        "POS Tag":      "TOTAL",
        "OOV Count":    total_count,
        "Correct":      total_correct,
        "OOV Accuracy": f"{total_correct / total_count:.1%}",
    })

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table12_oov_breakdown",
               "Table 12: Out-of-Vocabulary (OOV) Accuracy by POS Tag")


# ── TABLE 13: PRE-ANNOTATOR COMPARISON ────────────────────────────────────────
def table13_pre_annotator_comparison() -> None:
    """Generate Table 13: Pre-Annotator vs. Final Annotation Comparison.

    This compares what the lexicon-based pre-annotator would have assigned
    against the final researcher-corrected tags, to quantify how much
    manual correction was needed.
    """
    print("\n-- Table 13: Pre-Annotator vs. Final Annotation --")

    pre_path = os.path.join(DATA_DIR, "interim/pre_annotated_corpus.csv")
    final_path = os.path.join(DATA_DIR, "processed/annotated_corpus.csv")

    if not os.path.exists(pre_path):
        print("  ⚠ Pre-annotated corpus not found. Run 02b_pre_annotate.py first.")
        print(f"    Expected at: {pre_path}")
        return

    pre   = pd.read_csv(pre_path)
    final = pd.read_csv(final_path)

    # Align by sentence_id and word position
    # Both should have the same number of rows if tokenization is identical
    if len(pre) != len(final):
        print(f"  ⚠ Row count mismatch: pre={len(pre)}, final={len(final)}")
        print("    Comparison may be approximate.")

    # Simple comparison: merge on position
    min_len = min(len(pre), len(final))
    pre_tags   = pre["pre_annotated_tag"].iloc[:min_len].astype(str)
    final_tags = final["pos_tag"].iloc[:min_len].astype(str)

    total = min_len
    exact_match = (pre_tags == final_tags).sum()
    x_count = (pre_tags == "X").sum()
    changed_from_x = ((pre_tags == "X") & (final_tags != "X")).sum()
    wrong_tag = ((pre_tags != final_tags) & (pre_tags != "X") &
                 (pre_tags != "PUNCT")).sum()

    rows = [{
        "Metric":       "Total tokens",
        "Count":        total,
        "Percentage":   "100.0%",
    }, {
        "Metric":       "Pre-annotator matched final tag",
        "Count":        exact_match,
        "Percentage":   f"{exact_match / total:.1%}",
    }, {
        "Metric":       "Labeled X (not in lexicon)",
        "Count":        x_count,
        "Percentage":   f"{x_count / total:.1%}",
    }, {
        "Metric":       "X → corrected to valid tag",
        "Count":        changed_from_x,
        "Percentage":   f"{changed_from_x / total:.1%}",
    }, {
        "Metric":       "Wrong tag (non-X, non-PUNCT)",
        "Count":        wrong_tag,
        "Percentage":   f"{wrong_tag / total:.1%}",
    }]

    df_table = pd.DataFrame(rows)
    save_table(df_table, "table13_pre_annotator",
               "Table 13: Pre-Annotator vs. Final Annotation Comparison")


# ── CONFUSION MATRIX HEATMAP (enhanced) ──────────────────────────────────────
def generate_confusion_matrix_heatmap() -> None:
    """Generate an enhanced confusion matrix heatmap for thesis results."""
    print("\n-- Confusion Matrix Heatmap --")

    test_df = pd.read_csv(os.path.join(DATA_DIR, "processed/test.csv"), keep_default_na=False)

    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    sentences = []
    for _, group in test_df.groupby("sentence_id"):
        sentences.append(list(zip(
            group["word"].astype(str), group["pos_tag"].astype(str)
        )))

    X_test = [sent2features([w for w, _ in s]) for s in sentences]
    y_test = [sent2labels(s) for s in sentences]

    y_test_flat = [tag for tags in y_test for tag in tags]
    y_pred = crf.predict(X_test)
    y_pred_flat = [tag for tags in y_pred for tag in tags]

    labels = sorted(set(y_test_flat) | set(y_pred_flat))
    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)

    # Normalized confusion matrix (percentages)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Generate two heatmaps: raw counts and normalized
    for suffix, data, fmt, title in [
        ("counts", cm, "d",
         "Confusion Matrix (Raw Counts) — Held-Out Test Set"),
        ("normalized", cm_norm, ".2f",
         "Confusion Matrix (Normalized) — Held-Out Test Set"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_ylabel("Actual POS Tag", fontsize=12)
        ax.set_xlabel("Predicted POS Tag", fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{suffix}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  ✓ {path}")

    # Off-diagonal analysis: top misclassification pairs
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "True Tag":     true_label,
                    "Predicted As": pred_label,
                    "Count":        cm[i][j],
                    "Rate":         f"{cm_norm[i][j]:.1%}",
                })

    pairs_df = pd.DataFrame(pairs).sort_values("Count", ascending=False).head(20)
    save_table(pairs_df, "top_misclassifications",
               "Top 20 Misclassification Pairs (Off-Diagonal)")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("THESIS TABLE GENERATOR — RinTag Results")
    print("=" * 70)

    table5_corpus_composition()
    table6_length_distribution()
    table7_tag_distribution()
    table8_expert_agreement()
    table9_grid_search()
    table10_classification_report()
    table11_overall_performance()
    table12_oov_breakdown()
    table13_pre_annotator_comparison()
    generate_confusion_matrix_heatmap()

    print("\n" + "=" * 70)
    print(f"All thesis table outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
