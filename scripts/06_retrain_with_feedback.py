"""Retrain the RinTag CRF model using validated HITL corrections.

Supports model versioning, full sentence context preservation, and
comprehensive thesis-ready evaluation outputs.
All versioned model checkpoints are saved in 'models/' outside 'src/',
keeping 'src/rintag/models/' lean with only the active model.
"""

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score as sklearn_f1_score,
)
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rintag.features import sent2features, sent2labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_MODELS_DIR = PROJECT_ROOT / "src" / "rintag" / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

ACTIVE_MODEL = SRC_MODELS_DIR / "crf_model.pkl"
ACTIVE_VOCAB = SRC_MODELS_DIR / "vocab.pkl"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

ALGORITHM = "lbfgs"
MAX_ITERATIONS = 200
BEST_PARAMS = {"c1": 0.1, "c2": 0.1}


# ── Registry helpers ──

def _load_registry() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    registry = {
        "active_version": 1,
        "versions": [{
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dev_macro_f1": None,
            "source": "initial_training",
            "corrections_applied": 0,
            "notes": "Original baseline model from 05_train_model.py",
        }],
    }
    v1_model = MODELS_DIR / "crf_model_v1.pkl"
    v1_vocab = MODELS_DIR / "vocab_v1.pkl"
    if not v1_model.exists() and ACTIVE_MODEL.exists():
        shutil.copy2(ACTIVE_MODEL, v1_model)
    if not v1_vocab.exists() and ACTIVE_VOCAB.exists():
        shutil.copy2(ACTIVE_VOCAB, v1_vocab)
    _save_registry(registry)
    return registry


def _save_registry(registry: dict):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def _next_version(registry: dict) -> int:
    return max(v["version"] for v in registry["versions"]) + 1


def _activate_version(registry: dict, version: int):
    src_model = MODELS_DIR / f"crf_model_v{version}.pkl"
    src_vocab = MODELS_DIR / f"vocab_v{version}.pkl"
    if not src_model.exists():
        print(f"ERROR: {src_model} does not exist.")
        sys.exit(1)
    SRC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_model, ACTIVE_MODEL)
    if src_vocab.exists():
        shutil.copy2(src_vocab, ACTIVE_VOCAB)
    registry["active_version"] = version
    _save_registry(registry)


# ── Evaluation & Results Output ──

def evaluate_and_save_results(
    model, version: int, X_test, y_test_flat, y_pred_flat,
    train_vocab: set, test_tokens: list, labels: list,
    old_per_tag: dict | None, old_overall: dict | None,
    feedback_sentences: list, source: str, timing: dict,
):
    """Generate and save comprehensive evaluation results, markdown report, and tables."""
    ver_dir = RESULTS_DIR / f"retrain_v{version}"
    ver_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classification report (CSV — matches thesis Table 10 format)
    report_dict = classification_report(
        y_test_flat, y_pred_flat, labels=labels,
        output_dict=True, zero_division=0,
    )
    rows = []
    for tag in labels:
        m = report_dict.get(tag, {})
        rows.append({
            "POS Tag": tag,
            "Precision": round(m.get("precision", 0), 4),
            "Recall": round(m.get("recall", 0), 4),
            "F1-Score": round(m.get("f1-score", 0), 4),
            "Support": int(m.get("support", 0)),
        })
    pd.DataFrame(rows).to_csv(ver_dir / "classification_report.csv", index=False)
    with open(ver_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test_flat, y_pred_flat, labels=labels, digits=4))

    # 2. Overall performance (CSV — matches thesis Table 11 format)
    acc = accuracy_score(y_test_flat, y_pred_flat)
    macro_f1 = sklearn_f1_score(y_test_flat, y_pred_flat, labels=labels, average="macro", zero_division=0)
    weighted_f1 = sklearn_f1_score(y_test_flat, y_pred_flat, labels=labels, average="weighted", zero_division=0)

    oov_tokens = [t for t in test_tokens if t[0].lower() not in train_vocab]
    oov_correct = sum(1 for t in oov_tokens if t[1] == t[2])
    oov_acc = oov_correct / len(oov_tokens) if oov_tokens else 0.0

    perf_rows = [
        {"Metric": "Accuracy", "Value": round(acc, 4)},
        {"Metric": "Macro F1", "Value": round(macro_f1, 4)},
        {"Metric": "Weighted F1", "Value": round(weighted_f1, 4)},
        {"Metric": "OOV Accuracy", "Value": round(oov_acc, 4)},
        {"Metric": "OOV Tokens", "Value": f"{oov_correct}/{len(oov_tokens)}"},
        {"Metric": "Total Test Tokens", "Value": len(y_test_flat)},
    ]
    pd.DataFrame(perf_rows).to_csv(ver_dir / "overall_performance.csv", index=False)

    # 3. Confusion matrix heatmap (300 DPI)
    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual Tag")
    plt.xlabel("Predicted Tag")
    plt.title(f"Confusion Matrix — Retrained Model (v{version})")
    plt.tight_layout()
    plt.savefig(ver_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Top misclassifications (CSV)
    misclass = {}
    for true, pred in zip(y_test_flat, y_pred_flat):
        if true != pred:
            key = (true, pred)
            misclass[key] = misclass.get(key, 0) + 1
    tag_counts = {}
    for t in y_test_flat:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    mis_rows = []
    for (true, pred), count in sorted(misclass.items(), key=lambda x: -x[1])[:20]:
        rate = count / tag_counts.get(true, 1)
        mis_rows.append({
            "True Tag": true, "Predicted As": pred,
            "Count": count, "Rate": f"{rate:.1%}",
        })
    pd.DataFrame(mis_rows).to_csv(ver_dir / "top_misclassifications.csv", index=False)

    # 5. Per-tag comparison (CSV — old vs new F1 delta)
    new_per_tag = {
        tag: round(report_dict.get(tag, {}).get("f1-score", 0), 4)
        for tag in labels
    }
    comp_rows = []
    if old_per_tag:
        for tag in labels:
            old_f1 = old_per_tag.get(tag, 0.0)
            new_f1 = new_per_tag.get(tag, 0.0)
            delta = new_f1 - old_f1
            if delta > 0.001:
                status = "✅ improved"
            elif delta < -0.001:
                status = "⚠️ regressed"
            else:
                status = "— unchanged"
            comp_rows.append({
                "POS Tag": tag,
                "Old F1": round(old_f1, 4),
                "New F1": round(new_f1, 4),
                "Delta": f"{delta:+.4f}",
                "Status": status,
            })
        pd.DataFrame(comp_rows).to_csv(ver_dir / "per_tag_comparison.csv", index=False)

    # 6. Corrections log (CSV — full sentences applied)
    if feedback_sentences:
        log_rows = []
        for s in feedback_sentences:
            for w, tag in s["tokens"]:
                log_rows.append({"sentence": s["full_sentence"], "token": w, "tag": tag})
        pd.DataFrame(log_rows).to_csv(ver_dir / "corrections_applied.csv", index=False)

    # 7. Summary metadata (JSON)
    summary = {
        "version": version,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentences_applied": len(feedback_sentences),
        "tokens_applied": sum(len(s["tokens"]) for s in feedback_sentences),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "oov_accuracy": round(oov_acc, 4),
        "total_test_tokens": len(y_test_flat),
        "params": BEST_PARAMS,
        "timing": timing,
    }
    with open(ver_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 8. Thesis-Ready Markdown Report (thesis_report.md)
    report_md_lines = [
        f"# RinTag Retraining Evaluation Report — Version {version}\n",
        f"- **Date & Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Feedback Source:** `{source}` ({len(feedback_sentences)} sentences, {sum(len(s['tokens']) for s in feedback_sentences)} tokens applied)",
        f"- **Training Duration:** {timing.get('train_seconds', 0)} seconds\n",
        "## 1. Overall Performance Comparison (Thesis Table 11 Format)\n",
        "| Metric | Previous Model | Retrained (v" + str(version) + ") | Delta |",
        "|---|---|---|---|",
    ]
    if old_overall:
        for m_key, m_label in [("acc", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"), ("oov_acc", "OOV Accuracy")]:
            old_val = old_overall.get(m_key, 0.0)
            new_val = {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "oov_acc": oov_acc}[m_key]
            d = new_val - old_val
            report_md_lines.append(f"| {m_label} | {old_val:.4f} | {new_val:.4f} | {d:+.4f} |")
    else:
        report_md_lines.extend([
            f"| Accuracy | — | {acc:.4f} | — |",
            f"| Macro F1 | — | {macro_f1:.4f} | — |",
            f"| Weighted F1 | — | {weighted_f1:.4f} | — |",
            f"| OOV Accuracy | — | {oov_acc:.4f} | — |",
        ])

    report_md_lines.extend([
        f"| OOV Tokens | — | {oov_correct}/{len(oov_tokens)} | — |",
        f"| Total Test Tokens | — | {len(y_test_flat)} | — |\n",
        "## 2. Per-Tag Detailed Comparison (Thesis Table 10 Format)\n",
        "| POS Tag | Precision | Recall | Retrained F1 | Previous F1 | Delta | Status | Support |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for r in rows:
        tag = r["POS Tag"]
        prec = r["Precision"]
        rec = r["Recall"]
        f1 = r["F1-Score"]
        supp = r["Support"]
        old_f = old_per_tag.get(tag, 0.0) if old_per_tag else 0.0
        d_val = f1 - old_f if old_per_tag else 0.0
        status_sym = "✅ improved" if d_val > 0.001 else "⚠️ regressed" if d_val < -0.001 else "—"
        report_md_lines.append(f"| {tag} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {old_f:.4f} | {d_val:+.4f} | {status_sym} | {supp} |")

    report_md_lines.extend([
        "\n## 3. Top Misclassifications\n",
        "| True Tag | Predicted As | Error Count | Error Rate |",
        "|---|---|---|---|",
    ])
    for mr in mis_rows[:10]:
        report_md_lines.append(f"| {mr['True Tag']} | {mr['Predicted As']} | {mr['Count']} | {mr['Rate']} |")

    report_md_lines.append("\n## 4. Confusion Matrix\n\n![Confusion Matrix](confusion_matrix.png)\n")

    with open(ver_dir / "thesis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_md_lines))

    # 9. Terminal Output — Formatted Comparison Table
    print("\n" + "=" * 70)
    print(f"       RINTAG RETRAINING RESULTS SUMMARY (v{version})")
    print("=" * 70)
    if old_overall:
        print(f"{'Metric':<20} {'Previous':>12} {'Retrained':>12} {'Delta':>12}")
        print("-" * 70)
        for m_key, m_label in [("acc", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"), ("oov_acc", "OOV Accuracy")]:
            old_val = old_overall.get(m_key, 0.0)
            new_val = {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "oov_acc": oov_acc}[m_key]
            d = new_val - old_val
            print(f"{m_label:<20} {old_val:>12.4f} {new_val:>12.4f} {d:>+12.4f}")
        print("-" * 70)
    else:
        print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | OOV Accuracy: {oov_acc:.4f}")

    if comp_rows:
        improved = [c for c in comp_rows if "improved" in c["Status"]]
        regressed = [c for c in comp_rows if "regressed" in c["Status"]]
        if improved:
            print("Top Improved Tags:")
            for c in sorted(improved, key=lambda x: float(x["Delta"]), reverse=True)[:5]:
                print(f"   {c['POS Tag']}: {c['Old F1']:.4f} -> {c['New F1']:.4f} ({c['Delta']})")
        if regressed:
            print("Regressed Tags:")
            for c in sorted(regressed, key=lambda x: float(x["Delta"]))[:5]:
                print(f"   {c['POS Tag']}: {c['Old F1']:.4f} -> {c['New F1']:.4f} ({c['Delta']})")

    print(f"\n📁 Saved Thesis Artifacts to: {ver_dir}/")
    print("   ├── thesis_report.md          (Formatted markdown report with tables)")
    print("   ├── classification_report.csv (Table 10 per-tag metrics)")
    print("   ├── overall_performance.csv   (Table 11 overall metrics)")
    print("   ├── confusion_matrix.png      (Heatmap at 300 DPI)")
    print("   ├── top_misclassifications.csv")
    print("   ├── per_tag_comparison.csv")
    print("   ├── corrections_applied.csv")
    print("   └── summary.json")
    print("=" * 70 + "\n")

    return new_per_tag, {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "oov_acc": oov_acc}


# ── Correction loaders ──

def load_corrections_from_sheets():
    import gspread
    creds_val = os.environ.get("GOOGLE_CREDENTIALS", "").strip()
    sheet_url = os.environ.get("FEEDBACK_SHEET_URL", "").strip()
    if not creds_val or not sheet_url:
        print("ERROR: GOOGLE_CREDENTIALS and FEEDBACK_SHEET_URL env vars (or .env file) required.")
        sys.exit(1)
    if os.path.isfile(creds_val):
        gc = gspread.service_account(filename=creds_val)
    else:
        gc = gspread.service_account_from_dict(json.loads(creds_val))
    sheet = gc.open_by_url(sheet_url).sheet1
    records = sheet.get_all_records()
    
    if not records:
        print("No feedback records found in Google Sheets.")
        return []

    # Group by sentence_id to reconstruct full sentential context
    by_sentence = {}
    for r in records:
        sid = str(r.get("sentence_id", "")).strip() or str(r.get("full_sentence", "")).strip()
        by_sentence.setdefault(sid, []).append(r)

    sentences = []
    total_tokens = 0
    overrides_by_validator = 0
    user_corrections_kept = 0

    for sid, rows in by_sentence.items():
        rows.sort(key=lambda x: int(x.get("token_index", 0)))
        full_text = rows[0].get("full_sentence", "") or rows[0].get("sentence", "")
        tokens = []
        sentence_rejected = False

        for r in rows:
            w = str(r.get("token", "")).strip()
            if not w:
                continue

            v_tag = str(r.get("validator_tag", "")).strip().upper()
            u_tag = str(
                r.get("user-corrected_tag", "")
                or r.get("user_corrected_tag", "")
                or r.get("user_correction", "")
                or r.get("final_tag", "")
            ).strip().upper()
            m_tag = str(
                r.get("model_tag", "")
                or r.get("predicted_tag", "")
                or r.get("tag", "")
            ).strip().upper()

            # If validator explicitly marked row as REJECT, skip the row or sentence
            if v_tag in ("REJECT", "REJECTED", "INVALID", "NO", "SKIP", "IGNORE"):
                sentence_rejected = True
                break

            # Precedence rule:
            # 1. validator_tag (if validator explicitly assigned/changed it)
            # 2. user-corrected_tag (if user modified it and validator left validator_tag blank = agree)
            # 3. model_tag (neither modified; validator agreed with predicted model tag)
            if v_tag:
                tag = v_tag
                overrides_by_validator += 1
            elif u_tag:
                tag = u_tag
                user_corrections_kept += 1
            else:
                tag = m_tag

            if tag:
                tokens.append((w, tag))

        if not sentence_rejected and tokens:
            sentences.append({"full_sentence": full_text, "tokens": tokens})
            total_tokens += len(tokens)

    print(
        f"Loaded {total_tokens} tokens across {len(sentences)} full sentences from Google Sheets "
        f"({user_corrections_kept} user corrections agreed upon, {overrides_by_validator} validator overrides)."
    )
    return sentences


def load_corrections_from_csv(csv_path: str):
    by_sentence = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sentence_id", "")).strip() or str(row.get("full_sentence", "") or row.get("sentence", "")).strip()
            by_sentence.setdefault(sid, []).append(row)

    sentences = []
    total_tokens = 0
    for sid, rows in by_sentence.items():
        rows.sort(key=lambda x: int(x.get("token_index", 0)))
        full_text = rows[0].get("full_sentence", "") or rows[0].get("sentence", "")
        tokens = []
        sentence_rejected = False

        for r in rows:
            w = str(r.get("token", "")).strip()
            if not w:
                continue

            v_tag = str(r.get("validator_tag", "")).strip().upper()
            u_tag = str(
                r.get("user-corrected_tag", "")
                or r.get("user_corrected_tag", "")
                or r.get("user_correction", "")
                or r.get("final_tag", "")
            ).strip().upper()
            m_tag = str(
                r.get("model_tag", "")
                or r.get("predicted_tag", "")
                or r.get("tag", "")
            ).strip().upper()

            if v_tag in ("REJECT", "REJECTED", "INVALID", "NO", "SKIP", "IGNORE"):
                sentence_rejected = True
                break

            if v_tag:
                tag = v_tag
            elif u_tag:
                tag = u_tag
            else:
                tag = m_tag

            if tag:
                tokens.append((w, tag))

        if not sentence_rejected and tokens:
            sentences.append({"full_sentence": full_text, "tokens": tokens})
            total_tokens += len(tokens)

    print(f"Loaded {total_tokens} tokens across {len(sentences)} full sentences from {csv_path}.")
    return sentences


# ── Training data helpers ──

def apply_corrections_to_training_data(feedback_sentences):
    train_path = PROCESSED_DIR / "train.csv"
    df = pd.read_csv(train_path, keep_default_na=False)
    df["word"] = df["word"].astype(str).replace("", "nan")
    
    max_id = int(df["sentence_id"].max())
    new_rows = []
    updated_sentences = 0

    for item in feedback_sentences:
        sent_text = item["full_sentence"]
        tokens = item["tokens"]  # list of (word, tag)
        
        mask = df["full_sentence"] == sent_text
        if mask.any():
            sent_df = df[mask]
            for (w, tag), idx in zip(tokens, sent_df.index):
                df.at[idx, "pos_tag"] = tag
            updated_sentences += 1
        else:
            max_id += 1
            for w, tag in tokens:
                new_rows.append({
                    "sentence_id": max_id,
                    "full_sentence": sent_text,
                    "word": w,
                    "pos_tag": tag,
                })

    if updated_sentences:
        print(f"Updated tags in {updated_sentences} existing sentences in training data.")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} tokens from {len(feedback_sentences) - updated_sentences} new sentences.")

    return df


def load_sentences_from_df(df):
    sentences = []
    for _, group in df.groupby("sentence_id"):
        sentences.append(list(zip(group["word"].astype(str), group["pos_tag"].astype(str))))
    return sentences


# ── Retraining Runner ──

def retrain(df_train, source: str, feedback_sentences: list):
    registry = _load_registry()
    new_ver = _next_version(registry)

    sentences = load_sentences_from_df(df_train)
    X_train = [sent2features([w for w, _ in s]) for s in sentences]
    y_train = [sent2labels(s) for s in sentences]
    labels = sorted(list({tag for tags in y_train for tag in tags}))

    print(f"\nTraining corpus: {len(sentences):,} sentences, {sum(len(s) for s in sentences):,} tokens")

    # Load test set for evaluation
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv", keep_default_na=False)
    test_df["word"] = test_df["word"].astype(str).replace("", "nan")
    test_sents = load_sentences_from_df(test_df)
    X_test = [sent2features([w for w, _ in s]) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    y_test_flat = [tag for tags in y_test for tag in tags]

    # Training vocab for OOV evaluation
    train_vocab = set(df_train["word"].astype(str).str.lower())

    # ── Evaluate PREVIOUS active model ──
    with open(ACTIVE_MODEL, "rb") as f:
        old_model = pickle.load(f)
    y_pred_old = old_model.predict(X_test)
    y_pred_old_flat = [tag for tags in y_pred_old for tag in tags]
    old_f1 = flat_f1_score(y_test, y_pred_old, average="macro", labels=labels)
    old_report = classification_report(
        y_test_flat, y_pred_old_flat, labels=labels, output_dict=True, zero_division=0
    )
    old_per_tag = {tag: round(old_report[tag]["f1-score"], 4) for tag in labels}
    old_acc = accuracy_score(y_test_flat, y_pred_old_flat)
    old_macro = sklearn_f1_score(y_test_flat, y_pred_old_flat, labels=labels, average="macro", zero_division=0)
    old_weighted = sklearn_f1_score(y_test_flat, y_pred_old_flat, labels=labels, average="weighted", zero_division=0)

    active_vocab = set()
    if ACTIVE_VOCAB.exists():
        with open(ACTIVE_VOCAB, "rb") as f:
            active_vocab = pickle.load(f)
    old_oov_tokens = [sent[j][0] for sent in test_sents for j in range(len(sent)) if sent[j][0].lower() not in active_vocab]
    old_oov_correct = sum(
        1 for i, sent in enumerate(test_sents)
        for j in range(len(sent))
        if sent[j][0].lower() not in active_vocab and sent[j][1] == y_pred_old[i][j]
    )
    old_oov_acc = old_oov_correct / len(old_oov_tokens) if old_oov_tokens else 0.0
    old_overall = {"acc": old_acc, "macro_f1": old_macro, "weighted_f1": old_weighted, "oov_acc": old_oov_acc}

    print(f"\nActive model (v{registry['active_version']}) — Test Macro F1: {old_f1:.4f} (Accuracy: {old_acc:.4f})")

    # ── Train NEW model ──
    print(f"\nTraining v{new_ver} with params: {BEST_PARAMS}")
    t0 = time.perf_counter()
    new_model = sklearn_crfsuite.CRF(
        algorithm=ALGORITHM, max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True, **BEST_PARAMS,
    )
    new_model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.1f}s")

    # ── Evaluate NEW model ──
    y_pred_new = new_model.predict(X_test)
    y_pred_new_flat = [tag for tags in y_pred_new for tag in tags]
    new_f1 = flat_f1_score(y_test, y_pred_new, average="macro", labels=labels)
    delta = new_f1 - old_f1
    symbol = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    print(f"New model (v{new_ver}) — Test Macro F1: {new_f1:.4f} (Δ {delta:+.4f}, {symbol})")

    # Test tokens for OOV breakdown
    test_tokens = []
    for i, sent in enumerate(test_sents):
        for j, (word, true_tag) in enumerate(sent):
            test_tokens.append((word, true_tag, y_pred_new[i][j]))

    # ── Save comprehensive results & report ──
    evaluate_and_save_results(
        model=new_model, version=new_ver,
        X_test=X_test, y_test_flat=y_test_flat, y_pred_flat=y_pred_new_flat,
        train_vocab=train_vocab, test_tokens=test_tokens, labels=labels,
        old_per_tag=old_per_tag, old_overall=old_overall,
        feedback_sentences=feedback_sentences,
        source=source,
        timing={"train_seconds": round(train_time, 1)},
    )

    # ── Save versioned model & vocab outside src/ ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ver_model_path = MODELS_DIR / f"crf_model_v{new_ver}.pkl"
    ver_vocab_path = MODELS_DIR / f"vocab_v{new_ver}.pkl"
    with open(ver_model_path, "wb") as f:
        pickle.dump(new_model, f)
    with open(ver_vocab_path, "wb") as f:
        pickle.dump(train_vocab, f)
    print(f"Saved v{new_ver} model to {ver_model_path}")
    print(f"Saved v{new_ver} vocabulary ({len(train_vocab):,} words) to {ver_vocab_path}")

    registry["versions"].append({
        "version": new_ver,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dev_macro_f1": round(new_f1, 4),
        "source": source,
        "corrections_applied": len(feedback_sentences),
        "notes": "",
    })

    if new_f1 >= old_f1:
        _activate_version(registry, new_ver)
        df_train.to_csv(PROCESSED_DIR / "train.csv", index=False)
        print(f"✅ v{new_ver} is now the active model in src/rintag/models/.")
    else:
        _save_registry(registry)
        print(f"⚠️  v{new_ver} regressed. Active model remains v{registry['active_version']}.")
        print(f"    You can switch to it manually anytime via: python scripts/06_retrain_with_feedback.py rollback --version {new_ver}")

    return new_f1, old_f1


# ── CLI Commands ──

def cmd_retrain(args):
    if args.source == "sheets":
        feedback_sentences = load_corrections_from_sheets()
    else:
        if not args.csv_path:
            print("ERROR: --csv-path required when --source csv")
            sys.exit(1)
        feedback_sentences = load_corrections_from_csv(args.csv_path)
    if not feedback_sentences:
        print("No feedback sentences to apply. Exiting.")
        return

    print(f"\nFeedback sentences to integrate ({len(feedback_sentences)} sentences):")
    for s in feedback_sentences[:5]:
        tok_str = " ".join(f"{w}/{t}" for w, t in s["tokens"])
        print(f"  • {s['full_sentence']} → [{tok_str}]")
    if len(feedback_sentences) > 5:
        print(f"  ... and {len(feedback_sentences) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Preview completed. No model was trained or changed.")
        return

    df_updated = apply_corrections_to_training_data(feedback_sentences)
    retrain(df_updated, source=args.source, feedback_sentences=feedback_sentences)


def cmd_list(args):
    registry = _load_registry()
    active = registry["active_version"]
    print(f"\n{'Ver':>4}  {'Macro F1':>9}  {'Sentences':>12}  {'Source':>15}  {'Timestamp':<26}  Active")
    print("─" * 90)
    for v in registry["versions"]:
        f1 = f"{v['dev_macro_f1']:.4f}" if v.get("dev_macro_f1") is not None else "   —    "
        marker = "  ◀ ACTIVE" if v["version"] == active else ""
        print(f"  v{v['version']:<2}  {f1:>9}  {v.get('corrections_applied', 0):>12}  "
              f"{v.get('source', ''):>15}  {v.get('timestamp', '')[:19]:<26}{marker}")
    print()


def cmd_rollback(args):
    registry = _load_registry()
    target = args.version
    known = {v["version"] for v in registry["versions"]}
    if target not in known:
        print(f"ERROR: v{target} not found. Known versions: {sorted(known)}")
        sys.exit(1)
    if target == registry["active_version"]:
        print(f"v{target} is already the active model.")
        return
    _activate_version(registry, target)
    print(f"✅ Rolled back to v{target}. crf_model.pkl and vocab.pkl now point to v{target}.")


def main():
    parser = argparse.ArgumentParser(description="RinTag HITL model management")
    sub = parser.add_subparsers(dest="command", required=True)

    p_retrain = sub.add_parser("retrain", help="Retrain with validated HITL feedback")
    p_retrain.add_argument("--source", choices=["sheets", "csv"], required=True)
    p_retrain.add_argument("--csv-path", type=str, help="Path to feedback CSV")
    p_retrain.add_argument("--dry-run", action="store_true", help="Preview without training")

    sub.add_parser("list", help="List all versioned models")

    p_rollback = sub.add_parser("rollback", help="Switch active model to a previous version")
    p_rollback.add_argument("--version", type=int, required=True, help="Version number to restore")

    args = parser.parse_args()
    cmd_map = {"retrain": cmd_retrain, "list": cmd_list, "rollback": cmd_rollback}
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
