"""
05_feature_analysis.py
─────────────────────────────────────────────────────────────────────────────
Extracts and analyzes the top CRF feature weights from the trained model.

CRF models store learned lambda_k weights for each feature function. A high
positive weight means the feature is strong evidence FOR the associated tag.
A high negative weight means the feature is strong evidence AGAINST it.

This script extracts the top features for each POS tag, which supports the
Results and Discussion section by showing how morphological and contextual
features contribute to tagging accuracy (e.g., proving that the suffix "-on" strongly predicts
VERB, or the prefix "mag-" strongly predicts VERB).

Input:  src/rintag/models/crf_model.pkl    (trained CRF model)

Output:
  - experiments/results/top_features_per_tag.csv
  - experiments/results/feature_weights_summary.txt

Reference: Thesis Chapter 3, "Feature Engineering for RBL"
─────────────────────────────────────────────────────────────────────────────
"""

import pickle
import os
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "../src/rintag/models/crf_model.pkl"
OUTPUT_DIR  = "results"
CSV_PATH    = os.path.join(OUTPUT_DIR, "top_features_per_tag.csv")
TXT_PATH    = os.path.join(OUTPUT_DIR, "feature_weights_summary.txt")
TOP_N       = 20  # Number of top positive features to show per tag
BOTTOM_N    = 10  # Number of top negative features to show per tag


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("CRF Feature Weight Analysis")
    print("=" * 60)

    # ── STEP 1: LOAD MODEL ────────────────────────────────────────────────────
    print("\n[1/3] Loading trained CRF model …")
    with open(MODEL_PATH, "rb") as f:
        crf = pickle.load(f)

    # ── STEP 2: EXTRACT STATE FEATURES ────────────────────────────────────────
    # sklearn-crfsuite exposes state_features_ as a dict:
    #   {(attribute_name, label): weight}
    # and transition_features_ as a dict:
    #   {(label_from, label_to): weight}
    print("\n[2/3] Extracting feature weights …")

    state_features = crf.state_features_
    transition_features = crf.transition_features_

    print(f"  Total state features     : {len(state_features):,}")
    print(f"  Total transition features: {len(transition_features):,}")

    # Organize by label (POS tag)
    features_by_tag: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (attr, label), weight in state_features.items():
        features_by_tag[label].append((attr, weight))

    # Sort each tag's features by absolute weight (descending)
    for tag in features_by_tag:
        features_by_tag[tag].sort(key=lambda x: abs(x[1]), reverse=True)

    # ── STEP 3: GENERATE OUTPUTS ─────────────────────────────────────────────
    print("\n[3/3] Generating output files …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- CSV output ---
    csv_rows = []
    for tag in sorted(features_by_tag.keys()):
        feats = features_by_tag[tag]
        # Top positive features
        positive = [(attr, w) for attr, w in feats if w > 0]
        positive.sort(key=lambda x: x[1], reverse=True)
        for rank, (attr, w) in enumerate(positive[:TOP_N], 1):
            csv_rows.append({
                "pos_tag":      tag,
                "direction":    "positive",
                "rank":         rank,
                "feature_name": attr,
                "weight":       round(w, 4),
            })

        # Top negative features
        negative = [(attr, w) for attr, w in feats if w < 0]
        negative.sort(key=lambda x: x[1])  # most negative first
        for rank, (attr, w) in enumerate(negative[:BOTTOM_N], 1):
            csv_rows.append({
                "pos_tag":      tag,
                "direction":    "negative",
                "rank":         rank,
                "feature_name": attr,
                "weight":       round(w, 4),
            })

    import pandas as pd
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"  Saved feature table to: {CSV_PATH}")

    # --- Text summary output ---
    lines = []
    lines.append("=" * 70)
    lines.append("CRF FEATURE WEIGHT ANALYSIS — SUMMARY FOR THESIS RESULTS")
    lines.append("=" * 70)
    lines.append("")

    for tag in sorted(features_by_tag.keys()):
        feats = features_by_tag[tag]
        total_feats = len(feats)
        positive = [(a, w) for a, w in feats if w > 0]
        negative = [(a, w) for a, w in feats if w < 0]
        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])

        lines.append(f"-- {tag} ({total_feats} features) --")
        lines.append(f"  Top {min(TOP_N, len(positive))} POSITIVE features "
                      f"(evidence FOR {tag}):")
        for rank, (attr, w) in enumerate(positive[:TOP_N], 1):
            lines.append(f"    {rank:>3}. {attr:<40} {w:>+8.4f}")
        lines.append(f"  Top {min(BOTTOM_N, len(negative))} NEGATIVE features "
                      f"(evidence AGAINST {tag}):")
        for rank, (attr, w) in enumerate(negative[:BOTTOM_N], 1):
            lines.append(f"    {rank:>3}. {attr:<40} {w:>+8.4f}")
        lines.append("")

    # Transition features
    lines.append("-- TAG TRANSITIONS --")
    lines.append("  Top 20 transition weights (tag_from → tag_to):")
    sorted_trans = sorted(transition_features.items(),
                          key=lambda x: abs(x[1]), reverse=True)
    for rank, ((from_tag, to_tag), weight) in enumerate(sorted_trans[:20], 1):
        lines.append(f"    {rank:>3}. {from_tag:<8} → {to_tag:<8} {weight:>+8.4f}")

    lines.append("")
    lines.append("=" * 70)

    txt_content = "\n".join(lines)
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"  Saved summary to: {TXT_PATH}")

    # Print preview
    print("\n  Preview (first 30 lines):")
    for line in lines[:30]:
        print(f"  {line}")

    print("=" * 60)


if __name__ == "__main__":
    main()
