from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def split_corpus():
    df = pd.read_csv(PROCESSED_DIR / "annotated_corpus.csv", keep_default_na=False)
    
    # Group by sentence_id
    sentences = df.groupby("sentence_id").agg({
        "full_sentence": "first",
        "pos_tag": lambda x: pd.Series([t for t in x if t != "PUNCT"]).mode().iloc[0] if [t for t in x if t != "PUNCT"] else "PUNCT"
    }).reset_index()
    
    # Rename for clarity
    sentences.rename(columns={"pos_tag": "dominant_tag"}, inplace=True)
    
    # Handle rare classes for stratification
    tag_counts = sentences["dominant_tag"].value_counts()
    valid_stratify_tags = tag_counts[tag_counts >= 2].index
    top_tag = tag_counts.index[0]
    stratify_labels = sentences["dominant_tag"].apply(lambda x: x if x in valid_stratify_tags else top_tag)
    
    # Split 1: Train+Dev (1800) vs Test (200)
    train_dev, test = train_test_split(
        sentences, test_size=200, stratify=stratify_labels, random_state=42
    )
    
    # Re-calculate valid tags for train_dev split
    tag_counts_td = train_dev["dominant_tag"].value_counts()
    valid_stratify_tags_td = tag_counts_td[tag_counts_td >= 2].index
    top_tag_td = tag_counts_td.index[0]
    stratify_labels_td = train_dev["dominant_tag"].apply(lambda x: x if x in valid_stratify_tags_td else top_tag_td)

    # Split 2: Train (1600) vs Dev (200)
    train, dev = train_test_split(
        train_dev, test_size=200, stratify=stratify_labels_td, random_state=42
    )
    
    # Merge back to token level
    train_df = df[df["sentence_id"].isin(train["sentence_id"])]
    dev_df = df[df["sentence_id"].isin(dev["sentence_id"])]
    test_df = df[df["sentence_id"].isin(test["sentence_id"])]
    
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    dev_df.to_csv(PROCESSED_DIR / "dev.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    
    print("Corpus successfully split:")
    print(f"  Train: {len(train)} sentences, {len(train_df)} tokens")
    print(f"  Dev  : {len(dev)} sentences, {len(dev_df)} tokens")
    print(f"  Test : {len(test)} sentences, {len(test_df)} tokens")

if __name__ == "__main__":
    split_corpus()
