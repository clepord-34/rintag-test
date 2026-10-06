import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

def expert_validation():
    df = pd.read_csv("../data/validation/validator_annotations.csv")
    df["validator_tag_filled"] = df["validator_tag"].fillna(df["pos_tag"])
    
    researcher_tags = df["pos_tag"].astype(str)
    expert_tags = df["validator_tag_filled"].astype(str)
    
    kappa = cohen_kappa_score(researcher_tags, expert_tags)
    
    total_tokens = len(df)
    matches = (researcher_tags == expert_tags).sum()
    
    print(f"Cohen's Kappa (k): {kappa:.4f}")
    
    os.makedirs("results", exist_ok=True)
    
    # Save per-tag agreement table
    records = []
    for tag in sorted(researcher_tags.unique()):
        tag_df = df[df["pos_tag"] == tag]
        tag_matches = (tag_df["pos_tag"] == tag_df["validator_tag_filled"]).sum()
        records.append({
            "POS Tag": tag,
            "Total Annotated": len(tag_df),
            "Expert Agreement": tag_matches,
            "Agreement %": tag_matches / len(tag_df) if len(tag_df) > 0 else 0
        })
        
    agreement_df = pd.DataFrame(records)
    agreement_df.to_csv("results/expert_agreement.csv", index=False)
    print("Saved expert agreement stats to results/expert_agreement.csv")

if __name__ == "__main__":
    expert_validation()
