import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_kappa():
    df = pd.read_csv("../data/validation/validator_annotations.csv")
    
    # Fill blank validator tags with the researcher's tag (blank = agreement)
    df["validator_tag_filled"] = df["validator_tag"].fillna(df["pos_tag"])
    
    researcher_tags = df["pos_tag"].astype(str)
    expert_tags = df["validator_tag_filled"].astype(str)
    
    kappa = cohen_kappa_score(researcher_tags, expert_tags)
    
    total_tokens = len(df)
    blanks = df["validator_tag"].isna().sum()
    matches = (researcher_tags == expert_tags).sum()
    observed_agreement = matches / total_tokens
    
    print(f"Total tokens in validation set: {total_tokens}")
    print(f"Number of blanks (filled as agreement): {blanks}")
    print(f"Observed agreement (Po): {observed_agreement:.4f} ({matches}/{total_tokens})")
    print(f"Cohen's Kappa (k): {kappa:.4f}")
    
    if kappa < 0.00: interpretation = "Poor"
    elif kappa <= 0.20: interpretation = "Slight"
    elif kappa <= 0.40: interpretation = "Fair"
    elif kappa <= 0.60: interpretation = "Moderate"
    elif kappa <= 0.80: interpretation = "Substantial"
    else: interpretation = "Almost Perfect"
    
    print(f"Landis-Koch interpretation: {interpretation} Agreement")
    print("\nPer-tag Agreement Breakdown:")
    
    for tag in sorted(researcher_tags.unique()):
        tag_df = df[df["pos_tag"] == tag]
        tag_matches = (tag_df["pos_tag"] == tag_df["validator_tag_filled"]).sum()
        print(f"{tag:>6}: {tag_matches}/{len(tag_df)} agreed ({tag_matches/len(tag_df):.1%})")

if __name__ == "__main__":
    compute_kappa()
