import pandas as pd

def corpus_eda():
    df = pd.read_csv("../data/processed/annotated_corpus.csv")
    
    total_sentences = df["sentence_id"].nunique()
    total_tokens = len(df)
    
    print(f"Total Sentences: {total_sentences:,}")
    print(f"Total Tokens: {total_tokens:,}")
    
    print("\nPOS Tag Distribution:")
    tag_counts = df[df["pos_tag"] != "PUNCT"]["pos_tag"].value_counts()
    for tag, count in tag_counts.items():
        print(f"{tag:>6}: {count:>6,} ({count/tag_counts.sum():.1%})")
        
    sentence_lengths = df.groupby("sentence_id").size()
    print(f"\nSentence Length Stats (including PUNCT):")
    print(f"  Min   : {sentence_lengths.min():.0f}")
    print(f"  Max   : {sentence_lengths.max():.0f}")
    print(f"  Mean  : {sentence_lengths.mean():.1f}")
    print(f"  Median: {sentence_lengths.median():.0f}")
    
    vocab = set(df[df["pos_tag"] != "PUNCT"]["word"].str.lower())
    print(f"\nVocabulary Size: {len(vocab):,} unique lowercased words")

if __name__ == "__main__":
    corpus_eda()
