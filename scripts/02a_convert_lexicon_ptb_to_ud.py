"""
02a_convert_lexicon_ptb_to_ud.py
─────────────────────────────────────────────────────────────────────────────
Converts the RBL-Nabua Lexicon from Penn Treebank (PTB) tags to Universal
Dependencies (UD) tags using the mapping defined in Table 3 of the thesis.

The mapping consolidates detailed PTB categories into broader UD labels.
For example, all six PTB verb tags (VB, VBD, VBG, VBN, VBP, VBZ) map to
the single UD label VERB.

Input:  data/lexicon/lexicon.csv        (columns: id, words, tags)
Output: data/lexicon/lexicon_ud.csv     (columns: id, words, ptb_tag, ud_tag)

Reference: Thesis Chapter 3, Table 3 — Penn Treebank (PTB) to Universal
Dependencies (UD) Tag Mapping
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "../data/lexicon/lexicon.csv"
OUTPUT_PATH = "../data/lexicon/lexicon_ud.csv"

# ── PTB → UD MAPPING (Thesis Table 3) ────────────────────────────────────────
#
# This dictionary maps every Penn Treebank tag found in the RBL-Nabua Lexicon
# to its corresponding Universal Dependencies tag. The mapping follows the
# exact rules specified in thesis Table 3:
#
#   - All noun forms (NN, NNS) → NOUN
#   - All proper noun forms (NNP, NNPS) → PROPN
#   - All verb forms (VB, VBD, VBG, VBN, VBP, VBZ) → VERB
#   - All adjective degrees (JJ, JJR, JJS) → ADJ
#   - All adverb degrees (RB, RBR, RBS) → ADV
#   - All pronouns (PRP, PRP$, WP, WP$) → PRON
#   - All determiners (DT, WDT, PDT) → DET
#   - Prepositions/subordinating conjunctions (IN) → ADP
#   - Coordinating conjunctions (CC) → CCONJ
#   - Cardinal numbers (CD) → NUM
#   - Particles (RP) → PART
#   - Interjections (UH) → INTJ
#   - Foreign words, list markers, unclassified (FW, LS, XX) → X
#   - Symbols (SYM) → SYM
#   - Punctuation (all PTB punct tags) → PUNCT
#
PTB_TO_UD = {
    # Nouns
    "NN":   "NOUN",
    "NNS":  "NOUN",
    # Proper nouns
    "NNP":  "PROPN",
    "NNPS": "PROPN",
    # Verbs (PTB has no dedicated AUX category)
    "VB":   "VERB",
    "VBD":  "VERB",
    "VBG":  "VERB",
    "VBN":  "VERB",
    "VBP":  "VERB",
    "VBZ":  "VERB",
    # Adjectives
    "JJ":   "ADJ",
    "JJR":  "ADJ",
    "JJS":  "ADJ",
    # Adverbs
    "RB":   "ADV",
    "RBR":  "ADV",
    "RBS":  "ADV",
    # Pronouns
    "PRP":  "PRON",
    "PRP$": "PRON",
    "WP":   "PRON",
    "WP$":  "PRON",
    # Determiners
    "DT":   "DET",
    "WDT":  "DET",
    "PDT":  "DET",
    # Prepositions / subordinating conjunctions
    "IN":   "ADP",
    # Coordinating conjunctions
    "CC":   "CCONJ",
    # Cardinal numbers
    "CD":   "NUM",
    # Particles
    "RP":   "PART",
    # Interjections
    "UH":   "INTJ",
    # Foreign words, list markers, unclassified
    "FW":   "X",
    "LS":   "X",
    "XX":   "X",
    # Symbols
    "SYM":  "SYM",
    # Punctuation
    ".":    "PUNCT",
    ",":    "PUNCT",
    ":":    "PUNCT",
    "(":    "PUNCT",
    ")":    "PUNCT",
    "``":   "PUNCT",
    "''":   "PUNCT",
    "#":    "PUNCT",
    "$":    "SYM",
}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("PTB → UD Lexicon Converter (Thesis Table 3)")
    print("=" * 60)

    # ── STEP 1: LOAD LEXICON ──────────────────────────────────────────────────
    print("\n[1/3] Loading lexicon …")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {len(df):,} entries from {INPUT_PATH}")

    # Normalize tag column: strip whitespace, uppercase
    df["tags"] = df["tags"].astype(str).str.strip().str.upper()

    # ── STEP 2: MAP PTB → UD ─────────────────────────────────────────────────
    print("\n[2/3] Mapping PTB tags to UD tags …")

    # Track unmapped tags for reporting
    unmapped_tags = set()

    def convert_tag(ptb_tag: str) -> str:
        """Convert a single PTB tag to its UD equivalent.

        If the tag is not found in the mapping dictionary, it is assigned 'X'
        (the UD tag for unclassified tokens) and logged as unmapped.
        """
        ud_tag = PTB_TO_UD.get(ptb_tag)
        if ud_tag is None:
            unmapped_tags.add(ptb_tag)
            return "X"
        return ud_tag

    df["ptb_tag"] = df["tags"]
    df["ud_tag"]  = df["tags"].apply(convert_tag)

    # ── STEP 3: REPORT & SAVE ─────────────────────────────────────────────────
    print("\n  PTB → UD conversion summary:")
    print(f"  {'PTB Tag':<10} {'UD Tag':<10} {'Count':>7}")
    print("  " + "-" * 30)
    for ptb_tag in sorted(PTB_TO_UD.keys()):
        count = (df["ptb_tag"] == ptb_tag).sum()
        if count > 0:
            print(f"  {ptb_tag:<10} {PTB_TO_UD[ptb_tag]:<10} {count:>7,}")

    if unmapped_tags:
        print(f"\n  ⚠ WARNING: {len(unmapped_tags)} unmapped PTB tag(s) assigned to X:")
        for tag in sorted(unmapped_tags):
            count = (df["ptb_tag"] == tag).sum()
            print(f"    {tag}: {count} entries")

    # UD tag distribution
    print("\n  UD tag distribution after conversion:")
    ud_counts = df["ud_tag"].value_counts()
    for tag, count in ud_counts.items():
        print(f"  {tag:<10} {count:>7,} ({count / len(df):.1%})")

    # Save output: keep id, words, ptb_tag, ud_tag
    output_df = df[["id", "words", "ptb_tag", "ud_tag"]].copy()
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[3/3] Saved {len(output_df):,} entries to {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
