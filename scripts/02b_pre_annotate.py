"""
02b_pre_annotate.py
─────────────────────────────────────────────────────────────────────────────
Lexicon-Based Pre-Annotator for the RBL POS annotation pipeline.

This script implements the "Lexicon-Based Pre-Annotator" instrument described
in the thesis methodology (Chapter 3). It performs the following steps:

  1. Load the UD-converted lexicon (output of 02a_convert_lexicon_ptb_to_ud.py)
  2. Load the tokenized sampled sentences (output of 02_format_for_annotation.py)
  3. For each token, look it up in the lexicon (case-insensitive)
  4. If found → assign the lexicon's UD tag as the pre-annotated tag
  5. If NOT found → assign 'X' (unknown/needs manual review)

The output is a word-level CSV showing the pre-annotator's first-pass labels.
This was used as a starting point for manual annotation — researchers then
corrected and verified every tag to produce the final annotated corpus.

IMPORTANT: This script does NOT modify the final annotated corpus
(data/processed/annotated_corpus.csv). That file is the authoritative,
researcher-corrected dataset.

Input:
  - data/lexicon/lexicon_ud.csv             (UD-converted lexicon)
  - data/interim/formatted_sampled_corpus.csv (tokenized sampled sentences)

Output:
  - data/interim/pre_annotated_corpus.csv    (first-pass pre-annotations)

Reference: Thesis Chapter 3, "Corpus Preparation" and "Instrument" sections
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
LEXICON_PATH   = "../data/lexicon/lexicon_ud.csv"
SENTENCES_PATH = "../data/interim/formatted_sampled_corpus.csv"
OUTPUT_PATH    = "../data/interim/pre_annotated_corpus.csv"

# ── TOKENIZER (same as 02_format_for_annotation.py) ──────────────────────────
# We reuse the exact same tokenization logic so that the pre-annotated output
# aligns token-for-token with the annotation spreadsheet.
_LEAD = re.compile(r"^([^\w'''\`\-]+)",  re.UNICODE)
_TAIL = re.compile(r"([^\w'''\`\-]+)$",  re.UNICODE)
_INT  = re.compile(r"([.,;:!?]+)",       re.UNICODE)


def tokenize(sentence: str) -> list[tuple[str, str]]:
    """Return ordered (token, tag_hint) pairs for a sentence.

    Punctuation tokens receive tag_hint='PUNCT'; word tokens receive tag_hint=''.
    This is the same tokenizer used in 02_format_for_annotation.py.
    """
    result = []
    for tok in sentence.split():
        # 1. Leading punctuation → PUNCT
        lead_m = _LEAD.match(tok)
        if lead_m:
            result.append((lead_m.group(1), "PUNCT"))
            tok = tok[lead_m.end():]
        if not tok:
            continue

        # 2. Separate trailing punctuation
        tail_m = _TAIL.search(tok)
        if tail_m:
            middle = tok[:tail_m.start()]
            tail   = tail_m.group(1)
        else:
            middle = tok
            tail   = ""

        # 3. Internal punctuation in middle
        if middle:
            parts      = _INT.split(middle)
            words_only = [p for i, p in enumerate(parts) if i % 2 == 0 and p]
            puncts_seg = [p for i, p in enumerate(parts) if i % 2 == 1 and p]

            if puncts_seg and all(re.fullmatch(r"\d+", w) for w in words_only if w):
                result.append((middle, ""))
            else:
                for i, seg in enumerate(parts):
                    if not seg:
                        continue
                    if i % 2 == 0:
                        if re.search(r"[a-zA-Z0-9]", seg):
                            result.append((seg, ""))
                    else:
                        result.append((seg, "PUNCT"))

        # 4. Trailing punctuation → PUNCT
        if tail:
            result.append((tail, "PUNCT"))

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Lexicon-Based Pre-Annotator")
    print("=" * 60)

    # ── STEP 1: LOAD UD LEXICON ───────────────────────────────────────────────
    print("\n[1/4] Loading UD-converted lexicon …")
    lexicon_df = pd.read_csv(LEXICON_PATH)
    print(f"  Loaded {len(lexicon_df):,} lexicon entries")

    # Build a case-insensitive lookup dictionary: lowercase_word → ud_tag
    # If a word appears multiple times with different tags, the first
    # occurrence wins (which corresponds to the most common/primary sense).
    lexicon_lookup: dict[str, str] = {}
    duplicates = 0
    for _, row in lexicon_df.iterrows():
        word_lower = str(row["words"]).strip().lower()
        ud_tag = str(row["ud_tag"]).strip()
        if word_lower not in lexicon_lookup:
            lexicon_lookup[word_lower] = ud_tag
        else:
            duplicates += 1

    print(f"  Built lookup with {len(lexicon_lookup):,} unique words")
    if duplicates > 0:
        print(f"  ({duplicates} duplicate entries — first tag kept)")

    # ── STEP 2: LOAD SAMPLED SENTENCES ────────────────────────────────────────
    print("\n[2/4] Loading sampled sentences …")
    sentences_df = pd.read_csv(SENTENCES_PATH)
    print(f"  Loaded {len(sentences_df):,} sentences")

    # ── STEP 3: TOKENIZE & PRE-ANNOTATE ──────────────────────────────────────
    print("\n[3/4] Tokenizing and pre-annotating …")
    rows = []
    total_tokens  = 0
    lexicon_hits  = 0
    lexicon_miss  = 0
    punct_auto    = 0

    for _, record in sentences_df.iterrows():
        sid  = record["sentence_id"]
        text = str(record["text"]).strip()

        for word, tag_hint in tokenize(text):
            total_tokens += 1

            if tag_hint == "PUNCT":
                # Punctuation is auto-tagged (not looked up in lexicon)
                pre_tag = "PUNCT"
                punct_auto += 1
            else:
                # Look up the word in the lexicon (case-insensitive)
                lookup_result = lexicon_lookup.get(word.lower())
                if lookup_result is not None:
                    pre_tag = lookup_result
                    lexicon_hits += 1
                else:
                    pre_tag = "X"  # Not found → needs manual review
                    lexicon_miss += 1

            rows.append({
                "sentence_id":      sid,
                "full_sentence":    text,
                "word":             word,
                "pre_annotated_tag": pre_tag,
            })

    result = pd.DataFrame(rows, columns=[
        "sentence_id", "full_sentence", "word", "pre_annotated_tag"
    ])
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # ── STEP 4: SUMMARY ──────────────────────────────────────────────────────
    print("\n[4/4] Pre-annotation summary")
    print(f"  Total tokens        : {total_tokens:,}")
    print(f"  Punctuation (auto)  : {punct_auto:,}")
    print(f"  Lexicon hits        : {lexicon_hits:,}")
    print(f"  Lexicon misses (X)  : {lexicon_miss:,}")

    word_tokens = lexicon_hits + lexicon_miss
    if word_tokens > 0:
        coverage = lexicon_hits / word_tokens
        print(f"  Lexicon coverage    : {coverage:.1%} of word tokens")
        print(f"    (i.e., {coverage:.1%} of words got an automatic first-pass tag)")
        print(f"    (the remaining {1 - coverage:.1%} were labeled X for manual review)")

    # Pre-annotated tag distribution
    print(f"\n  Pre-annotated tag distribution:")
    tag_counts = result["pre_annotated_tag"].value_counts()
    for tag, count in tag_counts.items():
        print(f"    {tag:<10} {count:>7,} ({count / len(result):.1%})")

    print(f"\n  Output saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
