"""
format_for_pos_annotation.py
─────────────────────────────────────────────────────────────────────────────
Formats rbl_sampled_sentences.csv into a word-level CSV ready for POS tagging
in Google Sheets.

Output columns
  sentence_id   – maps back to rbl_sampled_sentences.csv
  full_sentence – the complete sentence (repeated per word row, for context)
  word          – individual token (word or punctuation mark)
  pos_tag       – PUNCT (auto-tagged) for punctuation; blank for all others

Tokenisation rules
  · Sentences are split on whitespace first.
  · Leading/trailing punctuation is detached from each token and emitted as
    its own PUNCT row (e.g. "sadi." → "sadi" [blank] + "." [PUNCT]).
  · Internal punctuation sequences between two alphanumeric characters are
    also split out as PUNCT rows (e.g. "pa.d" → "pa", ".", "d").
  · Exception: purely numeric tokens with internal commas/periods are kept
    intact as a single token (e.g. "411,418" → one NUM token, no split).
  · Internal hyphens and apostrophes are never split
    (e.g. "nag-ukrong", "pa'la" stay as single word tokens).
─────────────────────────────────────────────────────────────────────────────
"""

import re
import pandas as pd

CORPUS_PATH = "../data/interim/sampled_sentences.csv"
OUTPUT_PATH = "../data/interim/formatted_sampled_corpus.csv"

# ── TOKENISER ─────────────────────────────────────────────────────────────────
_LEAD = re.compile(r"^([^\w'''\`\-]+)",  re.UNICODE)
_TAIL = re.compile(r"([^\w'''\`\-]+)$",  re.UNICODE)
_INT  = re.compile(r"([.,;:!?]+)",       re.UNICODE)

def tokenize(sentence: str) -> list[tuple[str, str]]:
    """
    Return ordered (token, pos_tag) pairs for a sentence.
    Punctuation tokens receive pos_tag='PUNCT'; word tokens receive pos_tag=''.
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

        # 2. Separate trailing punctuation (emitted after middle is processed)
        tail_m = _TAIL.search(tok)
        if tail_m:
            middle = tok[:tail_m.start()]
            tail   = tail_m.group(1)
        else:
            middle = tok
            tail   = ""

        # 3. Internal punctuation in middle: split and emit
        if middle:
            parts      = _INT.split(middle)   # alternates: word, punct, word, …
            words_only = [p for i, p in enumerate(parts) if i % 2 == 0 and p]
            puncts_seg = [p for i, p in enumerate(parts) if i % 2 == 1 and p]

            # Keep thousand-separator numbers intact (e.g. 411,418 → one token)
            if puncts_seg and all(re.fullmatch(r"\d+", w) for w in words_only if w):
                result.append((middle, ""))
            else:
                for i, seg in enumerate(parts):
                    if not seg:
                        continue
                    if i % 2 == 0:   # word segment
                        if re.search(r"[a-zA-Z0-9]", seg):
                            result.append((seg, ""))
                    else:            # punctuation segment
                        result.append((seg, "PUNCT"))

        # 4. Trailing punctuation → PUNCT
        if tail:
            result.append((tail, "PUNCT"))

    return result

# ── MAIN ──────────────────────────────────────────────────────────────────────
corpus = pd.read_csv(CORPUS_PATH)

rows = []
for _, record in corpus.iterrows():
    sid  = record["sentence_id"]
    text = str(record["text"]).strip()
    for word, pos_tag in tokenize(text):
        rows.append({
            "sentence_id"   : sid,
            "full_sentence" : text,
            "word"          : word,
            "pos_tag"       : pos_tag,
        })

result = pd.DataFrame(rows, columns=["sentence_id", "full_sentence", "word", "pos_tag"])
result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
punct_rows = result[result["pos_tag"] == "PUNCT"]
word_rows  = result[result["pos_tag"] == ""]
print(f"Sentences  : {result['sentence_id'].nunique():,}")
print(f"Total rows : {len(result):,}")
print(f"  Word rows  (blank tag) : {len(word_rows):,}")
print(f"  PUNCT rows (auto-tag)  : {len(punct_rows):,}")
print(f"Output     : {OUTPUT_PATH}")