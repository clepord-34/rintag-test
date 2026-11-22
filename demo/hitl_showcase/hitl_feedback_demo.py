"""RinTag HITL Feedback Showcase — Uncertainty-Sampled Export

Demonstrates the complete Human-in-the-Loop (HITL) feedback workflow using
*real unseen sentences* from the raw Rinconada Bikol corpus that were never
included in the annotated training/dev/test sets.

Strategy: Uncertainty Sampling (Active Learning)
  1. Load all unseen sentences from the raw corpus
  2. Batch-tag them with the CRF model (fast: ~7s for ~18 K sentences)
  3. Rank sentences by mean token confidence (ascending — lowest = most uncertain)
  4. Export top-N most uncertain sentences as a HITL feedback CSV

This mirrors real-world active learning: prioritising the sentences the model
is *least* confident about for expert review maximises feedback quality per
annotation effort.

Usage
-----
  python demo/hitl_showcase/hitl_feedback_demo.py                   # defaults
  python demo/hitl_showcase/hitl_feedback_demo.py --top-n 50
  python demo/hitl_showcase/hitl_feedback_demo.py --top-n 25 --min-words 5 --max-words 20
  python demo/hitl_showcase/hitl_feedback_demo.py --no-simulate     # skip simulated corrections
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path bootstrap ──────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent                  # demo/hitl_showcase/
PROJECT_ROOT = HERE.parents[1]                          # RinTag/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

# Lazy imports from rintag internals for batch inference
from rintag.tokenizer import tokenize_sentence
from rintag.features import sent2features
from rintag.tagger import _load_model, _load_vocab
import rintag  # public API for export_feedback schema

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SEPARATOR = "─" * 68

# ── Step helpers ─────────────────────────────────────────────────────────────

def step_header(n: int, title: str):
    print(f"\n{SEPARATOR}")
    print(f"  STEP {n}: {title}")
    print(SEPARATOR)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load unseen sentences
# ══════════════════════════════════════════════════════════════════════════════

def load_unseen_sentences(min_words: int = 3, max_words: int = 30) -> list[str]:
    """Return sentences from the raw corpus not present in the annotated set."""
    step_header(1, "Loading unseen sentences from the raw corpus")

    raw      = pd.read_csv(DATA_DIR / "raw"       / "combined_corpus.csv")
    annotated= pd.read_csv(DATA_DIR / "processed" / "annotated_corpus.csv")
    sampled  = pd.read_csv(DATA_DIR / "interim"   / "sampled_sentences.csv")

    raw_sents     = set(raw["Sentiment"].str.strip())
    ann_sents     = set(annotated["full_sentence"].str.strip())
    sampled_sents = set(sampled["text"].str.strip())

    unseen_all = raw_sents - ann_sents - sampled_sents
    unseen = sorted(
        [s for s in unseen_all if min_words <= len(s.split()) <= max_words]
    )

    print(f"\n  {'Raw corpus sentences:':<38} {len(raw_sents):>6,}")
    print(f"  {'Annotated (training) sentences:':<38} {len(ann_sents):>6,}")
    print(f"  {'Sampled-but-not-annotated:':<38} {len(sampled_sents):>6,}")
    print(f"  {'Unseen after exclusion:':<38} {len(unseen_all):>6,}")
    print(f"  {'Filtered (' + str(min_words) + '–' + str(max_words) + ' words):':<38} {len(unseen):>6,}")
    print(f"\n  [OK] These sentences were never seen during training or annotation.")
    return unseen


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Batch-tag & rank by confidence
# ══════════════════════════════════════════════════════════════════════════════

def batch_tag_and_rank(sentences: list[str]) -> list[dict]:
    """Batch-tag all sentences with the CRF model and rank by mean confidence.

    Returns a list of dicts sorted ascending by mean_confidence (lowest first).
    Each dict has: sentence, tokens, tags, marginals, mean_confidence, min_confidence.
    """
    step_header(2, "Batch-tagging all unseen sentences (uncertainty ranking)")

    model = _load_model()
    vocab = _load_vocab()

    print(f"\n  Tagging {len(sentences):,} sentences — please wait ", end="", flush=True)
    t0 = time.perf_counter()

    # Tokenise and featurise in one pass
    tokenized = []
    valid_sents = []
    for s in sentences:
        toks = tokenize_sentence(s)
        if toks:                     # skip empty tokenization results
            tokenized.append(toks)
            valid_sents.append(s)

    features_all = [sent2features(t) for t in tokenized]

    # Batch predict — single CRF call per batch is much faster than one-by-one
    BATCH = 2000
    all_labels   = []
    all_marginals= []
    for start in range(0, len(features_all), BATCH):
        chunk = features_all[start:start + BATCH]
        all_labels    .extend(list(model.predict(chunk)))
        all_marginals .extend(list(model.predict_marginals(chunk)))
        print(".", end="", flush=True)

    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s, {elapsed / len(valid_sents) * 1000:.1f}ms/sent)")

    # Compute sentence-level confidence metrics
    ranked = []
    for sent, toks, labels, margs in zip(valid_sents, tokenized, all_labels, all_marginals):
        token_confs = [margs[i].get(labels[i], 0.0) for i in range(len(toks))]
        mean_conf = sum(token_confs) / len(token_confs) if token_confs else 0.0
        min_conf  = min(token_confs) if token_confs else 0.0

        ranked.append({
            "sentence":       sent,
            "tokens":         toks,
            "tags":           labels,
            "marginals":      margs,
            "mean_confidence": mean_conf,
            "min_confidence":  min_conf,
            "word_count":      len(toks),
            "oov_count":       sum(1 for t in toks if t.lower() not in vocab),
        })

    # Sort ascending — lowest mean confidence first (most uncertain)
    ranked.sort(key=lambda r: r["mean_confidence"])

    # Quick distribution summary
    low   = sum(1 for r in ranked if r["mean_confidence"] < 0.70)
    mid   = sum(1 for r in ranked if 0.70 <= r["mean_confidence"] < 0.85)
    high  = sum(1 for r in ranked if r["mean_confidence"] >= 0.85)
    print(f"\n  Confidence distribution across {len(ranked):,} sentences:")
    print(f"    < 70%  (critical):  {low:>5,}  sentences")
    print(f"    70–85% (uncertain): {mid:>5,}  sentences")
    print(f"    ≥ 85%  (confident): {high:>5,}  sentences")

    return ranked


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Export ranked list + top-N feedback CSV
# ══════════════════════════════════════════════════════════════════════════════

def export_ranked(ranked: list[dict], top_n: int, threshold: float) -> tuple[Path, Path, list[str]]:
    """Save the full ranked list and the top-N HITL feedback CSV."""
    step_header(3, f"Exporting ranked results & top-{top_n} HITL feedback CSV")

    vocab = _load_vocab()

    # ── 3a. Full ranked summary CSV (all unseen sentences) ──────────────────
    ranked_path = OUTPUT_DIR / "ranked_unseen.csv"
    with open(ranked_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "mean_confidence", "min_confidence",
            "word_count", "oov_count", "sentence",
        ])
        w.writeheader()
        for rank, r in enumerate(ranked, start=1):
            w.writerow({
                "rank":            rank,
                "mean_confidence": round(r["mean_confidence"], 4),
                "min_confidence":  round(r["min_confidence"],  4),
                "word_count":      r["word_count"],
                "oov_count":       r["oov_count"],
                "sentence":        r["sentence"],
            })
    print(f"\n  [OK] Full ranked list -> {ranked_path.relative_to(PROJECT_ROOT)}")

    # ── 3b. Top-N HITL feedback CSV (matches Google Sheets schema) ────────────
    # Headers match demo/app.py FEEDBACK_HEADERS exactly:
    #   sentence_id, timestamp, full_sentence, token_index, token,
    #   model_tag, user-corrected_tag, confidence, is_oov,
    #   validator_tag
    FEEDBACK_HEADERS = [
        "sentence_id", "timestamp", "full_sentence", "token_index",
        "token", "model_tag", "user-corrected_tag", "confidence",
        "is_oov", "validator_tag"
    ]
    feedback_path = OUTPUT_DIR / f"hitl_feedback_top{top_n}.csv"
    rows = []
    top_alts = []
    top = ranked[:top_n]
    now_iso = datetime.now(timezone.utc).isoformat()

    for sid, r in enumerate(top, start=1):
        for idx, (tok, tag, marg) in enumerate(
            zip(r["tokens"], r["tags"], r["marginals"])
        ):
            conf = marg.get(tag, 0.0)
            sorted_alts = sorted(marg.items(), key=lambda x: -x[1])
            top_alt = next(
                ((t, p) for t, p in sorted_alts if t != tag),
                (tag, 0.0)
            )
            top_alts.append(top_alt[0])
            rows.append({
                "sentence_id":        sid,
                "timestamp":          now_iso,
                "full_sentence":      r["sentence"],
                "token_index":        idx,
                "token":              tok,
                "model_tag":          tag,
                "user-corrected_tag": "",
                "confidence":         round(conf, 4),
                "is_oov":             tok.lower() not in vocab,
                "validator_tag":      "",
            })

    with open(feedback_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
        w.writeheader()
        w.writerows(rows)

    # Print preview of top-10 sentences
    print(f"\n  Top-{min(top_n, 10)} most uncertain sentences (lowest mean confidence first):\n")
    print(f"  {'Rank':>4}  {'Conf':>6}  {'OOV':>3}  {'Sentence'}")
    print(f"  {'────':>4}  {'──────':>6}  {'───':>3}  {'─'*50}")
    for rank, r in enumerate(top[:10], start=1):
        oov_flag = f"{r['oov_count']:>3}" if r["oov_count"] else "  0"
        conf_str = f"{r['mean_confidence']:.1%}"
        sent_preview = r["sentence"][:60] + ("…" if len(r["sentence"]) > 60 else "")
        print(f"  {rank:>4}  {conf_str:>6}  {oov_flag:>3}  {sent_preview}")

    low_conf_tokens = sum(1 for row in rows if float(row["confidence"]) < threshold)
    oov_tokens      = sum(1 for row in rows if row["is_oov"])
    print(f"\n  Feedback CSV summary ({top_n} sentences, {len(rows)} tokens total):")
    print(f"    Low-confidence tokens (<{threshold:.0%}): {low_conf_tokens:>4,}")
    print(f"    OOV tokens:                        {oov_tokens:>4,}")
    print(f"    'user-corrected_tag' column:       blank  ← fill in to submit feedback")
    print(f"\n  [OK] Feedback CSV -> {feedback_path.relative_to(PROJECT_ROOT)}")

    return ranked_path, feedback_path, top_alts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Simulate expert corrections (optional)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_corrections(feedback_path: Path, top_alts: list[str], threshold: float) -> Path:
    """Simulate a linguist filling in user-corrected_tag for flagged tokens.

    In reality a human expert opens the CSV, reviews each flagged row,
    and writes the correct POS tag in the user-corrected_tag column.
    Here we auto-fill flagged tokens with their top alternative as a stand-in.
    """
    step_header(4, "Simulating expert corrections (auto-fill flagged tokens)")

    with open(feedback_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    corrected = 0
    for row, top_alt in zip(rows, top_alts):
        conf = float(row.get("confidence", 1.0))
        is_oov = str(row.get("is_oov", "False")).lower() == "true"
        if conf < threshold or is_oov:
            # Stand-in for a human decision: use top alternative if available
            row["user-corrected_tag"] = top_alt or row["model_tag"]
            corrected += 1

    corrected_path = feedback_path.with_name(
        feedback_path.stem + "_corrected" + feedback_path.suffix
    )
    with open(corrected_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\n  {corrected:,} / {total:,} tokens received a simulated correction.")
    print(f"  ({corrected/total:.0%} of tokens flagged for review)")
    print(f"\n  [OK] Corrected CSV -> {corrected_path.relative_to(PROJECT_ROOT)}")

    # Show before/after for a few tokens
    changed = [r for r in rows if r["user-corrected_tag"] and
               r["user-corrected_tag"] != r["model_tag"]][:5]
    if changed:
        print(f"\n  Sample corrections (token: original → corrected):")
        for r in changed:
            print(f"    {r['token']:<20}  "
                  f"{r['model_tag']:<8} → {r['user-corrected_tag']:<8}  "
                  f"(conf: {float(r['confidence']):.1%})")

    return corrected_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Full loop summary
# ══════════════════════════════════════════════════════════════════════════════

def print_loop_summary(feedback_path: Path, corrected_path: Path):
    step_header(5, "HITL Feedback Loop — Summary")
    print(f"""
  How this connects to the full HITL pipeline:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Raw corpus sentences not in training set (unseen)           │
  │       ↓  batch-tag with CRF model                               │
  │  2. Rank by mean token confidence  (uncertainty sampling)       │
  │       ↓  lowest confidence first                                │
  │  3. Export top-N as feedback CSV  (blank user-corrected_tag)    │
  │       ↓  share with linguists / native speakers                 │
  │  4. Expert review: fill user-corrected_tag column               │
  │       ↓  submit via GitHub Issue or send CSV to developers       │
  │  5. Review and curate exported corrections                      │
  │  6. Loop: re-rank unseen pool with current model                │
  └─────────────────────────────────────────────────────────────────┘

  To submit real corrections via GitHub:
    → Open an Issue using .github/ISSUE_TEMPLATE/tag-correction.md
    → Attach your filled {feedback_path.name}
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RinTag HITL Showcase — uncertainty-ranked feedback export"
    )
    parser.add_argument(
        "--top-n", type=int, default=25,
        help="Number of most uncertain sentences to export (default: 25)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Confidence threshold for flagging tokens (default: 0.75)",
    )
    parser.add_argument(
        "--min-words", type=int, default=3,
        help="Min word count for unseen sentences to include (default: 3)",
    )
    parser.add_argument(
        "--max-words", type=int, default=30,
        help="Max word count for unseen sentences to include (default: 30)",
    )
    parser.add_argument(
        "--no-simulate", action="store_true",
        help="Skip the simulated expert corrections step",
    )
    args = parser.parse_args()

    print(f"\nRinTag HITL Feedback Showcase")
    print(f"Uncertainty Sampling — Active Learning for POS Tagging")
    print(f"{SEPARATOR}")
    print(f"  Top-N sentences to export : {args.top_n}")
    print(f"  Confidence threshold      : {args.threshold:.0%}")
    print(f"  Sentence length filter    : {args.min_words}–{args.max_words} words")

    # Step 1 — Load
    unseen = load_unseen_sentences(args.min_words, args.max_words)
    if not unseen:
        print("\n  No unseen sentences found. Check data paths.")
        sys.exit(1)

    # Step 2 — Batch tag + rank
    ranked = batch_tag_and_rank(unseen)

    # Step 3 — Export
    ranked_path, feedback_path, top_alts = export_ranked(ranked, args.top_n, args.threshold)

    # Step 4 — Simulate (optional)
    corrected_path = None
    if not args.no_simulate:
        corrected_path = simulate_corrections(feedback_path, top_alts, args.threshold)
    else:
        corrected_path = feedback_path  # point at uncorrected for summary

    # Step 5 — Loop summary
    print_loop_summary(feedback_path, corrected_path)

    print(f"\nDone! Output files in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/\n")


if __name__ == "__main__":
    main()
