"""
sample_rbl_corpus.py
──────────────────────────────────────────────────────────────────────────────
Samples 2000 sentences from the Rinconada Bikol corpus for POS tagging.

Methodology
───────────
1. CLEAN  – Remove exact duplicates and sentences outside the 4–40 word range.
2. STRATIFY – Divide the cleaned pool into four length buckets to ensure
             syntactic variety across the final annotated corpus:
               · Short    (4–7  words)  →  400 sentences
               · Medium   (8–15 words)  →  1000 sentences
               · Long     (16–25 words) →  500 sentences
               · Extended (26–40 words) →  100 sentences
3. GREEDY VOCAB COVERAGE – Within each stratum, sentences are selected by a
             lazy-heap greedy algorithm that maximises the number of unique
             word tokens added to a running "seen" set at each step.
             A fixed random seed (42) is applied before the heap is built so
             ties among equally-novel sentences are broken reproducibly.

The combination of stratification and greedy vocab coverage maximises both
syntactic diversity (via length strata) and lexical diversity (via greedy
coverage) — properties that directly benefit a low-resource POS tagger.

Output: rbl_sampled_sentences.csv  (sentence_id, text, word_count, stratum)
──────────────────────────────────────────────────────────────────────────────
"""

import heapq
import random
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
CORPUS_PATH   = "../data/raw/combined_corpus.csv"
OUTPUT_PATH   = "../data/interim/sampled_sentences.csv"
RANDOM_SEED   = 42

# Sentences to exclude from the sampling pool (e.g., already reserved).
# Populate this list with raw sentence strings before running.
EXCLUSION_SET: set[str] = set()

# Strata: (name, min_words, max_words, target_count)
STRATA = [
    ("short",    4,   7,  400),
    ("medium",   8,  15,  1000),
    ("long",    16,  25,  500),
    ("extended", 26, 40,  100),
]

# ── GREEDY VOCAB COVERAGE ─────────────────────────────────────────────────────
def greedy_vocab_sample(sentences: list[str], n: int, seed: int = 42) -> list[str]:
    """
    Select n sentences from `sentences` that maximise cumulative unique-token
    coverage using a lazy-heap greedy algorithm.

    Algorithm
    ─────────
    · Each sentence is represented as a frozenset of lowercase tokens.
    · A max-heap (negated min-heap) is keyed by the number of tokens the
      sentence would contribute to the current 'seen' set.
    · On each pop we lazily recompute the true marginal gain; if it differs
      from the stored key the entry is re-pushed with the updated gain, so
      the next pop re-evaluates only when needed.  In practice most entries
      are already stale-free, making this nearly O(n log n).
    · A random shuffle with a fixed seed is applied first so that sentences
      with equal marginal gain are broken by a reproducible random order.
    """
    rng = random.Random(seed)
    candidates = list(sentences)
    rng.shuffle(candidates)                         # reproducible tie-breaking

    token_sets = [frozenset(s.lower().split()) for s in candidates]

    # Build initial max-heap: (-initial_coverage, stable_random_tiebreak, idx)
    heap: list[tuple[int, float, int]] = []
    for i, ts in enumerate(token_sets):
        heapq.heappush(heap, (-len(ts), rng.random(), i))

    selected: list[str] = []
    seen: set[str] = set()

    while len(selected) < n and heap:
        neg_cov, tb, idx = heapq.heappop(heap)

        # Lazy recompute: how many tokens does this sentence still add?
        actual_new = len(token_sets[idx] - seen)

        if actual_new == -neg_cov:
            # Marginal gain is still accurate → select this sentence
            selected.append(candidates[idx])
            seen |= token_sets[idx]
        else:
            # Score is stale → re-push with the corrected gain
            heapq.heappush(heap, (-actual_new, tb, idx))

    return selected


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("RBL Corpus Sampler — POS Tagging Sentence Selection")
    print("=" * 60)

    # ── STEP 1: LOAD & CLEAN ──────────────────────────────────────────────────
    print("\n[1/4] Loading and cleaning corpus …")
    df = pd.read_csv(CORPUS_PATH)
    df = df.rename(columns={"Sentiment": "text"})[["text"]].copy()
    df["text"] = df["text"].astype(str).str.strip()

    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_after_dedup = len(df)

    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df = df[(df["word_count"] >= 4) & (df["word_count"] <= 40)].reset_index(drop=True)
    n_after_filter = len(df)

    print(f"  Raw rows            : {n_raw:,}")
    print(f"  After deduplication : {n_after_dedup:,}  (removed {n_raw - n_after_dedup:,} duplicates)")
    print(f"  After length filter : {n_after_filter:,}  (kept 4–40 words)")

    # ── STEP 2: APPLY EXCLUSION SET ───────────────────────────────────────────
    if EXCLUSION_SET:
        before = len(df)
        df = df[~df["text"].isin(EXCLUSION_SET)].reset_index(drop=True)
        print(f"  After exclusions    : {len(df):,}  (removed {before - len(df):,} reserved sentences)")

    # ── STEP 3: STRATIFIED GREEDY SAMPLING ───────────────────────────────────
    print("\n[2/4] Running stratified greedy vocabulary-coverage sampling …")
    print(f"  {'Stratum':<10} {'Range':<12} {'Pool':>7} {'Target':>7} {'Sampled':>8}")
    print("  " + "-" * 50)

    all_sampled: list[tuple[str, str]] = []

    for name, min_w, max_w, target in STRATA:
        pool = df[
            (df["word_count"] >= min_w) & (df["word_count"] <= max_w)
        ]["text"].tolist()

        if len(pool) < target:
            print(f"  WARNING: '{name}' pool ({len(pool)}) < target ({target}). Taking all.")
            target = len(pool)

        sampled = greedy_vocab_sample(pool, target, seed=RANDOM_SEED)
        all_sampled.extend((s, name) for s in sampled)

        range_str = f"{min_w}–{max_w}w"
        print(f"  {name:<10} {range_str:<12} {len(pool):>7,} {target:>7,} {len(sampled):>8,}")

    # ── STEP 4: EXPORT ────────────────────────────────────────────────────────
    print("\n[3/4] Assembling output …")
    result = pd.DataFrame(all_sampled, columns=["text", "stratum"])
    result.insert(0, "sentence_id", range(1, len(result) + 1))
    result["word_count"] = result["text"].apply(lambda x: len(x.split()))

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n[4/4] Summary")
    print(f"  Total sentences sampled : {len(result):,}")
    all_tokens = set(
        tok for sent in result["text"] for tok in sent.lower().split()
    )
    print(f"  Unique tokens covered   : {len(all_tokens):,}")
    print(f"  Output file             : {OUTPUT_PATH}")
    print()
    print("  Stratum breakdown:")
    for name, _, _, _ in STRATA:
        sub = result[result["stratum"] == name]
        toks = set(tok for sent in sub["text"] for tok in sent.lower().split())
        wc = sub["word_count"]
        print(f"    {name:<10}: {len(sub):>4} sentences | "
              f"{len(toks):>5} unique tokens | "
              f"avg {wc.mean():.1f}w (range {wc.min()}–{wc.max()})")
    print("=" * 60)


if __name__ == "__main__":
    main()
