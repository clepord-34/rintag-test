"""Corpus overlap checks and independent feedback leakage audit."""

from hitl.common import (
    HitlError, NEAR_DUP_JACCARD, Paths, jaccard, read_csv_rows,
    sentence_key, token_set,
)


class SplitIndex:
    def __init__(self, paths: Paths):
        self.by_split = {}
        self.token_sets = {}
        for split in ("train", "dev", "test"):
            path = paths.processed / f"{split}.csv"
            if not path.exists():
                raise HitlError(f"corpus split not found: {path}")
            _, rows = read_csv_rows(path)
            sentences = {}
            for row in rows:
                sid = row["sentence_id"]
                if sid not in sentences:
                    sentences[sid] = row["full_sentence"]
            ordered = sorted(sentences.items(), key=lambda item: int(item[0]))
            self.by_split[split] = {sentence_key(text): sid for sid, text in ordered}
            self.token_sets[split] = [(sid, token_set(sentence_key(text))) for sid, text in ordered]

        annotated_path = paths.processed / "annotated_corpus.csv"
        self.annotated = set()
        if annotated_path.exists():
            _, rows = read_csv_rows(annotated_path)
            self.annotated = {sentence_key(row["full_sentence"]) for row in rows}

    def exact_match(self, key):
        for split in ("test", "dev", "train"):
            if key in self.by_split[split]:
                return f"{split}:{self.by_split[split][key]}"
        if key in self.annotated:
            return "corpus"
        return None

    def near_match(self, key):
        if not key:
            return None
        candidate = token_set(key)
        best = None
        for split in ("test", "dev"):
            for sid, words in self.token_sets[split]:
                score = jaccard(candidate, words)
                if score >= NEAR_DUP_JACCARD and (best is None or score > best[1]):
                    best = (f"{split}:{sid}", score)
        return best


def leakage_audit(paths, sentences_csv=None):
    csv_path = sentences_csv or paths.feedback_validated / "feedback_sentences.csv"
    if not csv_path.exists():
        raise HitlError(f"feedback sentences file not found: {csv_path}")
    _, rows = read_csv_rows(csv_path)
    checked = [(row["uid"], row["full_sentence"], row.get("split", "")) for row in rows if row.get("status") == "ok"]
    index = SplitIndex(paths)
    exact = {name: [] for name in ("test", "dev", "train", "corpus")}
    near = {name: [] for name in ("test", "dev")}
    internal = []
    keyed = []
    for uid, text, split in checked:
        key = sentence_key(text)
        match = index.exact_match(key)
        if match:
            name, _, sid = match.partition(":")
            exact[name].append(uid)
        match = index.near_match(key)
        if match:
            corpus_ref, score = match
            name, _, sid = corpus_ref.partition(":")
            near[name].append({"uid": uid, "corpus_sentence_id": sid, "jaccard": score})
        keyed.append((uid, key, split))
    for i, (uid_a, key_a, split_a) in enumerate(keyed):
        for uid_b, key_b, split_b in keyed[i + 1:]:
            score = jaccard(token_set(key_a), token_set(key_b))
            if key_a == key_b or score >= NEAR_DUP_JACCARD:
                internal.append({"a": uid_a, "b": uid_b, "jaccard": score, "same_split": split_a == split_b})
    ok = not any(exact.values()) and not any(near.values()) and not internal
    return {
        "checked_sentences": len(checked), "exact_overlap": exact,
        "near_overlap": near, "internal_duplicates": internal, "ok": ok,
    }


def format_audit(audit):
    problems = []
    for split in ("test", "dev", "train", "corpus"):
        for uid in audit["exact_overlap"][split]:
            problems.append(f"  exact overlap with {split}: {uid}")
    for split in ("test", "dev"):
        for item in audit["near_overlap"][split]:
            problems.append(f"  near-duplicate of {split} {item['corpus_sentence_id']} (J={item['jaccard']:.2f}): {item['uid']}")
    for item in audit["internal_duplicates"]:
        problems.append(f"  internal duplicate: {item['a']} ~ {item['b']} (J={item['jaccard']:.2f}, same_split={item['same_split']})")
    if audit["ok"]:
        return f"LEAKAGE AUDIT: OK (checked {audit['checked_sentences']} feedback sentences against train/dev/test)"
    return f"LEAKAGE AUDIT: FAILED ({len(problems)} problem(s))\n" + "\n".join(problems)