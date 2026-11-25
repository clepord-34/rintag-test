"""Frozen corpus and feedback evaluation sets."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from hitl.common import HitlError, read_csv_rows


def load_sentences(csv_path):
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["word"] = df["word"].astype(str).replace("", "nan")
    sentences = []
    for _, group in df.groupby("sentence_id"):
        sentences.append(list(zip(group["word"].astype(str), group["pos_tag"].astype(str))))
    return sentences


@dataclass
class EvalSet:
    name: str
    id: str
    uids: list[str]
    tokens: list[list[str]]
    gold: list[list[str]]

    @property
    def n_sentences(self):
        return len(self.uids)

    @property
    def n_tokens(self):
        return sum(len(tokens) for tokens in self.tokens)


def compute_evalset_id(uids, tokens, gold):
    payload = "".join(
        uid + "\t" + "\t".join(f"{token}/{tag}" for token, tag in zip(sentence_tokens, sentence_gold)) + "\n"
        for uid, sentence_tokens, sentence_gold in zip(uids, tokens, gold)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evalset(name, uids, tokens, gold):
    return EvalSet(name, compute_evalset_id(uids, tokens, gold), uids, tokens, gold)


def load_evalset(paths, name):
    if name in ("test", "dev"):
        sentences = load_sentences(paths.processed / f"{name}.csv")
        frame = pd.read_csv(paths.processed / f"{name}.csv", keep_default_na=False)
        frame["word"] = frame["word"].astype(str).replace("", "nan")
        uids, tokens, gold = [], [], []
        for sentence_id, group in frame.groupby("sentence_id"):
            uids.append(f"{name}:{sentence_id}")
            tokens.append(group["word"].astype(str).tolist())
            gold.append(group["pos_tag"].astype(str).tolist())
        return _evalset(name, uids, tokens, gold)
    if name not in ("hitl_holdout", "hitl_holdout_user"):
        raise HitlError(f"unknown eval set: {name}")
    sentences_path = paths.feedback_validated / "feedback_sentences.csv"
    tokens_path = paths.feedback_validated / "feedback_tokens.csv"
    if not sentences_path.exists() or not tokens_path.exists():
        return _evalset(name, [], [], [])
    _, sentences = read_csv_rows(sentences_path)
    _, rows = read_csv_rows(tokens_path)
    if name == "hitl_holdout":
        selected = [row["uid"] for row in sentences if row.get("status") == "ok" and row.get("split") == "holdout" and row.get("validated") == "true"]
        gold_column = "validator_tag"
    else:
        selected = [row["uid"] for row in sentences if row.get("status") == "ok" and row.get("split") == "holdout"]
        gold_column = "user_effective_tag"
    selected_set = set(selected)
    order = list(dict.fromkeys(row["uid"] for row in rows if row["uid"] in selected_set))
    tokens, gold = [], []
    for uid in order:
        sentence_rows = sorted((row for row in rows if row["uid"] == uid), key=lambda row: int(row["token_index"]))
        tokens.append([row["token"] for row in sentence_rows])
        gold.append([row[gold_column] for row in sentence_rows])
    return _evalset(name, order, tokens, gold)


def load_corpus_train(paths):
    return load_sentences(paths.processed / "train.csv")