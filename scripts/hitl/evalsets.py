"""Frozen corpus evaluation sets."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from hitl.common import HitlError


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
    raise HitlError(f"unknown eval set: {name}")


def load_corpus_train(paths):
    return load_sentences(paths.processed / "train.csv")