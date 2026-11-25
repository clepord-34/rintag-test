import itertools
import pickle
import random
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from hitl.common import Paths, sentence_key, sha256_file, write_csv

WORDS = {
    "ako": "PRON", "ika": "PRON", "su": "DET", "igin": "NOUN",
    "balay": "NOUN", "dalagan": "VERB", "kaon": "VERB", "na": "PART",
    "sa": "ADP", "dakula": "ADJ",
}


def make_crf(**params):
    import sklearn_crfsuite
    from hitl.common import ALGORITHM, MAX_ITERATIONS

    return sklearn_crfsuite.CRF(
        algorithm=ALGORITHM,
        max_iterations=MAX_ITERATIONS,
        all_possible_transitions=True,
        **params,
    )


def _write_split(path, groups, first_id):
    rows = []
    for offset, words in enumerate(groups):
        sentence_id = first_id + offset
        text = " ".join(words)
        rows.extend({"sentence_id": sentence_id, "full_sentence": text, "word": word, "pos_tag": WORDS[word]} for word in words)
    write_csv(path, ["sentence_id", "full_sentence", "word", "pos_tag"], rows)


def write_feedback_csv(path, sentences, *, validator=True, user=True):
    rows = []
    for sentence_id, text, tags in sentences:
        for token_index, token in enumerate(text.split()):
            tag = tags.get(token, WORDS.get(token, "NOUN"))
            rows.append({
                "sentence_id": sentence_id, "full_sentence": text,
                "token_index": token_index, "token": token,
                "model_tag": tag, "user-corrected_tag": tag if user else "",
                "validator_tag": tag if validator else "",
                "confidence": "0.9", "is_oov": "False",
            })
    write_csv(path, ["sentence_id", "full_sentence", "token_index", "token", "model_tag", "user-corrected_tag", "validator_tag", "confidence", "is_oov"], rows)


@pytest.fixture
def tmp_repo(tmp_path):
    paths = Paths(tmp_path)
    rng = random.Random(31)
    groups = []
    seen = set()
    while len(groups) < 22:
        words = tuple(rng.sample(list(WORDS), 6))
        if frozenset(words) not in seen:
            seen.add(frozenset(words))
            groups.append(words)
    train, dev, test = groups[:14], groups[14:18], groups[18:22]
    _write_split(paths.processed / "train.csv", train, 1)
    _write_split(paths.processed / "dev.csv", dev, 101)
    _write_split(paths.processed / "test.csv", test, 201)
    _write_split(paths.processed / "annotated_corpus.csv", groups, 1)

    from rintag.features import sent2features
    model = make_crf(c1=0.01, c2=0.01)
    model.fit([sent2features(list(words)) for words in train], [[WORDS[w] for w in words] for words in train])
    vocab = {word.lower() for words in train for word in words}
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.bundled.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "src" / "rintag" / "features.py", paths.root / "src" / "rintag" / "features.py")
    with open(paths.model_file(1), "wb") as file:
        pickle.dump(model, file)
    with open(paths.vocab_file(1), "wb") as file:
        pickle.dump(vocab, file)
    (paths.bundled / "crf_model.pkl").write_bytes(paths.model_file(1).read_bytes())
    (paths.bundled / "vocab.pkl").write_bytes(paths.vocab_file(1).read_bytes())
    entry = {
        "version": 1, "timestamp": "2026-01-01T00:00:00Z", "dev_macro_f1": 0.0,
        "test_accuracy": 0.0, "test_macro_f1": 0.0, "test_weighted_f1": 0.0,
        "test_tokens": 24, "oov_accuracy": 0.0, "oov_correct": 0, "oov_total": 0,
        "cv_macro_f1": None, "cv_macro_f1_std": None, "best_c1": 0.01,
        "best_c2": 0.01, "cv_folds": None,
        "artifact_sha256": {"crf_model": sha256_file(paths.model_file(1)), "vocab": sha256_file(paths.vocab_file(1))},
        "source": "initial_training", "notes": "Synthetic baseline",
    }
    from hitl.common import write_json_atomic
    write_json_atomic(paths.registry, {"active_version": 1, "versions": [entry]})

    def feedback_candidates(count=8):
        combos = list(itertools.combinations(WORDS, 5))
        rng.shuffle(combos)
        candidates = []
        for combo in combos:
            text = "mamumuya " + " ".join(combo)
            key = sentence_key(text)
            corpus_keys = [set(words) for words in dev + test]
            if any(len(set(key.split()) & words) / len(set(key.split()) | words) >= 0.8 for words in corpus_keys):
                continue
            candidates.append((str(len(candidates) + 1), text, {**WORDS, "mamumuya": "VERB"}))
            if len(candidates) == count:
                break
        return candidates

    return paths, write_feedback_csv, feedback_candidates