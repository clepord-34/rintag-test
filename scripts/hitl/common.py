"""Shared constants and utilities for the RinTag HITL workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

ALGORITHM = "lbfgs"
MAX_ITERATIONS = 200
TAGSET = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB"]
NEAR_DUP_JACCARD = 0.8
LOW_CONF_THRESHOLD = 0.75
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
LABEL_POLICIES = ("validator", "validator_then_user", "user")


class HitlError(Exception):
    """An expected error in a HITL command."""


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls):
        return cls(REPO_ROOT)

    @property
    def processed(self):
        return self.root / "data" / "processed"

    @property
    def feedback_validated(self):
        return self.root / "data" / "feedback" / "validated"

    @property
    def feedback_snapshots(self):
        return self.root / "data" / "feedback" / "snapshots"

    @property
    def models(self):
        return self.root / "models"

    @property
    def registry(self):
        return self.models / "model_registry.json"

    @property
    def results(self):
        return self.models / "results"

    @property
    def bundled(self):
        return self.root / "src" / "rintag" / "models"

    @property
    def compare_out(self):
        return self.root / "experiments" / "results" / "hitl"

    def model_file(self, v):
        return self.models / f"crf_model_v{v}.pkl"

    def vocab_file(self, v):
        return self.models / f"vocab_v{v}.pkl"

    def version_results(self, v):
        return self.results / f"v{v}"

    def snapshot(self, v):
        return self.feedback_snapshots / f"v{v}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp_path, path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sentence_key(text: str) -> str:
    s = unicodedata.normalize("NFC", text).lower()
    s = "".join(ch if unicodedata.category(ch)[0] in "LNM" else " " for ch in s)
    return " ".join(s.split())


def token_set(key: str) -> frozenset:
    return frozenset(key.split())


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
