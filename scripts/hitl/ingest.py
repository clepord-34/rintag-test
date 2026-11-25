"""Validate, deduplicate, and label feedback CSV inputs."""

import hashlib
from collections import Counter, OrderedDict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from hitl.common import (
    LABEL_POLICIES, NEAR_DUP_JACCARD, TAGSET, HitlError, jaccard,
    read_csv_rows, sentence_key, token_set, utc_now_iso, write_csv,
    write_json_atomic,
)
from hitl.leakage import SplitIndex, format_audit, leakage_audit
from rintag.tokenizer import tokenize_sentence

SENTENCE_FIELDS = ["uid", "source", "source_sentence_id", "key_sha1", "full_sentence", "n_tokens", "status", "reason", "split", "validated", "trainable", "notes"]
TOKEN_FIELDS = ["uid", "token_index", "token", "model_tag", "user_tag", "user_effective_tag", "validator_tag", "final_tag", "split", "source", "confidence", "is_oov"]
REQUIRED_COLUMNS = ("sentence_id", "full_sentence", "token_index", "token", "model_tag", "user-corrected_tag", "validator_tag")


def _normalise(value):
    return (value or "").strip().upper()


def _source_name(path):
    source = Path(path).stem
    return source[:-9] if source.endswith("_feedback") else source


def _resolve_inputs(paths, inputs):
    if inputs is None:
        candidates = [paths.root / "prototype_feedback.csv", paths.root / "library_feedback.csv"]
        resolved = [path for path in candidates if path.exists()]
    else:
        resolved = [Path(path) if Path(path).is_absolute() else paths.root / Path(path) for path in inputs]
    if not resolved:
        raise HitlError("no feedback CSV inputs found")
    sources = [_source_name(path) for path in resolved]
    if len(set(sources)) != len(sources):
        raise HitlError("feedback inputs must have distinct source names")
    for path in resolved:
        if not path.exists():
            raise HitlError(f"feedback input not found: {path}")
    return list(zip(resolved, sources))


def _final_tag(policy, model_tag, user_tag, validator_tag):
    user_effective = user_tag or model_tag
    if policy == "validator":
        return validator_tag
    if policy == "validator_then_user":
        return validator_tag or user_effective
    return user_effective


def _reject_reason(rows, index):
    texts = {row.get("full_sentence", "") for row in rows}
    if len(texts) != 1:
        return "inconsistent_sentence_text"
    parsed = []
    try:
        parsed = [int(row.get("token_index", "")) for row in rows]
    except (TypeError, ValueError):
        return "bad_token_index"
    if sorted(parsed) != list(range(len(rows))):
        return "bad_token_index"
    ordered = sorted(rows, key=lambda row: int(row["token_index"]))
    if any(not row.get("token", "").strip() for row in ordered):
        return "empty_token"
    text = next(iter(texts))
    tokens = [row["token"] for row in ordered]
    if tokens != tokenize_sentence(text):
        return "tokenization_mismatch"
    key = sentence_key(text)
    if not key:
        return "empty_sentence_key"
    if any(not _normalise(row.get("model_tag", "")) for row in ordered):
        return "missing_model_tag"
    for column in ("model_tag", "user-corrected_tag", "validator_tag"):
        bad = sorted({_normalise(row.get(column, "")) for row in ordered if _normalise(row.get(column, "")) and _normalise(row.get(column, "")) not in TAGSET})
        if bad:
            return f"invalid_tag:{column}:{','.join(bad)}"
    exact = index.exact_match(key)
    if exact:
        if exact == "corpus":
            return "in_corpus"
        split, _, sentence_id = exact.partition(":")
        return f"in_{split}:{sentence_id}"
    near = index.near_match(key)
    if near:
        target, score = near
        split, _, sentence_id = target.partition(":")
        return f"near_dup_{split}:{sentence_id}:{score:.2f}"
    return ""


def _sentence_candidate(source, sentence_id, rows, policy, index, first_seen):
    ordered = sorted(rows, key=lambda row: int(row["token_index"])) if all(str(row.get("token_index", "")).isdigit() for row in rows) else list(rows)
    text = rows[0].get("full_sentence", "")
    key = sentence_key(text)
    reason = _reject_reason(rows, index)
    token_data = []
    if not reason:
        for row in ordered:
            model_tag = _normalise(row.get("model_tag", ""))
            user_tag = _normalise(row.get("user-corrected_tag", ""))
            validator_tag = _normalise(row.get("validator_tag", ""))
            effective = user_tag or model_tag
            token_data.append({
                "token": row["token"], "model_tag": model_tag, "user_tag": user_tag,
                "user_effective_tag": effective, "validator_tag": validator_tag,
                "final_tag": _final_tag(policy, model_tag, user_tag, validator_tag),
                "confidence": row.get("confidence", ""), "is_oov": row.get("is_oov", ""),
            })
    validated = bool(token_data) and all(token["validator_tag"] for token in token_data)
    trainable = bool(token_data) and all(token["final_tag"] for token in token_data)
    notes = " | ".join(dict.fromkeys(row.get("validator_note", "").strip() for row in rows if row.get("validator_note", "").strip()))
    uid = f"{source}:{sentence_id}"
    return {
        "uid": uid, "source": source, "source_sentence_id": sentence_id,
        "key_sha1": hashlib.sha1(key.encode("utf-8")).hexdigest(),
        "full_sentence": text, "n_tokens": len(rows),
        "status": "rejected" if reason else "ok", "reason": reason,
        "split": "", "validated": validated, "trainable": trainable,
        "notes": notes, "key": key, "tokens": token_data, "first_seen": first_seen,
    }


def _statistics(candidates):
    valid_tokens = [token for item in candidates if item["status"] == "ok" for token in item["tokens"] if token["validator_tag"]]
    if not valid_tokens:
        return {
            "n_tokens_with_validator_tag": 0, "validator_vs_model_agreement": None,
            "validator_vs_user_agreement": None, "validator_vs_model_kappa": None,
            "validator_vs_user_kappa": None, "top_model_to_validator_changes": None,
        }
    model = [token["model_tag"] for token in valid_tokens]
    user = [token["user_effective_tag"] for token in valid_tokens]
    validator = [token["validator_tag"] for token in valid_tokens]

    def kappa(other):
        if len(set(validator)) <= 1 or len(set(other)) <= 1:
            return None
        return float(cohen_kappa_score(validator, other))

    changes = Counter((token["model_tag"], token["validator_tag"]) for token in valid_tokens if token["model_tag"] != token["validator_tag"])
    top_changes = [{"model_tag": model_tag, "validator_tag": validator_tag, "count": count}
                   for (model_tag, validator_tag), count in sorted(changes.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:20]]
    return {
        "n_tokens_with_validator_tag": len(valid_tokens),
        "validator_vs_model_agreement": sum(a == b for a, b in zip(validator, model)) / len(valid_tokens),
        "validator_vs_user_agreement": sum(a == b for a, b in zip(validator, user)) / len(valid_tokens),
        "validator_vs_model_kappa": kappa(model), "validator_vs_user_kappa": kappa(user),
        "top_model_to_validator_changes": top_changes,
    }


def run_ingest(paths, inputs=None, label_policy="validator"):
    if label_policy not in LABEL_POLICIES:
        raise HitlError(f"unknown label policy: {label_policy}")
    sources = _resolve_inputs(paths, inputs)
    index = SplitIndex(paths)
    grouped = OrderedDict()
    input_metadata = []
    for path, source in sources:
        fieldnames, rows = read_csv_rows(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise HitlError(f"{path} missing required column: {missing[0]}")
        input_metadata.append({"path": str(path), "source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n_rows": len(rows)})
        for row in rows:
            key = (source, row.get("sentence_id", ""))
            grouped.setdefault(key, []).append(row)
    candidates = [_sentence_candidate(source, sid, rows, label_policy, index, i)
                  for i, ((source, sid), rows) in enumerate(grouped.items())]

    by_key = OrderedDict()
    for item in candidates:
        if item["status"] == "ok":
            by_key.setdefault(item["key"], []).append(item)
    for same_key in by_key.values():
        if len(same_key) > 1:
            kept = min(same_key, key=lambda item: (not item["validated"], item["first_seen"]))
            for item in same_key:
                if item is not kept:
                    item["status"] = "rejected"
                    item["reason"] = f"duplicate_of:{kept['uid']}"
                    item["split"] = ""
                    item["validated"] = item["trainable"] = ""
    near_order = sorted((item for item in candidates if item["status"] == "ok"), key=lambda item: (not item["validated"], item["first_seen"]))
    kept = []
    for item in near_order:
        duplicate = next((other for other in kept if jaccard(token_set(item["key"]), token_set(other["key"])) >= NEAR_DUP_JACCARD), None)
        if duplicate:
            item["status"] = "rejected"
            item["reason"] = f"near_duplicate_of:{duplicate['uid']}"
            item["split"] = ""
            item["validated"] = item["trainable"] = ""
        else:
            kept.append(item)
    for item in candidates:
        if item["status"] == "ok":
            item["split"] = "train"

    sentence_rows = []
    token_rows = []
    for item in candidates:
        sentence_row = {field: item[field] for field in SENTENCE_FIELDS}
        if item["status"] == "rejected":
            sentence_row["validated"] = ""
            sentence_row["trainable"] = ""
        else:
            sentence_row["validated"] = str(item["validated"]).lower()
            sentence_row["trainable"] = str(item["trainable"]).lower()
        sentence_rows.append(sentence_row)
        if item["status"] == "ok":
            for token_index, token in enumerate(item["tokens"]):
                token_rows.append({
                    "uid": item["uid"], "token_index": token_index, **{key: token[key] for key in ("token", "model_tag", "user_tag", "user_effective_tag", "validator_tag", "final_tag")},
                    "split": item["split"], "source": item["source"],
                    "confidence": token["confidence"], "is_oov": token["is_oov"],
                })
    paths.feedback_validated.mkdir(parents=True, exist_ok=True)
    write_csv(paths.feedback_validated / "feedback_sentences.csv", SENTENCE_FIELDS, sentence_rows)
    write_csv(paths.feedback_validated / "feedback_tokens.csv", TOKEN_FIELDS, token_rows)

    ok = [item for item in candidates if item["status"] == "ok"]
    trainable = [item for item in ok if item["trainable"]]
    rejected_reasons = Counter(item["reason"].split(":", 1)[0] for item in candidates if item["status"] == "rejected")
    train_items = trainable
    report = {
        "created_utc": utc_now_iso(), "label_policy": label_policy,
        "experimental": label_policy != "validator", "inputs": input_metadata,
        "near_dup_jaccard": NEAR_DUP_JACCARD,
        "n_candidates": len(candidates), "n_ok": len(ok),
        "n_rejected": len(candidates) - len(ok),
        "n_validated": sum(item["validated"] for item in ok),
        "n_trainable": len(trainable), "n_train": len(train_items),
        "n_train_tokens": sum(item["n_tokens"] for item in train_items),
        "rejected_by_reason": dict(rejected_reasons), **_statistics(candidates),
    }
    write_json_atomic(paths.feedback_validated / "ingest_report.json", report)
    audit = leakage_audit(paths)
    if not audit["ok"]:
        raise HitlError(format_audit(audit))
    return report