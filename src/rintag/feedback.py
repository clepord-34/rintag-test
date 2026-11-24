"""HITL feedback utilities for RinTag.

Identifies sentences with uncertain predictions or OOV tokens and exports
them to CSV so community users can review, correct, and submit back to
developers.
"""

import csv
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from rintag.tagger import TagResult

DEFAULT_THRESHOLD = 0.75

FEEDBACK_HEADERS = [
    "sentence_id", "timestamp", "full_sentence", "token_index",
    "token", "model_tag", "user-corrected_tag",
    "confidence", "is_oov"
]


class FeedbackCollector:
    """Passively collects uncertain or OOV sentences during rintag.tag() calls.
    
    Attributes:
        enabled: Whether collection is active.
        output_path: Optional CSV path to automatically append records to.
        threshold: Confidence threshold below which tokens trigger recording.
        log_oov: Whether out-of-vocabulary tokens trigger recording.
        deduplicate: Whether to skip sentences already recorded in this session.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        log_oov: bool = True,
        deduplicate: bool = True,
    ):
        env_val = os.environ.get("RINTAG_PASSIVE_FEEDBACK", "").lower()
        if not enabled and env_val in ("1", "true", "yes", "on"):
            enabled = True

        self.enabled = enabled
        self.output_path = Path(output_path) if output_path else None
        self.threshold = threshold
        self.log_oov = log_oov
        self.deduplicate = deduplicate

        self._buffer: list[dict] = []
        self._seen_sentences: set[str] = set()
        self._sentence_counter: int = 0

    def record(self, sentence: str, result: TagResult) -> bool:
        """Evaluate a TagResult and silently record it if uncertain or OOV.
        
        Returns:
            True if the sentence was recorded, False otherwise.
        """
        if not self.enabled:
            return False

        if not result or len(result) == 0:
            return False

        has_low_conf = any(tok.confidence < self.threshold for tok in result)
        has_oov = self.log_oov and any(tok.is_oov for tok in result)

        if not (has_low_conf or has_oov):
            return False

        sent_clean = sentence.strip()
        if self.deduplicate and sent_clean in self._seen_sentences:
            return False

        self._seen_sentences.add(sent_clean)
        self._sentence_counter += 1
        now_iso = datetime.now(timezone.utc).isoformat()

        rows = []
        for idx, tok in enumerate(result):
            rows.append({
                "sentence_id":        self._sentence_counter,
                "timestamp":          now_iso,
                "full_sentence":      sent_clean,
                "token_index":        idx,
                "token":              tok.token,
                "model_tag":          tok.tag,
                "user-corrected_tag": "",
                "confidence":         round(tok.confidence, 4),
                "is_oov":             tok.is_oov,
            })

        self._buffer.extend(rows)

        if self.output_path:
            self._append_to_file(rows)

        return True

    def _append_to_file(self, rows: list[dict]) -> None:
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def export(self, output_path: str | Path | None = None) -> Path:
        """Export all buffered records to a CSV file."""
        target = Path(output_path) if output_path else (self.output_path or Path("rintag_feedback.csv"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
            writer.writeheader()
            writer.writerows(self._buffer)
        return target.resolve()

    def get_buffer(self) -> list[dict]:
        """Return a copy of the current buffered feedback rows."""
        return list(self._buffer)

    def clear(self) -> None:
        """Clear the in-memory feedback buffer and seen sentences cache."""
        self._buffer.clear()
        self._seen_sentences.clear()
        self._sentence_counter = 0

    @property
    def sentence_count(self) -> int:
        return self._sentence_counter

    @property
    def token_count(self) -> int:
        return len(self._buffer)


# Global collector instance
_default_collector = FeedbackCollector()
_active_collector = _default_collector


def get_feedback_collector() -> FeedbackCollector:
    """Return the currently active feedback collector."""
    return _active_collector


def enable_passive_feedback(
    output_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    log_oov: bool = True,
    deduplicate: bool = True,
) -> FeedbackCollector:
    """Enable passive background feedback recording during rintag.tag() calls.
    
    Args:
        output_path: Optional CSV path to automatically append records to.
        threshold: Confidence cutoff below which tokens are flagged.
        log_oov: Flag out-of-vocabulary words.
        deduplicate: Avoid re-logging identical sentences within the session.
    """
    global _active_collector
    _active_collector.enabled = True
    if output_path is not None:
        _active_collector.output_path = Path(output_path)
    _active_collector.threshold = threshold
    _active_collector.log_oov = log_oov
    _active_collector.deduplicate = deduplicate
    return _active_collector


def disable_passive_feedback() -> None:
    """Disable passive background feedback recording."""
    global _active_collector
    _active_collector.enabled = False


def get_passive_feedback() -> list[dict]:
    """Retrieve all records collected in the passive feedback buffer."""
    return _active_collector.get_buffer()


def export_passive_feedback(output_path: str | Path) -> Path:
    """Export the passive feedback buffer to a CSV file."""
    return _active_collector.export(output_path)


def clear_passive_feedback() -> None:
    """Clear all records in the passive feedback buffer."""
    _active_collector.clear()


@contextmanager
def collect_feedback(
    output_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    log_oov: bool = True,
    deduplicate: bool = True,
):
    """Context manager to passively record uncertain sentences during a code block.
    
    Example:
        with rintag.collect_feedback("audit.csv", threshold=0.75) as collector:
            for line in corpus:
                rintag.tag(line)  # automatically captured if low-confidence or OOV
        print(f"Captured {collector.sentence_count} uncertain sentences.")
    """
    global _active_collector
    prev_collector = _active_collector
    session_collector = FeedbackCollector(
        enabled=True,
        output_path=output_path,
        threshold=threshold,
        log_oov=log_oov,
        deduplicate=deduplicate,
    )
    _active_collector = session_collector
    try:
        yield session_collector
    finally:
        if output_path and not session_collector.output_path:
            session_collector.export(output_path)
        _active_collector = prev_collector


def flag_low_confidence(result: TagResult, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Return info dicts for tokens whose confidence is below the threshold.
    
    Includes token, predicted tag, confidence, and OOV status.
    """
    flagged = []
    for tok in result:
        if tok.confidence < threshold:
            flagged.append({
                "token": tok.token,
                "predicted_tag": tok.tag,
                "confidence": round(tok.confidence, 4),
                "is_oov": tok.is_oov,
            })
    return flagged


def export_feedback(
    results: Sequence[tuple[str, TagResult]],
    output_path: str | Path = "rintag_feedback.csv",
    threshold: float = DEFAULT_THRESHOLD,
    start_sentence_id: int = 1,
) -> Path:
    """Export sentences with low-confidence tokens or OOV tokens to a CSV file.

    Only sentences containing at least one token with confidence < threshold
    OR at least one OOV token are exported. For qualifying sentences, ALL
    tokens are included with their sentence_id for complete sentential context.

    Args:
        results: Sequence of (sentence_str, TagResult) tuples.
        output_path: Path to the output CSV file.
        threshold: Confidence cutoff (default 0.75).
        start_sentence_id: Starting integer for sentence_id numbering.

    Returns:
        Resolved Path to the written CSV file.
    """
    output_path = Path(output_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    rows = []
    current_sid = start_sentence_id
    for sentence, result in results:
        has_low_conf = any(tok.confidence < threshold for tok in result)
        has_oov = any(tok.is_oov for tok in result)
        if not has_low_conf and not has_oov:
            continue

        for idx, tok in enumerate(result):
            rows.append({
                "sentence_id":        current_sid,
                "timestamp":          now_iso,
                "full_sentence":      sentence,
                "token_index":        idx,
                "token":              tok.token,
                "model_tag":          tok.tag,
                "user-corrected_tag": "",
                "confidence":         round(tok.confidence, 4),
                "is_oov":             tok.is_oov,
            })
        current_sid += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path.resolve()
