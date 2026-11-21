import csv
from pathlib import Path
from rintag import (
    tag,
    flag_low_confidence,
    export_feedback,
    enable_passive_feedback,
    disable_passive_feedback,
    get_passive_feedback,
    export_passive_feedback,
    clear_passive_feedback,
    collect_feedback,
)
from rintag.feedback import FEEDBACK_HEADERS, FeedbackCollector
from rintag.tagger import TagResult, TaggedToken


class TestFlagLowConfidence:
    def test_returns_list_at_high_threshold(self):
        result = tag("nagdalagan su igin")
        flagged = flag_low_confidence(result, threshold=1.0)
        assert len(flagged) == 3

    def test_threshold_zero_flags_nothing(self):
        result = tag("nagdalagan su igin")
        assert flag_low_confidence(result, threshold=0.0) == []

    def test_flagged_includes_oov_field(self):
        result = tag("nagdalagan su igin")
        flagged = flag_low_confidence(result, threshold=1.0)
        for f in flagged:
            assert "is_oov" in f
            assert isinstance(f["is_oov"], bool)


class TestOOVDetection:
    def test_is_oov_attribute_exists(self):
        result = tag("nagdalagan su igin")
        for tok in result:
            assert hasattr(tok, "is_oov")
            assert isinstance(tok.is_oov, bool)

    def test_oov_tokens_property(self):
        result = tag("nagdalagan su igin")
        oov = result.oov_tokens
        assert isinstance(oov, list)

    def test_is_oov_list_property(self):
        result = tag("nagdalagan su igin")
        assert len(result.is_oov) == len(result)
        assert all(isinstance(val, bool) for val in result.is_oov)


class TestExportFeedback:
    def test_exports_full_sentence_with_sentence_id(self, tmp_path):
        result = tag("nagdalagan su igin")
        out = tmp_path / "test_feedback.csv"
        export_feedback([("nagdalagan su igin", result)], out, threshold=1.0, start_sentence_id=42)

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        assert "sentence_id" in fieldnames
        assert "is_oov" in fieldnames
        assert "low_confidence" in fieldnames
        assert "user_correction" in fieldnames

        assert len(rows) == 3
        assert rows[0]["sentence_id"] == "42"
        assert rows[0]["sentence"] == "nagdalagan su igin"
        assert rows[0]["token_index"] == "0"
        assert rows[0]["token"] == "nagdalagan"
        assert rows[1]["sentence_id"] == "42"
        assert rows[2]["sentence_id"] == "42"

    def test_skips_when_all_tokens_high_confidence_and_not_oov(self, tmp_path):
        # "su igin" are both frequent words in training set
        result = tag("su igin")
        out = tmp_path / "test_feedback_skip.csv"
        # threshold 0.0 and if no OOV, should skip
        if not any(result.is_oov):
            export_feedback([("su igin", result)], out, threshold=0.0)
            with open(out, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 0


class TestPassiveFeedback:
    def setup_method(self):
        disable_passive_feedback()
        clear_passive_feedback()

    def teardown_method(self):
        disable_passive_feedback()
        clear_passive_feedback()

    def test_passive_feedback_disabled_by_default(self):
        tag("wagas na sasalan xyz123")
        assert len(get_passive_feedback()) == 0

    def test_enable_and_capture_uncertain_sentence(self):
        enable_passive_feedback(threshold=0.90)
        result = tag("wagas na sasalan")
        records = get_passive_feedback()
        assert len(records) > 0
        assert records[0]["full_sentence"] == "wagas na sasalan"
        assert "sentence_id" in records[0]
        assert "token" in records[0]
        assert "confidence" in records[0]
        assert "user-corrected_tag" in records[0]
        assert "validator_tag" not in records[0]
        assert "was_corrected" not in records[0]
        for token, record in zip(result, records):
            expected = next(
                (alt["tag"] for alt in token.alternatives if alt["tag"] != token.tag),
                "",
            )
            assert record["top_alternative"] == expected

    def test_passive_feedback_defaults_to_75_percent_threshold(self):
        collector = enable_passive_feedback()
        assert collector.threshold == 0.75

    def test_top_alternative_is_included_for_high_confidence_tokens(self):
        collector = FeedbackCollector(enabled=True, threshold=0.75)
        result = TagResult([
            TaggedToken(
                token="certain",
                tag="NOUN",
                confidence=0.99,
                alternatives=[{"tag": "NOUN", "probability": 0.99}, {"tag": "VERB", "probability": 0.01}],
                features={},
            ),
            TaggedToken(
                token="uncertain",
                tag="ADJ",
                confidence=0.50,
                alternatives=[{"tag": "ADJ", "probability": 0.50}, {"tag": "NOUN", "probability": 0.40}],
                features={},
            ),
        ])

        assert collector.record("certain uncertain", result)
        records = collector.get_buffer()
        assert records[0]["top_alternative"] == "VERB"
        assert records[1]["top_alternative"] == "NOUN"

    def test_export_passive_feedback(self, tmp_path):
        out = tmp_path / "passive_out.csv"
        enable_passive_feedback(threshold=1.0)
        tag("nagdalagan su igin")
        exported = export_passive_feedback(out)
        assert exported.exists()

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == FEEDBACK_HEADERS
            rows = list(reader)
        assert len(rows) == 3

    def test_collect_feedback_context_manager(self, tmp_path):
        out = tmp_path / "context_out.csv"
        with collect_feedback(output_path=out, threshold=1.0) as col:
            tag("Punan ta na")
            assert col.sentence_count == 1

        # Collector should export to file upon exit if output_path was given
        assert out.exists()
        # Passive feedback should be disabled outside context
        tag("ibani na 'ya bibi")
        assert get_passive_feedback() == []

    def test_deduplicate_sentences(self):
        enable_passive_feedback(threshold=1.0, deduplicate=True)
        tag("Punan ta na")
        tag("Punan ta na")  # identical sentence tagged again
        records = get_passive_feedback()
        # "Punan ta na" has 3 tokens, should only appear once (3 rows, sentence_id 1)
        sentence_ids = set(r["sentence_id"] for r in records)
        assert len(sentence_ids) == 1

