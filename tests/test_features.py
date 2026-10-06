import pytest

from rintag.features import (
    BASELINE_GROUPS,
    DEFAULT_GROUPS,
    ablation_configs,
    sent2features,
    word2features,
)

SENT = ["marhay", "na", "aldaw", "po", "."]


class TestGroups:
    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            word2features(SENT, 0, {"windw2": True})

    def test_baseline_has_no_extended_features(self):
        feats = word2features(SENT, 2, BASELINE_GROUPS)
        assert "-2:word.lower()" not in feats
        assert "+2:word.lower()" not in feats
        assert "word.shape" not in feats
        assert "word[-1:]" not in feats
        assert "-1:word.lower()|word.lower()" not in feats
        assert "-1:word.lower()" in feats  # base +/-1 window is always on

    def test_default_is_full(self):
        assert word2features(SENT, 2) == word2features(SENT, 2, DEFAULT_GROUPS)

    def test_ablation_configs_are_valid(self):
        configs = ablation_configs()
        assert "baseline" in configs and "full" in configs
        for groups in configs.values():
            word2features(SENT, 1, groups)  # must not raise


class TestNewFeatures:
    def test_bigram_conjunctions(self):
        feats = word2features(SENT, 1)
        assert feats["-1:word.lower()|word.lower()"] == "marhay|na"
        assert feats["word.lower()|+1:word.lower()"] == "na|aldaw"

    def test_bigrams_respect_boundaries(self):
        first = word2features(SENT, 0)
        last = word2features(SENT, 4)
        assert not any(k.startswith("-1:") and "|" in k for k in first)
        assert not any("|+1:" in k for k in last)

    def test_extended_affixes(self):
        feats = word2features(["dalagan"], 0)
        assert feats["word[-1:]"] == "n"
        assert feats["word[:1]"] == "d"
        assert feats["word[-5:]"] == "lagan"
        assert feats["word[:5]"] == "dalag"

    def test_shape(self):
        assert word2features(["Nag-abot"], 0)["word.shape"] == "Xx-x"
        assert word2features(["pa'la"], 0)["word.shape"] == "x'x"
        assert word2features(["2024"], 0)["word.shape"] == "d"

    def test_repeat_features(self):
        laugh = word2features(["hahaha"], 0)
        assert laugh["word.has_unit_repeat"] is True
        assert laugh["word.is_full_repeat"] is True
        stretched = word2features(["ayyy"], 0)
        assert stretched["word.has_char_run"] is True
        plain = word2features(["dalagan"], 0)
        assert plain["word.has_char_run"] is False
        assert plain["word.is_full_repeat"] is False

    def test_neighbor_punctuation(self):
        feats = word2features(SENT, 3)
        assert feats["+1:word.is_punctuation"] is True
        assert feats["-1:word.is_punctuation"] is False


class TestShape:
    def test_single_token_sentence(self):
        feats = sent2features(["hmm"])[0]
        assert feats["BOS"] is True and feats["EOS"] is True

    def test_two_token_sentence_no_crash(self):
        assert len(sent2features(["a", "b"])) == 2

    def test_values_are_crfsuite_compatible(self):
        for feats in sent2features(SENT):
            for key, value in feats.items():
                assert isinstance(key, str) and key
                assert isinstance(value, (str, bool, int, float)), key


class TestMissingTokens:
    def test_nan_token_is_treated_as_nan_string(self):
        nan = float("nan")
        feats = sent2features(["banwaan", nan, "mampaktonowon"])
        assert feats[1]["word.lower()"] == "nan"
        assert feats[0]["+1:word.lower()"] == "nan"
        assert feats[2]["-1:word.lower()"] == "nan"