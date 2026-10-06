import pytest
from rintag.tokenizer import tokenize_sentence, tokenize_to_pairs

class TestTokenizer:
    def test_hyphenated_word_stays_together(self):
        """nag-dalagan must be ONE token, not split on the hyphen."""
        assert tokenize_sentence("nag-dalagan") == ["nag-dalagan"]

    def test_apostrophe_word_stays_together(self):
        """pa'la must be ONE token, not split on the apostrophe."""
        assert tokenize_sentence("pa'la") == ["pa'la"]

    def test_apostrophe_d_stays_together(self):
        """nangga'd must be ONE token."""
        assert tokenize_sentence("nangga'd") == ["nangga'd"]

    def test_leading_apostrophe_stays(self):
        """'nganing must be ONE token."""
        assert tokenize_sentence("'nganing") == ["'nganing"]

    def test_trailing_period_splits(self):
        """sadi. must become two tokens: sadi and ."""
        result = tokenize_sentence("sadi.")
        assert result == ["sadi", "."]

    def test_punctuation_tagged_as_punct(self):
        pairs = tokenize_to_pairs("sadi.")
        assert pairs == [("sadi", ""), (".", "PUNCT")]

    def test_full_sentence(self):
        tokens = tokenize_sentence("nag-dalagan su igin.")
        assert tokens == ["nag-dalagan", "su", "igin", "."]

    def test_number_with_comma_stays_together(self):
        """411,418 is a number and must stay as one token."""
        assert tokenize_sentence("411,418") == ["411,418"]
