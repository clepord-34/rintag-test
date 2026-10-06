from rintag import tag

class TestTagger:
    def test_tag_returns_list_of_tuples(self):
        result = tag("nag-dalagan su igin")
        assert isinstance(result, list)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_tag_preserves_token_count(self):
        result = tag("nag-dalagan su igin")
        assert len(result) == 3

    def test_tag_empty_string(self):
        result = tag("")
        assert result == []

    def test_tags_are_valid_ud(self):
        valid_tags = {"NOUN","VERB","PRON","ADP","PUNCT","ADV","PART",
                      "DET","ADJ","PROPN","CCONJ","SCONJ","INTJ","NUM","AUX","X","SYM"}
        result = tag("nag-dalagan su igin")
        for _, pos_tag in result:
            assert pos_tag in valid_tags
