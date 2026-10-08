from rintag import tag
from rintag.tagger import TagResult, TaggedToken

class TestTagger:
    def test_tag_returns_tagresult(self):
        result = tag("nag-dalagan su igin")
        assert isinstance(result, TagResult)
        assert len(result) == 3
        
        # Test property access
        assert result.tokens == ["nag-dalagan", "su", "igin"]
        assert result.tags[1] == "DET"
        assert result.pairs == [("nag-dalagan", "VERB"), ("su", "DET"), ("igin", "NOUN")]
        
        # Test item access
        first_token = result[0]
        assert isinstance(first_token, TaggedToken)
        assert first_token.token == "nag-dalagan"
        assert first_token.tag == "VERB"
        assert isinstance(first_token.confidence, float)
        assert isinstance(first_token.alternatives, list)
        assert len(first_token.alternatives) > 0
        assert isinstance(first_token.features, dict)
        
    def test_tag_tuple_iteration(self):
        result = tag("nag-dalagan su igin")
        
        # pairs property should return list of tuples
        pairs = result.pairs
        assert pairs == [("nag-dalagan", "VERB"), ("su", "DET"), ("igin", "NOUN")]
        
        # Explicit unpacking in a list comprehension should also yield tuples
        unpacked = [(token, pos) for token, pos in result]
        assert unpacked == pairs

    def test_tag_empty_string(self):
        result = tag("")
        assert isinstance(result, TagResult)
        assert len(result) == 0

    def test_tags_are_valid_ud(self):
        valid_tags = {"NOUN","VERB","PRON","ADP","PUNCT","ADV","PART",
                      "DET","ADJ","PROPN","CCONJ","SCONJ","INTJ","NUM","AUX","X","SYM"}
        result = tag("nag-dalagan su igin")
        for token, pos_tag in result:
            assert pos_tag in valid_tags
