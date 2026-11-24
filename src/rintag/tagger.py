"""Main tagger module. Loads the bundled CRF model and exposes tag()."""

import pickle
from pathlib import Path

from rintag.tokenizer import tokenize_sentence
from rintag.features import sent2features

_MODEL_PATH = Path(__file__).parent / "models" / "crf_model.pkl"
_VOCAB_PATH = Path(__file__).parent / "models" / "vocab.pkl"
_model = None
_vocab = None


def _load_model():
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _load_vocab() -> set:
    """Load the training vocabulary for OOV detection."""
    global _vocab
    if _vocab is None:
        if _VOCAB_PATH.exists():
            with open(_VOCAB_PATH, "rb") as f:
                _vocab = pickle.load(f)
        else:
            _vocab = set()  # Graceful fallback: treat nothing as OOV
    return _vocab


def reload_model() -> None:
    """Clear cached model and vocab, forcing reload from disk on next tag() call.
    """
    global _model, _vocab
    _model = None
    _vocab = None


class TaggedToken:
    __slots__ = ('token', 'tag', 'confidence', 'alternatives', 'features', 'is_oov')

    def __init__(
        self,
        token: str,
        tag: str,
        confidence: float,
        alternatives: list[dict],
        features: dict,
        is_oov: bool = False,
    ):
        self.token = token
        self.tag = tag
        self.confidence = confidence
        self.alternatives = alternatives
        self.features = features
        self.is_oov = is_oov

    def __iter__(self):
        yield self.token
        yield self.tag

    def __repr__(self) -> str:
        oov = " OOV" if self.is_oov else ""
        return f"TaggedToken({self.token!r}, {self.tag!r}, confidence={self.confidence:.4f}{oov})"


class TagResult:
    def __init__(self, tagged_tokens: list[TaggedToken]):
        self._tokens = tagged_tokens

    @property
    def tokens(self) -> list[str]:
        return [t.token for t in self._tokens]

    @property
    def tags(self) -> list[str]:
        return [t.tag for t in self._tokens]

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(t.token, t.tag) for t in self._tokens]

    @property
    def confidence(self) -> list[float]:
        return [t.confidence for t in self._tokens]

    @property
    def alternatives(self) -> list[list[dict]]:
        return [t.alternatives for t in self._tokens]

    @property
    def features(self) -> list[dict]:
        return [t.features for t in self._tokens]

    @property
    def is_oov(self) -> list[bool]:
        """Return list of booleans indicating whether each token is out-of-vocabulary."""
        return [t.is_oov for t in self._tokens]

    @property
    def oov_tokens(self) -> list[str]:
        """Return tokens that are out-of-vocabulary."""
        return [t.token for t in self._tokens if t.is_oov]

    def __getitem__(self, index: int) -> TaggedToken:
        return self._tokens[index]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self) -> str:
        return f"TagResult({self.pairs!r})"


def tag(text: str) -> TagResult:
    """Tag a Rinconada Bikol sentence with confidence scores and alternative tags.
    
    Args:
        text: Raw RBL text string (one sentence).
        
    Returns:
        A TagResult object containing tokens, tags, confidence, alternatives,
        features, and OOV status.
    """
    tokens = tokenize_sentence(text)
    if not tokens:
        return TagResult([])
        
    features = sent2features(tokens)
    model = _load_model()
    vocab = _load_vocab()
    labels = model.predict([features])[0]
    marginals = model.predict_marginals([features])[0]
    
    results = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        token_marginals = marginals[i]
        sorted_tags = sorted(token_marginals.items(), key=lambda x: x[1], reverse=True)
        
        results.append(TaggedToken(
            token=token,
            tag=label,
            confidence=token_marginals.get(label, 0.0),
            alternatives=[
                {"tag": t, "probability": round(p, 6)}
                for t, p in sorted_tags
            ],
            features={k: v for k, v in features[i].items() if not k.startswith("bias")},
            is_oov=token.lower() not in vocab,
        ))
        
    tag_result = TagResult(results)

    # Passive background feedback (if enabled)
    from rintag.feedback import get_feedback_collector
    collector = get_feedback_collector()
    if collector.enabled:
        collector.record(text, tag_result)

    return tag_result
