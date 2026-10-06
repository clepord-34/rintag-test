"""Main tagger module. Loads the bundled CRF model and exposes tag()."""

import pickle
from pathlib import Path

from rintag.tokenizer import tokenize_sentence
from rintag.features import sent2features

_MODEL_PATH = Path(__file__).parent / "models" / "crf_model.pkl"
_model = None


def _load_model():
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def tag(text: str) -> list[tuple[str, str]]:
    """Tag a Rinconada Bikol sentence.
    
    Args:
        text: Raw RBL text string (one sentence).
    
    Returns:
        List of (token, POS_tag) tuples.
    
    Example:
        >>> tag("nag-dalagan su igin")
        [("nag-dalagan", "VERB"), ("su", "DET"), ("igin", "NOUN")]
    """
    tokens = tokenize_sentence(text)
    if not tokens:
        return []
    features = sent2features(tokens)
    model = _load_model()
    labels = model.predict([features])[0]
    return list(zip(tokens, labels))


def tag_detailed(text: str, top_k: int = 5) -> list[dict]:
    """Tag a Rinconada Bikol sentence with confidence scores and alternative tags.
    
    Args:
        text: Raw RBL text string (one sentence).
        top_k: Number of alternative tags to include.
        
    Returns:
        List of dicts, one per token, containing tag, confidence, alternatives, and features.
    """
    tokens = tokenize_sentence(text)
    if not tokens:
        return []
    features = sent2features(tokens)
    model = _load_model()
    labels = model.predict([features])[0]
    marginals = model.predict_marginals([features])[0]
    
    results = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        token_marginals = marginals[i]
        sorted_tags = sorted(token_marginals.items(),
                             key=lambda x: x[1], reverse=True)
        results.append({
            "token": token,
            "tag": label,
            "confidence": token_marginals.get(label, 0.0),
            "alternatives": [
                {"tag": t, "probability": round(p, 6)}
                for t, p in sorted_tags[:top_k]
            ],
            "features": {k: v for k, v in features[i].items()
                         if not k.startswith("bias")},
        })
    return results
