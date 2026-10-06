"""Surface-only CRF feature extraction for RinTag.

Every feature is computed from token strings only (characters, casing,
position, neighbouring strings). No lexicons, no tag knowledge, no
morphological rules.

Feature groups
--------------
The base features (word identity, 2-4 char prefixes/suffixes, casing,
digit/hyphen/punctuation flags, +/-1 context, BOS/EOS) are always on.
The groups below add to them and can be switched on/off for ablation:

    window2         +/-2 token context (same cues as the +/-1 window)
    bigrams         string conjunctions of neighbouring tokens with the
                    target token (e.g. "prev_word|word")
    affix_ext       1- and 5-char prefixes/suffixes of the target token
    shape           collapsed character shape, e.g. "Xx", "x-x", "x'x"
    repeat          orthographic repetition (char runs like "ayyy",
                    repeated units like "hahaha")
    neighbor_punct  whether the previous/next token is punctuation

IMPORTANT: a trained model only works with the same groups it was trained
with. The library's tag() must call sent2features() with the same groups
used at training time. DEFAULT_GROUPS is what the library uses; do not edit
it after the final model is trained, or retrain.
"""

import re
import unicodedata

DEFAULT_GROUPS = {
    "window2": True,
    "bigrams": True,
    "affix_ext": True,
    "shape": True,
    "repeat": True,
    "neighbor_punct": True,
}

# Reproduces the original feature set exactly (+/-1 window, no extras).
BASELINE_GROUPS = {name: False for name in DEFAULT_GROUPS}

_CHAR_RUN = re.compile(r"(.)\1\1")            # same char 3+ times: "ayyy"
_UNIT_REPEAT = re.compile(r"(.{2,})\1")       # repeated unit anywhere: "hahaha"
_FULL_REPEAT = re.compile(r"(.+?)\1+")        # whole token is one unit repeated


def resolve_groups(groups: dict | None = None) -> dict:
    """Merge user overrides onto DEFAULT_GROUPS (typos raise, not silently ignored)."""
    if groups is None:
        return DEFAULT_GROUPS
    unknown = set(groups) - set(DEFAULT_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {sorted(unknown)}. "
            f"Valid groups: {sorted(DEFAULT_GROUPS)}"
        )
    return {**DEFAULT_GROUPS, **groups}


def ablation_configs() -> dict[str, dict]:
    """Named group configurations for an ablation table.

    'baseline' = original features; each '+X' adds one group to the baseline;
    'full' = DEFAULT_GROUPS; each '-X' removes one group from 'full'.
    """
    configs = {"baseline": dict(BASELINE_GROUPS)}
    for name in DEFAULT_GROUPS:
        configs[f"+{name}"] = {**BASELINE_GROUPS, name: True}
    configs["full"] = dict(DEFAULT_GROUPS)
    for name, on in DEFAULT_GROUPS.items():
        if on:
            configs[f"-{name}"] = {**DEFAULT_GROUPS, name: False}
    return configs


def _s(token) -> str:
    """Coerce a token to str (a missing value such as NaN becomes 'nan').

    Some tokens in the CSVs (e.g. the word 'nan') are read back as NaN by
    pandas; depending on the pandas version, .astype(str) either converts
    them to 'nan' or leaves a float. This makes both behave the same.
    """
    return token if isinstance(token, str) else str(token)


def _is_punct(token: str) -> bool:
    return bool(token) and all(
        unicodedata.category(char).startswith("P") for char in token
    )


def _short_shape(token: str) -> str:
    """Map chars to X (upper), x (lower), d (digit); keep other chars; collapse runs."""
    out = []
    for ch in token:
        if ch.isupper():
            c = "X"
        elif ch.islower():
            c = "x"
        elif ch.isdigit():
            c = "d"
        else:
            c = ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def _context_cues(prefix: str, token: str) -> dict:
    """Cues for a neighbouring token (same set used for -2, -1, +1, +2)."""
    lower = token.lower()
    return {
        f"{prefix}:word.lower()": lower,
        f"{prefix}:word[-2:]": lower[-2:],
        f"{prefix}:word[-3:]": lower[-3:],
        f"{prefix}:word[:2]": lower[:2],
        f"{prefix}:word[:3]": lower[:3],
        f"{prefix}:word.istitle()": token.istitle(),
        f"{prefix}:word.isupper()": token.isupper(),
    }


def word2features(sent: list[str], i: int, groups: dict | None = None) -> dict:
    """Extract CRF features for the token at position i in the sentence.

    Args:
        sent: List of token strings, e.g. ["nag-dalagan", "su", "igin"]
        i: Index of the target token (0-based)
        groups: Optional overrides for feature groups (see module docstring).

    Returns:
        Dictionary of feature name -> feature value pairs.
    """
    g = resolve_groups(groups)
    n = len(sent)
    word = _s(sent[i])
    word_lower = word.lower()

    features = {
        "bias": 1.0,
        "word.lower()": word_lower,
        # -- Suffixes (length 2, 3, 4) --
        "word[-2:]": word_lower[-2:],
        "word[-3:]": word_lower[-3:],
        "word[-4:]": word_lower[-4:],
        # -- Prefixes (length 2, 3, 4) --
        "word[:2]": word_lower[:2],
        "word[:3]": word_lower[:3],
        "word[:4]": word_lower[:4],
        # -- Token properties --
        "word.len": len(word),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.has_hyphen": "-" in word,
        "word.is_punctuation": _is_punct(word),
    }

    # -- Extended affixes of the target token --
    if g["affix_ext"]:
        features.update({
            "word[-1:]": word_lower[-1:],
            "word[-5:]": word_lower[-5:],
            "word[:1]": word_lower[:1],
            "word[:5]": word_lower[:5],
        })

    # -- Character shape --
    if g["shape"]:
        features["word.shape"] = _short_shape(word)

    # -- Orthographic repetition (laughter, elongation, repeated units) --
    if g["repeat"]:
        features.update({
            "word.has_char_run": bool(_CHAR_RUN.search(word_lower)),
            "word.has_unit_repeat": bool(_UNIT_REPEAT.search(word_lower)),
            "word.is_full_repeat": bool(_FULL_REPEAT.fullmatch(word_lower)),
        })

    # -- Previous token (i-1) context --
    if i > 0:
        features.update(_context_cues("-1", _s(sent[i - 1])))
        if g["neighbor_punct"]:
            features["-1:word.is_punctuation"] = _is_punct(_s(sent[i - 1]))
    else:
        features["BOS"] = True   # Beginning Of Sentence

    # -- Token i-2 context --
    if g["window2"] and i > 1:
        features.update(_context_cues("-2", _s(sent[i - 2])))

    # -- Next token (i+1) context --
    if i < n - 1:
        features.update(_context_cues("+1", _s(sent[i + 1])))
        if g["neighbor_punct"]:
            features["+1:word.is_punctuation"] = _is_punct(_s(sent[i + 1]))
    else:
        features["EOS"] = True   # End Of Sentence

    # -- Token i+2 context --
    if g["window2"] and i < n - 2:
        features.update(_context_cues("+2", _s(sent[i + 2])))

    # -- String conjunctions with neighbours --
    if g["bigrams"]:
        if i > 0:
            prev = _s(sent[i - 1]).lower()
            features.update({
                "-1:word.lower()|word.lower()": f"{prev}|{word_lower}",
                "-1:word[-3:]|word.lower()": f"{prev[-3:]}|{word_lower}",
                "-1:word[:2]|word.lower()": f"{prev[:2]}|{word_lower}",
                "-1:word[-2:]|word[-2:]": f"{prev[-2:]}|{word_lower[-2:]}",
            })
        if i < n - 1:
            nxt = _s(sent[i + 1]).lower()
            features.update({
                "word.lower()|+1:word.lower()": f"{word_lower}|{nxt}",
                "word.lower()|+1:word[:3]": f"{word_lower}|{nxt[:3]}",
                "word.lower()|+1:word[-3:]": f"{word_lower}|{nxt[-3:]}",
                "word[-2:]|+1:word[-2:]": f"{word_lower[-2:]}|{nxt[-2:]}",
            })

    return features


def sent2features(sent: list[str], groups: dict | None = None) -> list[dict]:
    """Convert a full sentence (list of tokens) to a list of feature dicts."""
    g = resolve_groups(groups)
    return [word2features(sent, i, g) for i in range(len(sent))]


def sent2labels(tagged_sent: list[tuple[str, str]]) -> list[str]:
    """Extract just the POS labels from a list of (token, label) pairs."""
    return [label for _, label in tagged_sent]


def sent2tokens(tagged_sent: list[tuple[str, str]]) -> list[str]:
    """Extract just the tokens from a list of (token, label) pairs."""
    return [token for token, _ in tagged_sent]