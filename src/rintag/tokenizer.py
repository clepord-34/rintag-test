import re

_LEAD = re.compile(r"^([^\w'''\`\-]+)",  re.UNICODE)
_TAIL = re.compile(r"([^\w'''\`\-]+)$",  re.UNICODE)
_INT  = re.compile(r"([.,;:!?]+)",       re.UNICODE)

def tokenize(sentence: str) -> list[tuple[str, str]]:
    """
    Return ordered (token, pos_tag) pairs for a sentence.
    Punctuation tokens receive pos_tag='PUNCT'; word tokens receive pos_tag=''.
    """
    result = []
    for tok in sentence.split():

        # 1. Leading punctuation → PUNCT
        lead_m = _LEAD.match(tok)
        if lead_m:
            result.append((lead_m.group(1), "PUNCT"))
            tok = tok[lead_m.end():]
        if not tok:
            continue

        # 2. Separate trailing punctuation (emitted after middle is processed)
        tail_m = _TAIL.search(tok)
        if tail_m:
            middle = tok[:tail_m.start()]
            tail   = tail_m.group(1)
        else:
            middle = tok
            tail   = ""

        # 3. Internal punctuation in middle: split and emit
        if middle:
            parts      = _INT.split(middle)   # alternates: word, punct, word, …
            words_only = [p for i, p in enumerate(parts) if i % 2 == 0 and p]
            puncts_seg = [p for i, p in enumerate(parts) if i % 2 == 1 and p]

            # Keep thousand-separator numbers intact (e.g. 411,418 → one token)
            if puncts_seg and all(re.fullmatch(r"\d+", w) for w in words_only if w):
                result.append((middle, ""))
            else:
                for i, seg in enumerate(parts):
                    if not seg:
                        continue
                    if i % 2 == 0:   # word segment
                        if re.search(r"[a-zA-Z0-9]", seg):
                            result.append((seg, ""))
                    else:            # punctuation segment
                        result.append((seg, "PUNCT"))

        # 4. Trailing punctuation → PUNCT
        if tail:
            result.append((tail, "PUNCT"))

    return result

def tokenize_sentence(text: str) -> list[str]:
    """Return just the token strings (no PUNCT/blank tags).
    
    Example:
        tokenize_sentence("nag-dalagan su igin.")
        # Returns: ["nag-dalagan", "su", "igin", "."]
    """
    return [token for token, _ in tokenize(text)]


def tokenize_to_pairs(text: str) -> list[tuple[str, str]]:
    """Return (token, tag_hint) pairs. Punctuation gets 'PUNCT', words get ''.
    
    Example:
        tokenize_to_pairs("nag-dalagan su igin.")
        # Returns: [("nag-dalagan", ""), ("su", ""), ("igin", ""), (".", "PUNCT")]
    """
    return tokenize(text)
