# RinTag

A fast, lightweight, Python-based Part-of-Speech (POS) tagger for the **Rinconada Bikol Language** (*Riŋkonāda* / Inland Bikol) using Conditional Random Fields (CRF).

[![PyPI Version](https://img.shields.io/pypi/v/rintag.svg)](https://pypi.org/project/rintag/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rintag.svg)](https://pypi.org/project/rintag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**RinTag** addresses the digital underrepresentation of regional Philippine languages by providing an out-of-the-box POS tagger for Rinconada Bikol. It uses a bundled Linear-Chain Conditional Random Field (CRF) trained on 2,000 carefully curated and validated sentences (31,035 tokens) following the **Universal Dependencies (UD)** v2 guidelines across 15 part-of-speech tags.

- **Zero configuration:** Bundles pre-trained CRF weights and training vocabulary.
- **Rich token metadata:** Provides predicted UPOS tags, token-level marginal confidence scores, alternative tag rankings, and feature explanations.
- **Built-in Out-of-Vocabulary (OOV) detection:** Flags words not seen during training.
- **Opt-in Passive Feedback:** Allows developers and linguists to capture uncertain or OOV predictions during inference and export them for human review.

---

## Source Code & Community

- **GitHub Repository:** [https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>) *(Source code, corpus preparation scripts, interactive web demo, and thesis benchmarks)*
- **Bug Reports & Tag Feedback:** [https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/issues](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/issues)

---

## Installation

```bash
pip install rintag
```

### Requirements
- Python `>= 3.10`
- `sklearn-crfsuite >= 0.3.6`
- `scikit-learn >= 1.0`

---

## Quickstart

```python
import rintag

# Tag a sentence
result = rintag.tag("nag-dalagan su igin")

# 1. Iterate like (token, tag) tuples (backward-compatible)
for token, tag in result:
    print(f"{token} -> {tag}")
# nag-dalagan -> VERB
# su -> DET
# igin -> NOUN

# 2. Access bulk properties directly
print(result.tokens)       # ['nag-dalagan', 'su', 'igin']
print(result.tags)         # ['VERB', 'DET', 'NOUN']
print(result.confidence)   # [0.9834, 0.9521, 0.9912]
print(result.is_oov)       # [True, False, False]
print(result.oov_tokens)   # ['nag-dalagan']

# 3. Inspect individual TaggedToken objects
first = result[0]
print(first.token)         # 'nag-dalagan'
print(first.tag)           # 'VERB'
print(first.confidence)    # 0.9834
print(first.is_oov)        # True
print(first.alternatives)  # [{'tag': 'VERB', 'probability': 0.9834}, ...]
```

---

## Universal Dependencies (UD) Tagset

RinTag uses 15 Universal Dependencies (UD) part-of-speech tags:

| Tag | Category | Description | Example (Rinconada Bikol) |
|---|---|---|---|
| `NOUN` | Noun | Common nouns, entities, objects | *igin* (child), *balay* (house) |
| `PROPN` | Proper Noun | Names of persons, places, entities | *Iriga*, *Bikol*, *Maria* |
| `VERB` | Verb | Actions, states, processes | *nag-dalagan* (ran), *magkaon* (to eat) |
| `ADJ` | Adjective | Qualifiers and descriptors | *dakula* (big), *sadit* (small) |
| `ADV` | Adverb | Time, manner, degree modifiers | *nguwan* (now), *mabilis* (quickly) |
| `PRON` | Pronoun | Personal and demonstrative pronouns | *ako* (I/me), *ika* (you), *kamo* (you pl.) |
| `DET` | Determiner | Articles and markers | *su* (the/focus), *adong* (that) |
| `ADP` | Adposition | Prepositions / case markers | *sa* (in/to/at), *para* (for) |
| `AUX` | Auxiliary | Auxiliary / modal markers | *dapat* (must), *puydi* (can) |
| `CCONJ` | Coordinating Conjunction | Joining coordinate phrases | *ag* / *kag* (and), *o* (or) |
| `SCONJ` | Subordinating Conjunction | Introducing subordinate clauses | *ta* (because), *kung* (if) |
| `PART` | Particle | Enclitics, aspect/mood particles | *na* (already/now), *pa* (still), *baga* |
| `NUM` | Numeral | Cardinal and ordinal numbers | *usad* (one), *duwa* (two), *10* |
| `INTJ` | Interjection | Expressive exclamations | *ay*, *oy*, *marhay* |
| `PUNCT` | Punctuation | Punctuation marks | `.`, `,`, `?`, `!`, `-` |

---

## Human-in-the-Loop (HITL) Passive Feedback

You can help improve RinTag by opting in to passive background feedback. RinTag silently buffers sentences containing low-confidence predictions or out-of-vocabulary (OOV) tokens:

```python
import rintag

# Enable passive collection (defaults: threshold=0.75, log_oov=True)
rintag.enable_passive_feedback()

# Use RinTag normally in your pipeline or web application:
for doc in documents:
    result = rintag.tag(doc)

# Export the buffered sentences whenever you are ready:
rintag.export_passive_feedback("rintag_feedback.csv")
```

### Scoped Collection via Context Manager

```python
with rintag.collect_feedback("audit_batch.csv", threshold=0.80):
    for text in raw_texts:
        rintag.tag(text)
```

### Reviewing and Contributing Feedback

Open the exported CSV and inspect the `user-corrected_tag` column:
1. Leave blank if the model prediction (`model_tag`) is correct.
2. Enter the valid UPOS tag if the prediction is incorrect.
3. Share the reviewed CSV via a [GitHub Issue](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/issues) using the **Tag Correction Feedback** template.

---

## Authors

- **Vince Clifford C. Aguilar**
- **Nhel Adam S. Benosa**
- **Elyssa Olivares**

*College of Computer Studies, Camarines Sur Polytechnic Colleges (CSPC)*

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
