# RinTag

A Python-based POS tagger for the Rinconada Bikol Language using Conditional Random Fields (CRF).

## Installation

```bash
pip install rintag
```

## Usage

```python
import rintag

# Tag a sentence
result = rintag.tag("nag-dalagan su igin")

# Iterate like tuples (backward compatible)
for token, tag in result:
    print(f"{token} -> {tag}")

# Access bulk properties directly
print(result.tokens)       # ['nag-dalagan', 'su', 'igin']
print(result.tags)         # ['VERB', 'DET', 'NOUN']
print(result.confidence)   # [0.9834, 0.9521, 0.9912]
print(result.is_oov)       # [True, False, False] (Out-of-Vocabulary flags)
print(result.oov_tokens)   # ['nag-dalagan']

# Get detailed info for a specific token
first = result[0]
print(first.token)         # 'nag-dalagan'
print(first.tag)           # 'VERB'
print(first.confidence)    # 0.9834
print(first.is_oov)        # True
print(first.alternatives[:3]) # Top 3 alternative tags
print(first.features)      # CRF features used for this token
```

## Human-in-the-Loop (HITL) Feedback

RinTag supports two complementary feedback strategies:

### 1. Help Improve RinTag Through Passive Feedback
You can help the RinTag developers improve the tagger by opting in to passive feedback while using the library. RinTag buffers sentences that contain a low-confidence prediction or an OOV token; it includes every token in a captured sentence so reviewers have context. Feedback is not sent automatically. You decide when to export, review, and share it.

```python
import rintag

# Enable feedback collection (threshold=0.75 and OOV logging are defaults)
rintag.enable_passive_feedback()

# Continue using RinTag normally:
for text in my_sentences:
    result = rintag.tag(text)

# Export the collected rows whenever you are ready to review them:
rintag.export_passive_feedback("rintag_feedback.csv")
```

Open the exported CSV in a spreadsheet editor and review the `user-corrected_tag` column:

- Leave `user-corrected_tag` blank when you agree with `model_tag`.
- If the prediction is wrong, enter the correct UPOS tag in `user-corrected_tag`.
- Treat `top_alternative` as a model suggestion, not as a verified correction.
- Keep the sentence and token fields intact so each correction retains its context.

When convenient, share the reviewed CSV with the RinTag developers to contribute corrections for future model improvements. You can also scope collection to a specific batch using a context manager:

```python
with rintag.collect_feedback("audit.csv"):
    for doc in scraped_documents:
        rintag.tag(doc)
```

The passive-feedback threshold defaults to `0.75`, and OOV logging is enabled by default for both APIs; either option can be configured explicitly. Captured token rows include `top_alternative`, the highest-probability tag other than the model prediction whenever one is available, regardless of confidence. Library feedback exports contain no validator-only fields.

### 2. Maintainer Active Learning (Uncertainty Sampling Showcase)
For dataset curators and researchers: batch-tag unseen raw corpus sentences, rank them by model uncertainty, and export the hardest samples for expert annotation:

```bash
# Maintainer active learning on raw corpus:
python demo/hitl_showcase/hitl_feedback_demo.py --top-n 25

# Passive background feedback demo:
python demo/hitl_showcase/demo_passive_feedback.py
```

See [`demo/hitl_showcase/`](demo/hitl_showcase/) for full documentation.

## Authors
- Vince Clifford C. Aguilar
- Nhel Adam S. Benosa
- Elyssa Olivares

## License
MIT
