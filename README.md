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

# Get detailed info for a specific token
first = result[0]
print(first.token)         # 'nag-dalagan'
print(first.tag)           # 'VERB'
print(first.confidence)    # 0.9834
print(first.alternatives[:3]) # Top 3 alternative tags
print(first.features)      # CRF features used for this token
```

## Authors
- Vince Clifford C. Aguilar
- Nhel Adam S. Benosa
- Elyssa Olivares

## License
MIT
