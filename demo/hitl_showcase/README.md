# HITL Feedback Showcase

Demonstrates the complete **Human-in-the-Loop (HITL) feedback workflow** using
real unseen sentences from the Rinconada Bikol corpus, ranked by model uncertainty.

## Strategy: Uncertainty Sampling

This demo runs the CRF model on **all ~18K sentences not in the annotated
corpus**, ranks them by mean token confidence (ascending), and exports the
**top-N most uncertain** sentences as a feedback CSV for expert review.

This mirrors real-world **active learning**: the sentences the model is *least*
confident about yield the most informative corrections per annotation effort.

## Quick Start

### 1. Maintainer Active Learning (Batch Uncertainty Sampling)
Run batch inference over all ~18K unseen sentences to rank and export the hardest samples:
```bash
python demo/hitl_showcase/hitl_feedback_demo.py --top-n 25
```
**Expected runtime:** ~10–15s (batch tagging ~18K sentences).

### 2. Passive Background Feedback (Zero-Friction for End Users)
Demonstrates how `rintag.tag()` silently buffers uncertain and OOV predictions in the background without requiring developers to call feedback functions explicitly:
```bash
python demo/hitl_showcase/demo_passive_feedback.py
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--top-n N` | `25` | Number of most uncertain sentences to export |
| `--threshold T` | `0.75` | Confidence threshold for flagging tokens |
| `--min-words N` | `3` | Minimum sentence length (words) |
| `--max-words N` | `30` | Maximum sentence length (words) |
Examples:

```bash
# Export top 50, stricter threshold
python demo/hitl_showcase/hitl_feedback_demo.py --top-n 50 --threshold 0.75

# Longer sentences only
python demo/hitl_showcase/hitl_feedback_demo.py --min-words 8 --max-words 20
```

## Output Files

Generated in `output/` on each run (gitignored):

| File | Description |
|---|---|
| `ranked_unseen.csv` | All unseen sentences ranked by mean confidence |
| `hitl_feedback_top{N}.csv` | Top-N export — blank `user-corrected_tag` for experts |
| `passive_feedback_captured.csv` | Passive library feedback from the demo sentences |
| `batch_audit.csv` | Passive feedback captured within the scoped batch example |

Passive feedback uses a `0.75` confidence threshold by default; set `threshold` to override it. Feedback exports contain only prediction and user-correction fields; validator annotations are handled separately by maintainers.

## Schema: `hitl_feedback_top{N}.csv`

The maintainer active-learning export matches the column naming convention used in the Google Sheets feedback database:

| Column | Description |
|---|---|
| `sentence_id` | Identifier grouping all tokens of one sentence |
| `timestamp` | ISO UTC timestamp when exported/reviewed |
| `full_sentence` | Complete sentence text |
| `token_index` | 0-based position of this token in the sentence |
| `token` | The word |
| `model_tag` | Model's predicted UPOS tag |
| `user-corrected_tag` | **Fill this in** with the correct tag |
| `confidence` | Model's confidence for this token (0–1) |
| `is_oov` | `True` if word not seen in training vocabulary |

## Full HITL Loop

```
Unseen sentences (raw corpus)
       ↓  batch CRF inference
Rank by mean confidence (lowest first)
       ↓  uncertainty sampling
Export top-N feedback CSV (Google Sheets schema)
       ↓  share with linguists / native speakers
Expert reviews → fills user-corrected_tag
       ↓  submit via Google Sheets, GitHub Issue, or CSV
Use reviewed corrections for manual corpus curation and analysis.
```
