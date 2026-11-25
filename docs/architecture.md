# RinTag Architecture & Codebase Design

This document details the internal architecture, feature engineering pipeline, model storage contracts, and data partitioning of RinTag.

---

## 1. System Architecture

RinTag is designed with a strict separation between:
1. **Public Library Runtime (`src/rintag`)**: Distributed via PyPI. Lightweight, zero dependencies beyond `sklearn-crfsuite` and `scikit-learn`, bundling pre-trained weights and training vocabulary.
2. **Offline Pipeline & Experiments (`scripts/` & `experiments/`)**: Corpus sampling, PTB-to-UD mapping, inter-annotator agreement calculations, grid search cross-validation, feature ablation, and thesis report generation.
3. **Human-in-the-Loop (HITL) Subsystem (`scripts/hitl/` & `data/feedback/`)**: Feedback ingestion, validation, data leakage prevention, candidate retraining, version comparison, and guarded promotion.
4. **Interactive Demonstration (`demo/`)**: Deployable web application and uncertainty sampling showcases.

```
┌────────────────────────────────────────────────────────┐
│                   RinTag Ecosystem                     │
├──────────────────────────┬─────────────────────────────┤
│   Public Library (PyPI)  │     Development & Thesis    │
│  src/rintag/             │  scripts/ & experiments/    │
│  ├── tagger.py           │  ├── run_all.py             │
│  ├── tokenizer.py        │  ├── scripts/01..05         │
│  ├── features.py         │  └── experiments/01..06     │
│  ├── feedback.py         │                             │
│  └── models/ (bundled)   │  HITL Retraining Engine     │
│      ├── crf_model.pkl   │  scripts/hitl/              │
│      └── vocab.pkl       │  ├── ingest.py / retrain.py │
│                          │  ├── leakage.py / compare.py│
│                          │  └── promote.py / status.py │
└──────────────────────────┴─────────────────────────────┘
```

---

## 2. Package Internals (`src/rintag/`)

### `tokenizer.py`
Rule-based sentence tokenizer adhering to Rinconada Bikol orthographic conventions:
- **Whitespace normalization**: Splitting on whitespace sequences.
- **Punctuation isolation**: Detaches leading and trailing punctuation as individual `PUNCT` tokens.
- **Hyphen & Apostrophe Preservation**: Maintains internal hyphens (e.g., `nag-dalagan`, `nag-ukrong`) and internal apostrophes (e.g., `pa'la`, `'ya`) intact to preserve affixation markers.
- **Number Handling**: Thousand-separated numbers (e.g., `411,418`) remain unified numeric tokens rather than being split on commas.

### `features.py`
Surface-only feature extraction for sequence labeling using Conditional Random Fields. All features are derived strictly from token character sequences without external dictionaries:
- **Base Features**: Word identity, lowercased form, 2–4 character prefixes and suffixes, title/digit/hyphen/punctuation flags, and boundary indicators (`BOS`/`EOS`).
- **Extended Feature Groups** (configured via `DEFAULT_GROUPS`):
  - `window2`: $\pm 2$ contextual token window.
  - `bigrams`: Contextual string conjunctions (e.g., `prev_word|word`, `word|next_word`).
  - `affix_ext`: 1-character and 5-character affixes.
  - `shape`: Collapsed character shape patterns (e.g., `Xx`, `x-x`).
  - `repeat`: Orthographic repetition detection (character runs like `ayyy`, repeated units like `hahaha`).
  - `neighbor_punct`: Boundary flags indicating if adjacent tokens are punctuation.

### `tagger.py`
The primary inference interface:
- **Model Cache**: Lazily deserializes `src/rintag/models/crf_model.pkl` and `vocab.pkl`.
- **`tag(text: str) -> TagResult`**: Runs tokenization, extracts features, queries the CRF model for Viterbi sequence predictions and marginal probabilities, flags OOV tokens against the vocabulary set, and checks active passive feedback collectors.
- **`TagResult` & `TaggedToken`**: Immutable, rich output containers supporting both backward-compatible tuple iteration `(token, tag)` and structured inspection (`confidence`, `is_oov`, `alternatives`, `features`).

### `feedback.py`
Client-side feedback collection utilities:
- **`FeedbackCollector`**: Thread-safe in-memory buffer capturing sentences with tokens below confidence cutoff (default `0.75`) or unseen OOV words.
- **Convenience Functions**: `enable_passive_feedback()`, `disable_passive_feedback()`, `export_passive_feedback(csv_path)`, and `collect_feedback(context_manager)`.

---

## 3. Model Storage & Registry Contracts

RinTag maintains two distinct model locations:

| Directory | Role | Description |
|---|---|---|
| `src/rintag/models/` | **Active Bundled Model** | The live model files (`crf_model.pkl`, `vocab.pkl`) shipped inside the Python wheel. |
| `models/` | **Versioned Model Archive** | Immutable version archives (`crf_model_v1.pkl`, `crf_model_v2.pkl`, etc.) tracked by `models/model_registry.json`. |

When a new candidate model is trained using `scripts/06_hitl.py retrain`, it is created in `models/` as an archived version. It is **not** activated in `src/rintag/models/` until explicitly promoted via `scripts/06_hitl.py promote --version N`.

---

## 4. Corpus Partitions & Data Directory Layout

The repository organizes data according to pipeline stages:

- `data/raw/`: Original uncleaned text corpus (`combined_corpus.csv`).
- `data/lexicon/`: Reference lexicons (`lexicon.csv`, `lexicon_ud.csv`).
- `data/interim/`: Sampled sentences (`sampled_sentences.csv`) and rule-based pre-annotated data (`pre_annotated_corpus.csv`).
- `data/validation/`: Expert validation split and independent linguist annotations (`validator_annotations.csv`, `sentence_id_reference.csv`).
- `data/processed/`: Gold standard corpus splits:
  - `annotated_corpus.csv`: Full 2,000-sentence gold standard corpus.
  - `train.csv`: 1,600 training sentences (24,862 tokens).
  - `dev.csv`: 200 validation sentences (2,990 tokens).
  - `test.csv`: 200 held-out evaluation sentences (3,183 tokens).
- `data/feedback/`: Validated feedback snapshots and ingested training sets used for HITL retraining.
