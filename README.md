# RinTag

A Python-based Part-of-Speech (POS) Tagger for the **Rinconada Bikol Language** (*Riŋkonāda* / Inland Bikol) using Conditional Random Fields (CRF).

[![CI](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rintag.svg)](https://pypi.org/project/rintag/)
[![Python](https://img.shields.io/pypi/pyversions/rintag.svg)](https://pypi.org/project/rintag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [About RinTag](#about-rintag)
- [Empirical Benchmarks & Thesis Results](#empirical-benchmarks--thesis-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation Index](#documentation-index)
- [Repository Structure](#repository-structure)
- [Experimental Pipeline](#experimental-pipeline)
- [Human-in-the-Loop (HITL) Workflow](#human-in-the-loop-hitl-workflow)
- [Interactive Web Demo](#interactive-web-demo)
- [Development & Testing](#development--testing)
- [Contributing Feedback](#contributing-feedback)
- [Authors & Citation](#authors--citation)
- [License](#license)

---

## About RinTag

**Rinconada Bikol** (*ISO 639-3: `bto`*) is an Austronesian language spoken by approximately 600,000 people primarily in the Rinconada district of Camarines Sur, Philippines. Despite its vibrant spoken presence, it remains an under-resourced language in computational linguistics and NLP.

**RinTag** is designed to address this digital underrepresentation by providing:
1. **A lightweight, production-ready Python library (`rintag`)** powered by a linear-chain CRF model with bundled weights and zero heavy runtime dependencies.
2. **A gold-standard annotated corpus** of 2,000 sentences (31,035 tokens) annotated under the **Universal Dependencies (UD)** v2 standard across 15 POS categories with high inter-annotator agreement.
3. **An active Human-in-the-Loop (HITL) retraining architecture** with strict data leakage safeguards, deterministic version archiving, and paired comparative evaluation.

---

## Empirical Benchmarks & Thesis Results

Evaluated on the held-out test partition (200 sentences, 3,183 tokens):

| Metric | Baseline ($v1$) | HITL ($v2$) | HITL ($v3$) |
|---|---|---|---|
| **Overall Accuracy** | **83.00%** | **83.32%** | **83.66%** |
| **Macro $F_1$ Score** | **0.8193** | **0.8216** | **0.8242** |
| **Weighted $F_1$ Score** | **0.8288** | **0.8320** | **0.8354** |
| **Out-of-Vocabulary (OOV) Accuracy** | **69.23%** (495/715) | **69.96%** (496/709) | **71.61%** (502/701) |
| **Mean Model Confidence** | 92.46% | 92.52% | 92.54% |
| **Inter-Annotator Agreement ($\kappa$)** | **0.9725** (*Almost Perfect*) | — | — |

*Full evaluation reports and confusion matrices are generated in [`experiments/results/thesis_tables/`](experiments/results/thesis_tables/).*

---

## Installation

### From PyPI (Standard User)

```bash
pip install rintag
```

### From Source (Development & Research)

```bash
git clone https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>.git
cd RinTag
pip install -e ".[dev]"
```

---

## Quick Start

```python
import rintag

# Tag a Rinconada Bikol sentence
result = rintag.tag("nag-dalagan su igin")

# 1. Iterate like (token, tag) pairs
for token, tag in result:
    print(f"{token:15} -> {tag}")
# nag-dalagan     -> VERB
# su              -> DET
# igin            -> NOUN

# 2. Bulk properties
print("Tokens:    ", result.tokens)      # ['nag-dalagan', 'su', 'igin']
print("Tags:      ", result.tags)        # ['VERB', 'DET', 'NOUN']
print("Confidence:", result.confidence)  # [0.9834, 0.9521, 0.9912]
print("OOV Words: ", result.oov_tokens)  # ['nag-dalagan']

# 3. Rich token metadata
first = result[0]
print(f"Token: {first.token}, Tag: {first.tag}, Confidence: {first.confidence:.2%}, OOV: {first.is_oov}")
print("Top Alternative Tags:", first.alternatives[:2])
```

---

## Documentation Index

Detailed guides are located in the [`docs/`](docs/) directory:

- [**docs/pypi/README.md**](docs/pypi/README.md) — PyPI package description and end-user reference.
- [**docs/architecture.md**](docs/architecture.md) — System architecture, tokenizer design, feature engineering groups, and model registry contracts.
- [**docs/pipeline.md**](docs/pipeline.md) — End-to-end replication guide for Stages 1 through 11 via `run_all.py`.
- [**docs/hitl_workflow.md**](docs/hitl_workflow.md) — Ingestion, leakage prevention, label policies, and version promotion.
- [**docs/demo_and_deployment.md**](docs/demo_and_deployment.md) — Web application instructions, Google Sheets integration, and Render deployment.
- [**demo/hitl_showcase/README.md**](demo/hitl_showcase/README.md) — Maintainer active learning showcase using uncertainty sampling.

---

## Repository Structure

```
RinTag/
├── .github/                      # GitHub Actions CI and issue templates
│   ├── workflows/                # CI test, TestPyPI, and PyPI publish workflows
│   └── ISSUE_TEMPLATE/           # Community tag correction submission template
├── data/                         # Data partitions and experimental corpora
│   ├── raw/                      # Unfiltered raw corpus (combined_corpus.csv)
│   ├── interim/                  # Sampled sentences & rule-pre-annotated corpus
│   ├── processed/                # Gold standard splits (train.csv, dev.csv, test.csv)
│   ├── validation/               # Inter-annotator validation annotations
│   └── feedback/                 # Validated feedback snapshots and training sets
├── demo/                         # Web demo application and uncertainty showcases
│   ├── app.py                    # Flask web application
│   ├── templates/ & static/      # Web UI assets
│   └── hitl_showcase/            # Uncertainty sampling batch inference showcase
├── docs/                         # Technical documentation & PyPI distribution README
│   ├── pypi/README.md            # Dedicated PyPI README
│   ├── architecture.md           # Architecture & feature specification
│   ├── pipeline.md               # Pipeline replication guide
│   ├── hitl_workflow.md          # HITL retraining & leakage prevention
│   └── demo_and_deployment.md    # Demo & Render deployment guide
├── experiments/                  # Thesis experiment scripts & generated tables
│   ├── 01_corpus_eda.py          # Exploratory data analysis
│   ├── 02_cross_validation.py    # 5-fold cross-validation
│   ├── 03_evaluation.py          # Held-out test set evaluation
│   ├── 04_expert_validation.py   # Kappa agreement breakdown
│   ├── 05_feature_analysis.py    # Learned feature weight extraction
│   ├── 06_thesis_tables.py       # Formatted thesis tables (Tables 5–14)
│   └── results/                  # Thesis tables, figures, and metrics
├── models/                       # Versioned model archives and metadata
│   ├── model_registry.json       # Version registry with hashes and evaluation metrics
│   ├── crf_model_v*.pkl          # Archived CRF models
│   └── vocab_v*.pkl              # Archived vocabularies
├── scripts/                      # Corpus preparation & training pipeline
│   ├── 01_sample_corpus.py       # Stratified greedy vocabulary sampling
│   ├── 02_format_for_annotation.py
│   ├── 02a_convert_lexicon_ptb_to_ud.py
│   ├── 02b_pre_annotate.py
│   ├── 03_compute_kappa.py       # Cohen's Kappa calculation
│   ├── 04_split_corpus.py        # 80/10/10 corpus splitting
│   ├── 05_train_model.py         # CRF training & hyperparameter search
│   ├── 06_hitl.py                # HITL CLI orchestrator
│   └── hitl/                     # Ingest, audit, retrain, evaluate, compare, promote
├── src/                          # Main Python package source
│   └── rintag/
│       ├── __init__.py           # Public API exports
│       ├── tagger.py             # Main inference tagger
│       ├── tokenizer.py          # Rinconada Bikol tokenizer
│       ├── features.py           # CRF surface feature extractors
│       ├── feedback.py           # Passive feedback collection engine
│       └── models/               # Active bundled model and vocabulary
├── tests/                        # Automated unit and integration test suite (65 tests)
├── .env.example                  # Environment template for demo feedback storage
├── LICENSE                       # MIT License
├── MANIFEST.in                   # Packaging source manifest
├── pyproject.toml                # Build system & package metadata
├── render.yaml                   # Render.com deployment blueprint
└── run_all.py                    # Master pipeline orchestrator (Stages 1–11)
```

---

## Experimental Pipeline

To reproduce all experiments from raw data to trained models and thesis tables:

```bash
# Run all stages sequentially
python run_all.py

# Or run specific stages:
python run_all.py --from 5   # From model training onwards
python run_all.py --only 11  # Re-generate thesis tables only
```

See [`docs/pipeline.md`](docs/pipeline.md) for stage parameters and inputs.

---

## Human-in-the-Loop (HITL) Workflow

RinTag features an auditable HITL loop:

1. **Capture feedback**: Passively via library usage or actively via uncertainty sampling.
2. **Ingest & Audit**: Automatically checks feedback against `train`, `dev`, and `test` to guarantee zero data leakage.
3. **Retrain & Compare**:
   ```bash
   python scripts/06_hitl.py ingest
   python scripts/06_hitl.py audit
   python scripts/06_hitl.py retrain --label-policy validator
   python scripts/06_hitl.py compare
   python scripts/06_hitl.py promote --version N
   ```

See [`docs/hitl_workflow.md`](docs/hitl_workflow.md) for full commands and policy guidelines.

---

## Interactive Web Demo

An interactive Flask interface is available in `demo/`:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```
Open `http://localhost:5000` to interactively tag sentences and view feature breakdowns. For deployment on Render, see [`docs/demo_and_deployment.md`](docs/demo_and_deployment.md).

---

## Development & Testing

Run the automated test suite with pytest:

```bash
pytest tests/ -v
```

All 65 unit and integration tests cover tokenization, feature extraction, ablation configurations, passive feedback buffering, leakage audits, registry updates, and model promotion safeguards.

---

## Contributing Feedback

We welcome native Rinconada Bikol speakers, linguists, and NLP researchers to contribute corrections:
1. Export a feedback CSV using `rintag.export_passive_feedback("rintag_feedback.csv")`.
2. Review the rows and fill in corrected UPOS tags in `user-corrected_tag`.
3. Open a [Tag Correction Feedback Issue](https://github.com/<GITHUB_OWNER>/<RINTAG_REPO>/issues/new?template=tag-correction.md) and attach the CSV.

---

## Authors & Citation

- **Vince Clifford C. Aguilar**
- **Nhel Adam S. Benosa**
- **Elyssa Olivares**

*Department of Computer Studies, Camarines Sur Polytechnic Colleges (CSPC), Nabua, Camarines Sur, Philippines.*

```bibtex
@thesis{aguilar2026rintag,
  author       = {Aguilar, Vince Clifford C. and Benosa, Nhel Adam S. and Olivares, Elyssa},
  title        = {RinTag: A Part-of-Speech Tagger for the Rinconada Bikol Language using Conditional Random Fields},
  school       = {Camarines Sur Polytechnic Colleges},
  year         = {2026},
  address      = {Nabua, Camarines Sur, Philippines}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
