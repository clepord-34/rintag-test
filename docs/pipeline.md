# RinTag Experimental Pipeline & Replication Guide

This guide details how to reproduce the entire RinTag model training, cross-validation, evaluation, and thesis table generation pipeline from scratch.

---

## 1. Orchestrated Execution (`run_all.py`)

The root script `run_all.py` orchestrates the complete experimental pipeline in dependency order:

```bash
# Execute the entire end-to-end pipeline (Stages 1 through 11)
python run_all.py

# Run starting from a specific stage (e.g., from model training onwards)
python run_all.py --from 5

# Execute only a specific stage
python run_all.py --only 8
```

---

## 2. Pipeline Stages Breakdown

| Stage | Script | Purpose | Key Inputs | Primary Outputs |
|---|---|---|---|---|
| **1** | `scripts/01_sample_corpus.py` | Stratified greedy vocabulary sampling (2,000 sentences across 4 length strata) | `data/raw/combined_corpus.csv` | `data/interim/sampled_sentences.csv` |
| **2** | `scripts/02_format_for_annotation.py` | Converts raw sampled sentences into tokenized tabular format for annotation | `data/interim/sampled_sentences.csv` | `data/interim/formatted_sampled_corpus.csv` |
| **2a** | `scripts/02a_convert_lexicon_ptb_to_ud.py` | Maps Penn Treebank (PTB) tags in the Nabua lexicon to Universal Dependencies (UD) | `data/lexicon/lexicon.csv` | `data/lexicon/lexicon_ud.csv` |
| **2b** | `scripts/02b_pre_annotate.py` | Applies lexicon-based rule pre-tagging to bootstrap manual annotation | `data/lexicon/lexicon_ud.csv` | `data/interim/pre_annotated_corpus.csv` |
| **3** | `scripts/03_compute_kappa.py` | Computes inter-annotator Cohen's Kappa ($\kappa$) between researchers and language expert | `data/validation/validator_annotations.csv` | Console summary & validation metrics |
| **4** | `scripts/04_split_corpus.py` | Partitions gold corpus into Train (1,600), Dev (200), and Test (200) | `data/processed/annotated_corpus.csv` | `data/processed/{train,dev,test}.csv` |
| **5** | `scripts/05_train_model.py` | 5-fold grid search ($c_1, c_2$), trains final CRF model, exports vocabulary and registry entry | `data/processed/train.csv` | `src/rintag/models/crf_model.pkl`, `vocab.pkl`, `models/model_registry.json` |
| **6** | `experiments/01_corpus_eda.py` | Exploratory data analysis of token lengths, vocabulary size, and POS distributions | `data/processed/annotated_corpus.csv` | Summary stats |
| **7** | `experiments/02_cross_validation.py` | Cached 5-fold cross-validation grid search evaluation across parameter space | `data/processed/train.csv` | `experiments/results/cv_results.csv` |
| **8** | `experiments/03_evaluation.py` | Held-out test set evaluation: classification report, OOV accuracy, and confusion matrix | `data/processed/test.csv` | `experiments/results/classification_report.txt`, `confusion_matrix.png` |
| **9** | `experiments/04_expert_validation.py` | Detailed per-tag agreement breakdown with language expert | `data/validation/validator_annotations.csv` | `experiments/results/expert_agreement.csv` |
| **10** | `experiments/05_feature_analysis.py` | Extracts top positive and negative learned CRF transition and state feature weights | Trained CRF model | `experiments/results/top_features_per_tag.csv`, `feature_weights_summary.txt` |
| **11** | `experiments/06_thesis_tables.py` | Consolidates all statistics into thesis-ready Tables 5 through 14 (CSV + formatted text) | All processed splits & results | `experiments/results/thesis_tables/` |

---

## 3. Thesis Results Reference Summary

The table generation script (`experiments/06_thesis_tables.py`) produces the following verified thesis deliverables in `experiments/results/thesis_tables/`:

- **Table 5 (`table5_corpus_composition`)**: Total corpus composition (2,000 sentences, 31,035 tokens, 9,036 vocabulary).
- **Table 6 (`table6_length_distribution`)**: Sentence length distributions across strata.
- **Table 7 (`table7_tag_distribution`)**: UPOS frequency distribution across the corpus.
- **Table 8 (`table8_expert_agreement`)**: Inter-annotator agreement per POS tag ($\kappa = 0.9725$, Almost Perfect Agreement).
- **Table 9 (`table9_grid_search`)**: Hyperparameter grid search ($c_1=0.01, c_2=0.01 \to \text{Macro } F_1 = 0.8207 \pm 0.0076$).
- **Table 10 (`table10_classification_report`)**: Per-class precision, recall, and $F_1$ scores on the 200 held-out test sentences.
- **Table 11 (`table11_overall_performance`)**: Overall model performance (Accuracy: 0.8300, Macro $F_1$: 0.8193, Weighted $F_1$: 0.8288).
- **Table 12 (`table12_oov_breakdown`)**: Out-of-vocabulary evaluation (69.23% overall OOV accuracy across 715 unseen tokens).
- **Table 13 (`table13_pre_annotator`)**: Error reduction comparison from pre-annotator baseline to researcher-validated corpus.
- **Table 14 (`top_features_per_tag.csv`)**: Most influential morphological affixes and context bigrams per UPOS class.
