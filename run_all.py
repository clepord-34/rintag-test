"""
run_all.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates the full RinTag pipeline from raw corpus to trained model and
evaluation results.

This script runs each selected stage in dependency order.

Usage:
    python run_all.py              # Run all stages
    python run_all.py --from 3     # Start from stage 3 onwards
    python run_all.py --only 5     # Run only stage 5

Stages:
    1. Sample corpus         (scripts/01_sample_corpus.py)
    2. Format for annotation (scripts/02_format_for_annotation.py)
   2a. Convert lexicon PTB→UD (scripts/02a_convert_lexicon_ptb_to_ud.py)
   2b. Pre-annotate          (scripts/02b_pre_annotate.py)
    3. Compute Kappa         (scripts/03_compute_kappa.py)
    4. Split corpus          (scripts/04_split_corpus.py)
    5. Train model           (scripts/05_train_model.py)
    6. Corpus EDA            (experiments/01_corpus_eda.py)
    7. Cross-validation      (experiments/02_cross_validation.py)
    8. Evaluation            (experiments/03_evaluation.py)
    9. Expert validation     (experiments/04_expert_validation.py)
   10. Feature analysis      (experiments/05_feature_analysis.py)
   11. Thesis tables         (experiments/06_thesis_tables.py)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── STAGE DEFINITIONS ─────────────────────────────────────────────────────────
# Each stage: (number, name, script_path, working_directory)

STAGES = [
    (1,    "Sample corpus",
        "scripts/01_sample_corpus.py",   "scripts"),

    (2,    "Format for annotation",
        "scripts/02_format_for_annotation.py", "scripts"),

    (2.1,  "Convert lexicon PTB → UD",
        "scripts/02a_convert_lexicon_ptb_to_ud.py", "scripts"),

    (2.2,  "Pre-annotate corpus",
        "scripts/02b_pre_annotate.py",   "scripts"),

    (3,    "Compute Kappa",
        "scripts/03_compute_kappa.py",   "scripts"),

    (4,    "Split corpus",
        "scripts/04_split_corpus.py",    "scripts"),

    (5,    "Train model",
        "scripts/05_train_model.py",     "scripts"),

    (6,    "Corpus EDA",
        "experiments/01_corpus_eda.py",  "experiments"),

    (7,    "Cross-validation",
        "experiments/02_cross_validation.py", "experiments"),

    (8,    "Evaluation",
        "experiments/03_evaluation.py",  "experiments"),

    (9,    "Expert validation",
        "experiments/04_expert_validation.py", "experiments"),

    (10,   "Feature analysis",
        "experiments/05_feature_analysis.py", "experiments"),

    (11,   "Thesis tables",
        "experiments/06_thesis_tables.py", "experiments"),
]


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RinTag pipeline.")
    parser.add_argument("--from", type=float, dest="from_stage", default=1,
                        help="Start from this stage number")
    parser.add_argument("--only", type=float, default=None,
                        help="Run only this stage number")
    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()

    print("=" * 60)
    print("RinTag Pipeline Orchestrator")
    print(f"Project root: {project_root}")
    print("=" * 60)

    for stage_num, name, script, cwd in STAGES:
        # Filter by --from and --only
        if args.only is not None and stage_num != args.only:
            continue
        if stage_num < args.from_stage:
            continue

        print(f"\n{'-' * 60}")
        print(f"Stage {stage_num}: {name}")
        print(f"{'-' * 60}")

        # Run the script
        script_path = project_root / script
        work_dir    = project_root / cwd

        if not script_path.exists():
            print(f"  ⚠ Script not found: {script}")
            print(f"    Expected at: {script_path}")
            continue

        print(f"  Running: python {script}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),
            capture_output=False,
        )

        if result.returncode != 0:
            print(f"  ✗ FAILED (exit code {result.returncode})")
            print(f"    Fix the error above before continuing.")
            sys.exit(result.returncode)
        else:
            print(f"  ✓ Completed successfully")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
