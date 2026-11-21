#!/usr/bin/env python3
"""Passive Background Feedback Demo for RinTag.

Demonstrates how RinTag passively and silently collects uncertain predictions
and out-of-vocabulary (OOV) tokens in the background during normal rintag.tag()
calls — without requiring developers to write intentional feedback code.
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import rintag

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 68)
    print("  RinTag Passive Background Feedback Showcase")
    print("=" * 68)

    # ──────────────────────────────────────────────────────────────────
    # Scenario: A developer is building a Rinconada Bikol text app.
    # They just want to tag sentences without worrying about annotating.
    # ──────────────────────────────────────────────────────────────────

    sample_sentences = [
        # Familiar, high-confidence sentences from standard speech
        ("Nagdalagan su igin sa tinampo.", "Familiar / Standard"),
        ("Marhay na aga saindo gabos.",    "Common greeting"),
        ("Su lalaki nagbasa libro.",       "Standard syntax"),

        # Difficult, low-confidence, or dialectal/OOV sentences
        ("wagas na sasalan",               "Uncertain vocab / idioms"),
        ("sare n upod ah ngkasulo",        "Colloquial / OOV tokens"),
        ("pampa laki ing kayo nang bayarin", "Low-confidence particles"),
    ]

    print("\n[Step 1] Normal usage with passive feedback ENABLED in background")
    print("------------------------------------------------------------------")
    print("Configuring passive logging (e.g. at app startup or via env var):")
    print("  rintag.enable_passive_feedback(threshold=0.75, log_oov=True)\n")

    # Turn on passive background logging
    rintag.enable_passive_feedback(threshold=0.75, log_oov=True)

    print("Running rintag.tag() in standard developer workflow:")
    for text, label in sample_sentences:
        # Standard library usage — developer does NOT call any feedback functions!
        result = rintag.tag(text)

        mean_conf = sum(t.confidence for t in result) / len(result)
        oov_words = [t.token for t in result if t.is_oov]

        print(f"  Tagged: {text:<35} -> mean conf: {mean_conf:.1%}", end="")
        if oov_words:
            print(f" [OOV: {', '.join(oov_words)}]")
        else:
            print()

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Inspect what was passively captured
    # ──────────────────────────────────────────────────────────────────
    buffered = rintag.get_passive_feedback()
    unique_sids = set(r["sentence_id"] for r in buffered)

    print("\n[Step 2] What was captured silently in the background?")
    print("------------------------------------------------------------------")
    print(f"  Total sentences tagged           : {len(sample_sentences)}")
    print(f"  Sentences captured for review    : {len(unique_sids)}")
    print(f"  Total token rows in buffer       : {len(buffered)}")

    print("\n  High-confidence sentences were IGNORED automatically.")
    print("  Only uncertain/OOV sentences were preserved:")
    
    seen_sids = set()
    for row in buffered:
        sid = row["sentence_id"]
        if sid not in seen_sids:
            seen_sids.add(sid)
            print(f"    • [ID {sid}] {row['full_sentence']}")

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Export the passive buffer for linguist review
    # ──────────────────────────────────────────────────────────────────
    export_path = OUTPUT_DIR / "passive_feedback_captured.csv"
    rintag.export_passive_feedback(export_path)

    print("\n[Step 3] Exporting to standardized CSV")
    print("------------------------------------------------------------------")
    print(f"  Exported -> {export_path.relative_to(PROJECT_ROOT)}")
    print(f"  Schema: sentence_id, timestamp, full_sentence, token_index, token,")
    print(f"          model_tag, top_alternative, user-corrected_tag, confidence, is_oov")
    print("  (Ready for review and retraining)")

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Context manager usage (alternative for batch jobs)
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 4] Alternative: Scoped Context Manager for batch pipelines")
    print("------------------------------------------------------------------")
    context_csv = OUTPUT_DIR / "batch_audit.csv"
    print("  with rintag.collect_feedback('batch_audit.csv', threshold=0.80):")
    print("      for doc in documents:")
    print("          rintag.tag(doc)")

    batch_docs = [
        "lyo baga totoo yan",
        "ibani na 'ya bibi",
    ]
    with rintag.collect_feedback(output_path=context_csv, threshold=0.80) as collector:
        for doc in batch_docs:
            rintag.tag(doc)

    print(f"\n  Batch audit completed: captured {collector.sentence_count} sentences")
    print(f"  Written directly to -> {context_csv.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 68)
    print("  Summary: Zero developer friction.")
    print("  Library users simply use rintag.tag().")
    print("  Uncertain predictions are silently collected for HITL retraining.")
    print("=" * 68)


if __name__ == "__main__":
    main()
