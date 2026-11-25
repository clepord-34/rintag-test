#!/usr/bin/env python
"""RinTag HITL retraining tool. Usage: python scripts/06_hitl.py <command> --help"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hitl.common import BOOTSTRAP_RESAMPLES, LABEL_POLICIES, HitlError, Paths


def _ingest_args(p):
    p.add_argument("--input", action="append", default=None, metavar="CSV", help="Feedback CSV (repeatable). Default: prototype_feedback.csv and library_feedback.csv at the repo root.")
    p.add_argument("--label-policy", choices=LABEL_POLICIES, default="validator", help="Training labels. validator is thesis-grade; user policies are experimental.")


def build_parser():
    ap = argparse.ArgumentParser(description="RinTag HITL retraining tool")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="Re-score an archived version and compare it with the registry.")
    p.add_argument("--version", type=int, required=True)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("evaluate", help="Score a version on an eval set.")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--evalset", choices=["test", "dev", "hitl_holdout", "hitl_holdout_user", "all"], default="all")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("status", help="List versions and feedback counts.")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("ingest", help="Validate feedback, reject leakage, and assign frozen splits.")
    _ingest_args(p)
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("audit", help="Independently audit feedback leakage.")
    p.add_argument("--version", type=int, default=None)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("retrain", help="Train and archive a candidate version.")
    _ingest_args(p)
    p.add_argument("--parent", type=int, default=None)
    p.add_argument("--c1", type=float, default=None)
    p.add_argument("--c2", type=float, default=None)
    p.add_argument("--min-sentences", type=int, default=1)
    p.set_defaults(func=cmd_retrain)
    p = sub.add_parser("compare", help="Write comparison tables and plots.")
    p.add_argument("--versions", type=int, nargs="*", default=None)
    p.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("promote", help="Promote or roll back the bundled model.")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(func=cmd_promote)
    return ap


def cmd_verify(paths, args):
    from hitl.evaluate import verify_version
    problems = verify_version(paths, args.version)
    if problems:
        print(f"VERIFY v{args.version}: FAILED")
        for line in problems:
            print("  - " + line)
        return 1
    print(f"VERIFY v{args.version}: OK (registry numbers and artifact hashes reproduced)")
    return 0


def cmd_evaluate(paths, args):
    from hitl.evaluate import evaluate_version
    names = ["test", "dev", "hitl_holdout", "hitl_holdout_user"] if args.evalset == "all" else [args.evalset]
    for name in names:
        metrics = evaluate_version(paths, args.version, name, force=args.force)
        if metrics is None:
            print(f"v{args.version} {name}: eval set is empty, skipped")
        else:
            print(f"v{args.version} {name}: n={metrics['n_tokens']} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
    return 0


def cmd_status(paths, args):
    from hitl.registry import load_registry
    from hitl.common import read_json
    registry = load_registry(paths)
    print(f"active version: v{registry['active_version']}")
    for entry in registry["versions"]:
        mark = "*" if entry["version"] == registry["active_version"] else " "
        print(f"{mark} v{entry['version']}  parent={entry.get('parent_version')}  policy={entry.get('label_policy', '-')}  experimental={entry.get('experimental', False)}  test_acc={entry['test_accuracy']}  test_macro_f1={entry['test_macro_f1']}")
    report_path = paths.feedback_validated / "ingest_report.json"
    if report_path.exists():
        report = read_json(report_path)
        print(f"last ingest (policy={report['label_policy']}): ok={report['n_ok']} validated={report['n_validated']} trainable={report['n_trainable']} (train={report['n_train']}, holdout={report['n_holdout']}) rejected={report['n_rejected']}")
    return 0


def cmd_ingest(paths, args):
    from hitl.ingest import run_ingest
    report = run_ingest(paths, args.input, args.label_policy)
    print(f"ingest: candidates={report['n_candidates']} ok={report['n_ok']} validated={report['n_validated']} trainable={report['n_trainable']} (train={report['n_train']}, holdout={report['n_holdout']}) rejected={report['n_rejected']}")
    return 0


def cmd_audit(paths, args):
    from hitl.leakage import format_audit, leakage_audit
    snapshot = paths.snapshot(args.version) / "feedback_sentences.csv" if args.version else None
    audit = leakage_audit(paths, snapshot)
    print(format_audit(audit))
    return 0 if audit["ok"] else 1


def cmd_retrain(paths, args):
    from hitl.retrain import retrain
    if args.label_policy != "validator":
        print(f"NOTE: label policy '{args.label_policy}' trains without validator-only labels. The new version will be marked EXPERIMENTAL and cannot be promoted without --force.")
    version = retrain(paths, args.input, args.label_policy, args.parent, args.c1, args.c2, args.min_sentences)
    print(f"retrain: created candidate v{version}. It is NOT active. Next: compare, then promote.")
    return 0


def cmd_compare(paths, args):
    from hitl.compare import compare
    meta = compare(paths, args.versions, args.resamples)
    print(f"compare: wrote results to {paths.compare_out} for versions {meta['versions']}")
    return 0


def cmd_promote(paths, args):
    from hitl.promote import promote
    promote(paths, args.version, args.yes, args.force, args.skip_tests)
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(Paths.default(), args) or 0)
    except HitlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())