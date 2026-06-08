"""
eval_gate.py — Quality gate: pass/fail based on eval_report.json thresholds.

Usage:
    python -m rfpose.evaluation.eval_gate --report eval_report.json
    python -m rfpose.evaluation.eval_gate --report eval_report.json --max-mpjpe 0.10
"""
from __future__ import annotations

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="Quality gate for model evaluation")
    ap.add_argument("--report", required=True, help="Path to eval_report.json")
    ap.add_argument("--min-macro-f1", type=float, default=0.50)
    ap.add_argument("--max-latency-p95-ms", type=float, default=500.0)
    ap.add_argument("--max-mpjpe", type=float, default=None,
                    help="Max MPJPE threshold (skip if not set)")
    ap.add_argument("--max-pa-mpjpe", type=float, default=None,
                    help="Max PA-MPJPE threshold (skip if not set)")
    args = ap.parse_args()

    r = json.load(open(args.report))

    checks = {
        "latency_p95": r.get("latency_ms_p95_batch", 1e9) <= args.max_latency_p95_ms,
    }

    if "macro_f1" in r:
        checks["macro_f1"] = r["macro_f1"] >= args.min_macro_f1

    if args.max_mpjpe is not None and "mpjpe" in r:
        checks["mpjpe"] = r["mpjpe"] <= args.max_mpjpe

    if args.max_pa_mpjpe is not None and "pa_mpjpe" in r:
        checks["pa_mpjpe"] = r["pa_mpjpe"] <= args.max_pa_mpjpe

    passed = all(checks.values())
    result = {
        "passed": passed,
        "checks": checks,
        "thresholds": {k: v for k, v in vars(args).items() if k != "report"},
        "values": {
            "mpjpe": r.get("mpjpe"),
            "pa_mpjpe": r.get("pa_mpjpe"),
            "macro_f1": r.get("macro_f1"),
            "latency_p95": r.get("latency_ms_p95_batch"),
            "accuracy": r.get("accuracy"),
        },
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
