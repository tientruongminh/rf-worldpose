from __future__ import annotations
import argparse, json, sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--min-macro-f1', type=float, default=0.50)
    ap.add_argument('--max-latency-p95-ms', type=float, default=500.0)
    args=ap.parse_args()
    r=json.load(open(args.report))
    checks={
        'macro_f1': r.get('macro_f1',0) >= args.min_macro_f1,
        'latency_p95': r.get('latency_ms_p95_batch',1e9) <= args.max_latency_p95_ms,
    }
    result={'passed': all(checks.values()), 'checks': checks, 'thresholds': vars(args)}
    print(json.dumps(result,indent=2))
    sys.exit(0 if result['passed'] else 2)
if __name__=='__main__': main()
