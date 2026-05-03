#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, urllib.request

def post_json(url: str, payload: dict):
    data=json.dumps(payload).encode()
    req=urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req) as r:
        print(r.read().decode())

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--api', default='http://localhost:8080')
    ap.add_argument('--dataset-version', required=True)
    ap.add_argument('--artifact-uri', required=True)
    ap.add_argument('--preprocess-version', default='rfpose-preprocess-v0.1')
    ap.add_argument('--stats-json', default=None)
    args=ap.parse_args()
    stats=json.load(open(args.stats_json)) if args.stats_json else {}
    post_json(args.api.rstrip('/') + '/api/v1/datasets', {'id': args.dataset_version, 'artifact_uri': args.artifact_uri, 'preprocess_version': args.preprocess_version, 'stats': stats})
