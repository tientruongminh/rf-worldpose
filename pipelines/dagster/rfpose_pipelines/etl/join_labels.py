from __future__ import annotations
import json
from pathlib import Path

def load_events(path: str|Path):
    p=Path(path)
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def label_for_timestamp(events, ts_us:int, default='unlabeled'):
    ts=ts_us/1_000_000
    active=default
    for e in events:
        if e.get('t',0) <= ts and e.get('event')=='start': active=e.get('label',active)
        if e.get('t',0) <= ts and e.get('event')=='end' and e.get('label')==active: active=default
    return active
