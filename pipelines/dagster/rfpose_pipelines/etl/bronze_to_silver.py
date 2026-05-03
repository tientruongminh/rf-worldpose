from __future__ import annotations
from pathlib import Path
import json
import math
from typing import Iterable

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None


def iter_bronze_batches(bronze_root: str | Path) -> Iterable[Path]:
    root = Path(bronze_root)
    yield from sorted(root.rglob('*.json'))


def decode_packet_record(packet: dict, deployment_id: str, source_file: str) -> dict:
    pkt = json.loads(packet['packet_json']) if isinstance(packet.get('packet_json'), str) else packet
    amp = pkt.get('amplitude') or []
    return {
        'deployment_id': deployment_id,
        'source_file': source_file,
        'buffer_id': packet.get('id'),
        'received_at_ms': packet.get('received_at_ms'),
        'node_id': int(pkt.get('node_id', packet.get('node_id', 0))),
        'seq': int(pkt.get('seq', packet.get('seq', 0))),
        'timestamp_us': int(pkt.get('timestamp_us', packet.get('timestamp_us', 0))),
        'rssi': int(pkt.get('rssi', 0)),
        'noise_floor': int(pkt.get('noise_floor', 0)),
        'channel': int(pkt.get('channel', 0)),
        'n_subcarriers': int(pkt.get('n_subcarriers', len(amp))),
        'firmware_version': int(pkt.get('firmware_version', 0)),
        'amplitude': amp,
        'crc32': int(pkt.get('crc32', 0)),
    }


def bronze_to_silver(bronze_root: str | Path, silver_out: str | Path) -> dict:
    rows: list[dict] = []
    for file in iter_bronze_batches(bronze_root):
        obj = json.loads(file.read_text())
        deployment_id = obj.get('deployment_id', 'unknown')
        for packet in obj.get('packets', []):
            rows.append(decode_packet_record(packet, deployment_id, str(file)))
    out = Path(silver_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if pl is not None:
        pl.DataFrame(rows).write_parquet(out)
    else:
        out.write_text('\n'.join(json.dumps(r) for r in rows))
    node_ids = sorted({r['node_id'] for r in rows})
    seq_drops = 0
    for node in node_ids:
        seqs = sorted(r['seq'] for r in rows if r['node_id'] == node)
        for a, b in zip(seqs, seqs[1:]):
            if b > a + 1:
                seq_drops += b - a - 1
    report = {
        'rows': len(rows),
        'node_ids': node_ids,
        'node_count': len(node_ids),
        'seq_drops_est': seq_drops,
        'status': 'ok' if rows else 'empty',
    }
    (out.parent / 'quality_report.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bronze-root', required=True)
    ap.add_argument('--silver-out', required=True)
    args = ap.parse_args()
    print(json.dumps(bronze_to_silver(args.bronze_root, args.silver_out), indent=2))
