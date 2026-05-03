from __future__ import annotations
from pathlib import Path
import json, hashlib, shutil, argparse

def sha256(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def package_model(model_path: str, eval_report: str, output_dir: str, name: str, dataset_version: str):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    mp=Path(model_path); er=Path(eval_report)
    shutil.copy2(mp, out/mp.name); shutil.copy2(er, out/'eval_report.json')
    digest=sha256(out/mp.name)
    metrics=json.loads(er.read_text()) if er.exists() else {}
    card=f"""# Model Card: {name}\n\n- Dataset version: `{dataset_version}`\n- Artifact: `{mp.name}`\n- SHA256: `{digest}`\n- Status: candidate until eval gates pass.\n\n## Metrics\n\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n## Intended use\n\nWiFi CSI human sensing inference through RF-WorldPose edge/cloud serving.\n\n## Limitations\n\nRoom/layout dependent; requires CSI quality gates and domain adaptation.\n"""
    (out/'model_card.md').write_text(card)
    (out/'manifest.json').write_text(json.dumps({'name':name,'dataset_version':dataset_version,'sha256':digest,'metrics':metrics},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--eval-report',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--name',default='rfworldpose'); ap.add_argument('--dataset-version',default='unknown')
    a=ap.parse_args(); package_model(a.model,a.eval_report,a.output_dir,a.name,a.dataset_version)
if __name__=='__main__': main()
