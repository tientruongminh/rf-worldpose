from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from rfpose.data.window_dataset import CsiWindowDataset, WindowDatasetConfig
from rfpose.models.rf_worldpose import RFWorldPose


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', default='data/gold/stub')
    ap.add_argument('--output', default='artifacts/eval/eval_report.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location=args.device)
    cfg=WindowDatasetConfig(**ckpt.get('cfg', {'path': args.dataset}))
    cfg.path=args.dataset
    ds=CsiWindowDataset(cfg,'val')
    dl=DataLoader(ds,batch_size=32,shuffle=False)
    model=RFWorldPose(num_nodes=cfg.num_nodes, window_frames=cfg.window_frames, n_subcarriers=cfg.n_subcarriers, channels=cfg.channels, num_classes=cfg.num_classes).to(args.device)
    model.load_state_dict(ckpt['model']); model.eval()
    ys=[]; preds=[]; lat=[]
    with torch.no_grad():
        for x,y in dl:
            x=x.to(args.device)
            t0=time.perf_counter(); logits=model(x)['action_logits']; lat.append((time.perf_counter()-t0)*1000)
            preds.extend(logits.argmax(-1).cpu().numpy().tolist()); ys.extend(y.numpy().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(ys,preds,average='macro',zero_division=0)
    report={
        'accuracy': float(accuracy_score(ys,preds)),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'latency_ms_p50_batch': float(np.percentile(lat,50)),
        'latency_ms_p95_batch': float(np.percentile(lat,95)),
        'confusion_matrix': confusion_matrix(ys,preds).tolist(),
    }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
