from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rfpose.data.window_dataset import CsiWindowDataset, WindowDatasetConfig
from rfpose.models.rf_worldpose import RFWorldPose

def set_seed(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def kd_loss(student_logits, teacher_logits, y, temperature: float, alpha: float):
    ce = nn.functional.cross_entropy(student_logits, y)
    kl = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        nn.functional.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return alpha * kl + (1 - alpha) * ce

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/gold/stub')
    ap.add_argument('--teacher-checkpoint', required=True)
    ap.add_argument('--output', default='artifacts/runs/distill')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--temperature', type=float, default=4.0)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); set_seed(42)
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg=WindowDatasetConfig(path=args.dataset)
    ds=CsiWindowDataset(cfg,'train'); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True)
    teacher=RFWorldPose(num_nodes=cfg.num_nodes, window_frames=cfg.window_frames, n_subcarriers=cfg.n_subcarriers, channels=cfg.channels, dim=128, depth=4, heads=4, num_classes=cfg.num_classes).to(args.device)
    ckpt=torch.load(args.teacher_checkpoint,map_location=args.device); teacher.load_state_dict(ckpt['model']); teacher.eval()
    student=RFWorldPose(num_nodes=cfg.num_nodes, window_frames=cfg.window_frames, n_subcarriers=cfg.n_subcarriers, channels=cfg.channels, dim=64, depth=2, heads=2, num_classes=cfg.num_classes).to(args.device)
    opt=torch.optim.AdamW(student.parameters(),lr=5e-4)
    hist=[]
    for epoch in range(args.epochs):
        total=0; steps=0
        for x,y in dl:
            x=x.to(args.device); y=y.to(args.device)
            with torch.no_grad(): tlog=teacher(x)['action_logits']
            slog=student(x)['action_logits']
            loss=kd_loss(slog,tlog,y,args.temperature,args.alpha)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item(); steps += 1
        row={'epoch':epoch+1,'distill_loss':total/max(steps,1)}; hist.append(row); print(json.dumps(row))
    torch.save({'model':student.state_dict(),'cfg':cfg.__dict__,'distilled_from':args.teacher_checkpoint}, out/'student.pt')
    (out/'history.json').write_text(json.dumps(hist,indent=2))
if __name__=='__main__': main()
