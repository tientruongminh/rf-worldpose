from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rfpose.data.window_dataset import CsiWindowDataset, WindowDatasetConfig
from rfpose.models.rf_worldpose import RFWorldPose


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def accuracy(logits, y):
    return (logits.argmax(dim=-1) == y).float().mean().item()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='data/gold/stub')
    ap.add_argument('--output', default='artifacts/runs/smoke')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    set_seed(args.seed)
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg=WindowDatasetConfig(path=args.dataset)
    train_ds=CsiWindowDataset(cfg,'train'); val_ds=CsiWindowDataset(cfg,'val')
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=RFWorldPose(num_nodes=cfg.num_nodes, window_frames=cfg.window_frames, n_subcarriers=cfg.n_subcarriers, channels=cfg.channels, num_classes=cfg.num_classes).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1))
    loss_fn=nn.CrossEntropyLoss()
    best_acc=-1.0
    history=[]
    for epoch in range(args.epochs):
        model.train(); total_loss=0; total_acc=0; steps=0
        for x,y in train_dl:
            x=x.to(args.device); y=y.to(args.device)
            opt.zero_grad(set_to_none=True)
            outp=model(x)
            loss=loss_fn(outp['action_logits'],y)
            loss.backward(); opt.step()
            total_loss += loss.item(); total_acc += accuracy(outp['action_logits'],y); steps += 1
        sched.step()
        model.eval(); val_loss=0; val_acc=0; vsteps=0
        with torch.no_grad():
            for x,y in val_dl:
                x=x.to(args.device); y=y.to(args.device)
                outp=model(x); loss=loss_fn(outp['action_logits'],y)
                val_loss += loss.item(); val_acc += accuracy(outp['action_logits'],y); vsteps += 1
        row={
            'epoch': epoch+1,
            'train_loss': total_loss/max(steps,1),
            'train_acc': total_acc/max(steps,1),
            'val_loss': val_loss/max(vsteps,1),
            'val_acc': val_acc/max(vsteps,1),
            'lr': sched.get_last_lr()[0],
        }
        history.append(row); print(json.dumps(row))
        if row['val_acc'] > best_acc:
            best_acc=row['val_acc']
            torch.save({'model':model.state_dict(),'cfg':cfg.__dict__,'metrics':row}, out/'best.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'metrics.json').write_text(json.dumps({'best_val_acc':best_acc},indent=2))

if __name__=='__main__': main()
