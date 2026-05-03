from __future__ import annotations
import argparse
from pathlib import Path
import torch
from rfpose.data.window_dataset import WindowDatasetConfig
from rfpose.models.rf_worldpose import RFWorldPose


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', default='artifacts/models/model.onnx')
    ap.add_argument('--opset', type=int, default=17)
    args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu')
    cfg=WindowDatasetConfig(**ckpt.get('cfg', {'path':'data/gold/stub'}))
    model=RFWorldPose(num_nodes=cfg.num_nodes, window_frames=cfg.window_frames, n_subcarriers=cfg.n_subcarriers, channels=cfg.channels, num_classes=cfg.num_classes)
    model.load_state_dict(ckpt['model']); model.eval()
    dummy=torch.randn(1,cfg.num_nodes,cfg.window_frames,cfg.n_subcarriers,cfg.channels)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model, dummy, out, input_names=['csi'], output_names=['outputs'], opset_version=args.opset, dynamic_axes={'csi':{0:'batch'}})
    print(f'exported {out}')
if __name__=='__main__': main()
