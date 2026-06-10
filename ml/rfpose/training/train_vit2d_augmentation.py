"""
train_vit2d_augmentation.py
"""

import logging
import os
import time
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from rfpose.data.gold_npz_dataset import GoldNpzDataset, NUM_ACTIONS
from rfpose.models.vit2d_pose import CSIViT2DPose, CsiPatchEmbedding2D
from rfpose.utils.loss_aug import RFPoseLoss, LossConfig
from rfpose.data.augmentations import CSISpecAugment

log = logging.getLogger(__name__)

def build_model_and_tokenizer(cfg: DictConfig, device: torch.device):
    d = cfg.data
    m = cfg.model
    patch_freq = m.patch_size[0] if isinstance(m.patch_size, list) else m.patch_size
    
    tokenizer = CsiPatchEmbedding2D(
        in_channels=2,
        d_model=m.d_model,
        patch_freq=patch_freq,
        patch_time=3
    ).to(device)

    model = CSIViT2DPose(
        n_subcarriers=d.n_subcarriers,
        patch_freq=patch_freq,
        d_model=m.d_model,
        n_layers=m.n_spatial_layers,
        n_heads=m.spatial_heads,
        n_joints=d.get("n_joints", 17),
        num_actions=m.get("num_actions", NUM_ACTIONS)
    ).to(device)

    return tokenizer, model

def train_one_epoch(cfg, epoch, tokenizer, model, loader, optimizer, loss_fn, device, spec_aug):
    tokenizer.train()
    model.train()
    spec_aug.train()

    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        csi = batch["csi"].to(device)
        csi = spec_aug(csi)

        optimizer.zero_grad()
        tokens = tokenizer(csi)
        out = model(tokens)

        # init dicts for gt and pred
        gt_dict = {}
        pred_dict = {}

        # if data has coords and vis, add them to the dict
        if "coords" in batch and "vis" in batch:
            gt_dict["coords"] = batch["coords"].to(device)
            gt_dict["vis"] = batch["vis"].to(device)
            pred_dict["coords"] = out["coords"]
            pred_dict["vis_logits"] = out["vis_logits"]

        # if data has action labels, add them to the dict
        if "action_label" in batch:
            gt_dict["action_label"] = batch["action_label"].to(device)
            if "action_logits" in out:
                pred_dict["action_logits"] = out["action_logits"]

        # if batch has nothing, pass
        if not gt_dict:
            continue

        loss, breakdown = loss_fn(pred_dict, gt_dict)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.get("grad_clip", 1.0))
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(cfg, tokenizer, model, loader, loss_fn, device):
    tokenizer.eval()
    model.eval()
    
    total_val_loss = 0.0
    correct_actions = 0
    total_actions = 0
    valid_batches = 0
    
    for batch in loader:
        csi = batch["csi"].to(device)
        
        # Validation not apply SpecAugment
        tokens = tokenizer(csi)
        out = model(tokens)
        
        gt_dict = {}
        pred_dict = {}
        
        # pose
        if "coords" in batch and "vis" in batch:
            gt_dict["coords"] = batch["coords"].to(device)
            gt_dict["vis"] = batch["vis"].to(device)
            pred_dict["coords"] = out["coords"]
            pred_dict["vis_logits"] = out["vis_logits"]

        # action
        if "action_label" in batch:
            gt_dict["action_label"] = batch["action_label"].to(device)
            if "action_logits" in out:
                pred_dict["action_logits"] = out["action_logits"]

        if not gt_dict:
            continue

        # loss
        loss, breakdown = loss_fn(pred_dict, gt_dict)
        total_val_loss += loss.item()
        valid_batches += 1
        
        # Accuracy for action classification if available
        if "action_logits" in out and "action_label" in batch:
            preds = torch.argmax(out["action_logits"], dim=1)
            correct_actions += (preds == gt_dict["action_label"]).sum().item()
            total_actions += preds.size(0)
            
    avg_loss = (total_val_loss / valid_batches) if valid_batches > 0 else 0.0
    action_acc = (correct_actions / total_actions) if total_actions > 0 else 0.0
    return avg_loss, action_acc


@hydra.main(config_path="../../configs", config_name="vit2d_full_augmentation", version_base=None)
def train(cfg: DictConfig) -> None:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    tokenizer, model = build_model_and_tokenizer(cfg, device)
    
    # Initialize SpecAugment
    aug_cfg = cfg.get("augmentation", {})
    spec_aug = CSISpecAugment(
        time_mask_param=aug_cfg.get("time_mask", 12),
        freq_mask_param=aug_cfg.get("freq_mask", 20),
        p=aug_cfg.get("prob", 0.5)
    ).to(device)

    # Dataloaders
    train_ds = GoldNpzDataset(cfg.data.gold_dir, split="train", datasets=cfg.data.datasets)
    val_ds = GoldNpzDataset(cfg.data.gold_dir, split="val", datasets=cfg.data.datasets)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    loss_mode = cfg.get("weighting_mode", "uncertainty") 
    loss_fn = RFPoseLoss(LossConfig(weighting_mode=loss_mode)).to(device)

    log.info(f"Initialize model with SpecAugment (Time: {spec_aug.time_mask_param}, Freq: {spec_aug.freq_mask_param}, Prob: {spec_aug.p})")

    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        
        train_loss = train_one_epoch(cfg, epoch, tokenizer, model, train_loader, optimizer, loss_fn, device, spec_aug)
        val_loss, val_action_acc = validate(cfg, tokenizer, model, val_loader, loss_fn, device)
        
        epoch_time = time.time() - start_time
        log.info(f"Epoch {epoch} | Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Action Acc: {val_action_acc:.4f}")

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(cfg.training.checkpoint_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            log.info(f"--> Saved new checkpoint at epoch {epoch}")

if __name__ == "__main__":
    train()