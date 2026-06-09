import logging
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from rfpose.models.rf_transformer import RFPoseModel
from rfpose.utils.transformer_loss import RFPoseLoss
from evaluation.eval_transformer import PoseEvaluator

log = logging.getLogger(__name__)


class GoldNpzDataset(torch.utils.data.Dataset):
    """
    Load một dataset từ thư mục Gold v2.
        x.npy        (N, 2, 60, 270)
        y.npz        pose(N,J,3), pose_mask(N,J), action_label(N,), action_mask(N,)
        metadata.npz split per sample
    """
    def __init__(self, dataset_dir: str, split: str = "train"):
        self.dir = Path(dataset_dir)
        x_path   = self.dir / "x.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {x_path}")

        self.x = np.load(str(x_path), mmap_mode="r")
        y = np.load(str(self.dir / "y.npz"), allow_pickle=True)
        self.pose         = y["pose"]
        self.pose_mask    = y["pose_mask"]
        self.action_label = y["action_label"]
        self.action_mask  = y["action_mask"]

        meta = np.load(str(self.dir / "metadata.npz"), allow_pickle=True)
        self.indices = np.where(meta["split"] == split)[0]
        log.info(f"  [{self.dir.name}] {split}: {len(self.indices)} samples")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return {
            "x":            torch.from_numpy(self.x[i].copy()).float(),
            "pose":         torch.from_numpy(self.pose[i].copy()).float(),
            "pose_mask":    torch.from_numpy(self.pose_mask[i].copy()).float(),
            "action_label": torch.tensor(int(self.action_label[i]), dtype=torch.long),
            "action_mask":  torch.tensor(float(self.action_mask[i]), dtype=torch.float),
        }


def build_dataloaders(cfg: DictConfig):
    gold_dir = Path(cfg.data.gold_dir)
    names    = list(cfg.data.datasets) if cfg.data.datasets else [
        d.name for d in sorted(gold_dir.iterdir())
        if d.is_dir() and (d / "x.npy").exists()
    ]
    log.info(f"Datasets ({len(names)}): {names}")

    train_sets, val_sets = [], []
    for name in names:
        try:
            train_sets.append(GoldNpzDataset(str(gold_dir / name), "train"))
            val_sets.append(GoldNpzDataset(str(gold_dir / name), "val"))
        except FileNotFoundError as e:
            log.warning(f"Bỏ qua {name}: {e}")

    if not train_sets:
        raise RuntimeError("Không load được dataset nào!")

    log.info(f"Total train={sum(len(d) for d in train_sets):,}  "
             f"val={sum(len(d) for d in val_sets):,}")

    def make_loader(ds_list, shuffle):
        return DataLoader(
            ConcatDataset(ds_list),
            batch_size  = cfg.training.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.training.get("num_workers", 4),
            pin_memory  = True,
            drop_last   = shuffle,
        )
    return make_loader(train_sets, True), make_loader(val_sets, False)


# ──────────────────────────────────────────────
# LR Scheduler
# ──────────────────────────────────────────────

def build_scheduler(optimizer, cfg, n_steps_per_epoch):
    warmup = cfg.training.get("warmup_epochs", 3) * n_steps_per_epoch
    total  = cfg.training.epochs * n_steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
# Training loop — 1 epoch
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scheduler,
                    scaler, device, cfg, epoch) -> dict:
    model.train()
    model.encoder.eval()   # encoder frozen, đảm bảo BN/dropout không update

    accum = {"total": 0., "pose": 0., "action": 0., "vis": 0., "pres": 0.}
    n_batches = 0
    log_every = cfg.training.get("log_every", 50)

    for step, batch in enumerate(loader):
        x            = batch["x"].to(device, non_blocking=True)
        pose         = batch["pose"].to(device, non_blocking=True)
        pose_mask    = batch["pose_mask"].to(device, non_blocking=True)
        action_label = batch["action_label"].to(device, non_blocking=True)
        action_mask  = batch["action_mask"].to(device, non_blocking=True)
        # presence_target: assume tất cả samples đều có người (=1)
        # nếu dataset có field "presence" thì load thêm
        presence_target = batch.get(
            "presence",
            torch.ones(x.shape[0], device=device)
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.training.get("amp", True)):
            out = model(x)                  # dict: coords, vis_logits, ...
            loss, info = loss_fn(
                pred_coords     = out["coords"],
                pred_vis        = out["vis_logits"],
                pred_action     = out["action_logits"],
                pred_presence   = out["presence_logit"],
                target_joints   = pose,
                pose_mask       = pose_mask,
                action_label    = action_label,
                action_mask     = action_mask,
                presence_target = presence_target,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=cfg.training.get("grad_clip", 1.0),
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        accum["total"]  += info["loss_total"]
        accum["pose"]   += info["loss_pose"]
        accum["action"] += info["loss_action"]
        accum["vis"]    += info["loss_vis"]
        accum["pres"]   += info["loss_presence"]
        n_batches += 1

        if (step + 1) % log_every == 0:
            log.info(
                f"  Epoch {epoch} [{step+1}/{len(loader)}] "
                f"total={info['loss_total']:.4f}  "
                f"pose={info['loss_pose']:.4f}  "
                f"action={info['loss_action']:.4f}  "
                f"vis={info['loss_vis']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return {
        "train_loss_total":    accum["total"]  / n_batches,
        "train_loss_pose":     accum["pose"]   / n_batches,
        "train_loss_action":   accum["action"] / n_batches,
        "train_loss_vis":      accum["vis"]    / n_batches,
        "train_loss_presence": accum["pres"]   / n_batches,
        "lr": scheduler.get_last_lr()[0],
    }


# ──────────────────────────────────────────────
# Validation loop
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn, evaluator, device, cfg) -> dict:
    model.eval()

    accum = {"total": 0., "pose": 0., "action": 0., "vis": 0., "pres": 0.}
    n_batches = 0
    evaluator.reset()

    for batch in loader:
        x            = batch["x"].to(device, non_blocking=True)
        pose         = batch["pose"].to(device, non_blocking=True)
        pose_mask    = batch["pose_mask"].to(device, non_blocking=True)
        action_label = batch["action_label"].to(device, non_blocking=True)
        action_mask  = batch["action_mask"].to(device, non_blocking=True)
        presence_target = batch.get(
            "presence",
            torch.ones(x.shape[0], device=device)
        )

        with autocast(enabled=cfg.training.get("amp", True)):
            out = model(x)
            _, info = loss_fn(
                pred_coords     = out["coords"],
                pred_vis        = out["vis_logits"],
                pred_action     = out["action_logits"],
                pred_presence   = out["presence_logit"],
                target_joints   = pose,
                pose_mask       = pose_mask,
                action_label    = action_label,
                action_mask     = action_mask,
                presence_target = presence_target,
            )

        evaluator.update(
            model_out        = {k: v.float() for k, v in out.items()},
            target_joints    = pose.float(),
            pose_mask        = pose_mask,
            action_label     = action_label,
            action_mask      = action_mask,
            presence_target  = presence_target,
        )

        accum["total"]  += info["loss_total"]
        accum["pose"]   += info["loss_pose"]
        accum["action"] += info["loss_action"]
        accum["vis"]    += info["loss_vis"]
        accum["pres"]   += info["loss_presence"]
        n_batches += 1

    metrics = evaluator.compute()
    metrics.update({
        "val_loss_total":    accum["total"]  / n_batches,
        "val_loss_pose":     accum["pose"]   / n_batches,
        "val_loss_action":   accum["action"] / n_batches,
        "val_loss_vis":      accum["vis"]    / n_batches,
        "val_loss_presence": accum["pres"]   / n_batches,
    })
    return metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="transformer_gold", version_base=None)
def train(cfg: DictConfig):
    torch.manual_seed(cfg.training.get("seed", 42))
    np.random.seed(cfg.training.get("seed", 42))

    device = torch.device(
        cfg.training.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Device: {device}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params({
            "model_version":    "v3",
            "d_model":          cfg.model.d_model,
            "n_spatial_layers": cfg.model.n_spatial_layers,
            "n_temporal_layers":cfg.model.n_temporal_layers,
            "n_decoder_layers": cfg.model.get("n_decoder_layers", 3),
            "batch_size":       cfg.training.batch_size,
            "lr":               cfg.training.lr,
            "epochs":           cfg.training.epochs,
            "datasets":         str(cfg.data.datasets),
            "ssl_pretrained":   cfg.training.get("ssl_pretrained", ""),
        })

        # ── Model ──
        ssl_ckpt = cfg.training.get("ssl_pretrained", "")
        if not ssl_ckpt:
            raise ValueError("Cần chỉ định training.ssl_pretrained trong config!")

        model = RFPoseModel.from_ssl_checkpoint(
            ssl_ckpt_path      = ssl_ckpt,
            n_joints           = cfg.data.get("n_joints", 13),
            num_actions        = cfg.model.get("num_actions", 28),
            freeze_encoder     = cfg.training.get("freeze_encoder", True),
            n_decoder_layers   = cfg.model.get("n_decoder_layers", 3),
            n_temporal_layers  = cfg.model.get("n_decoder_temporal_layers", 2),
        ).to(device)

        # ── Loss ──
        loss_fn = RFPoseLoss(
            label_smoothing=cfg.loss.get("label_smoothing", 0.1),
        ).to(device)

        # ── Optimizer — chỉ train decoder + cls + Kendall sigmas ──
        trainable_params = (
            list(model.pose_decoder.parameters()) +
            list(model.cls_module.parameters())   +
            list(loss_fn.combiner.parameters())
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.get("weight_decay", 1e-4),
        )

        # ── Data ──
        train_loader, val_loader = build_dataloaders(cfg)
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))
        scaler    = GradScaler(enabled=cfg.training.get("amp", True))
        evaluator = PoseEvaluator(num_actions=cfg.model.get("num_actions", 28))

        ckpt_dir   = Path(cfg.training.get("checkpoint_dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_mpjpe = float("inf")
        no_improve = 0
        patience   = cfg.training.get("patience", 15)
        save_every = cfg.training.get("save_every", 5)

        for epoch in range(1, cfg.training.epochs + 1):
            t0 = time.time()

            train_m = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                      scheduler, scaler, device, cfg, epoch)
            val_m   = validate(model, val_loader, loss_fn, evaluator, device, cfg)

            elapsed = time.time() - t0
            mlflow.log_metrics({**train_m, **val_m}, step=epoch)

            log.info(
                f"Epoch {epoch:03d}/{cfg.training.epochs} [{elapsed:.0f}s] | "
                f"train={train_m['train_loss_total']:.4f} | "
                f"val={val_m['val_loss_total']:.4f} | "
                f"{evaluator.log_str()}"
            )

            # ── Best checkpoint ──
            cur_mpjpe = val_m.get("mpjpe", float("inf"))
            if cur_mpjpe < best_mpjpe:
                best_mpjpe = cur_mpjpe
                no_improve = 0
                best_path  = ckpt_dir / "best.pt"
                torch.save({
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "loss_fn":     loss_fn.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_mpjpe":   best_mpjpe,
                    "val_metrics": val_m,
                    "cfg":         OmegaConf.to_container(cfg),
                }, str(best_path))
                mlflow.log_artifact(str(best_path))
                log.info(f"  ✅ Best saved  MPJPE={best_mpjpe:.2f}  "
                         f"VisAcc={val_m.get('vis_acc', float('nan')):.3f}")
            else:
                no_improve += 1

            if epoch % save_every == 0:
                p = ckpt_dir / f"epoch_{epoch:03d}.pt"
                torch.save({"epoch": epoch, "model": model.state_dict()}, str(p))
                mlflow.log_artifact(str(p))

            if no_improve >= patience:
                log.info(f"Early stopping sau {patience} epochs không cải thiện.")
                break

        log.info(f"Done! Best MPJPE={best_mpjpe:.2f}")
        mlflow.log_metric("best_mpjpe", best_mpjpe)


if __name__ == "__main__":
    train()