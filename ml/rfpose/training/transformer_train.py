"""
transformer_train.py
--------------------
Training script cho CSI Transformer Pose model.

Features:
    - Hydra config management (override từ CLI)
    - MLflow experiment tracking (tích hợp với stack đã có)
    - Gradient clipping, LR scheduling, early stopping
    - Mixed precision training (AMP) cho Helios GH200
    - Checkpoint save/load
    - Validation metrics: MPJPE, PA-MPJPE

Large-scale optimizations:
    - Lazy Parquet loading: chỉ đọc row-group cần thiết, không load cả file
    - Cached index file (JSON) để tránh re-scan toàn bộ dataset khi restart
    - Vectorized PA-MPJPE: loại bỏ double for-loop B×T
    - Multi-node Parquet: dùng pl.col() syntax đúng cho Polars filter
    - torch.amp API mới (không dùng torch.cuda.amp deprecated)

Usage:
    python transformer_train.py
    python transformer_train.py model.d_model=512 training.lr=1e-4
    python transformer_train.py training.resume_from=checkpoints/epoch_10.pt
    python transformer_train.py training.dry_run=true

"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader, random_split

import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from rfpose.models.csi_tokenizer import CSITokenizer
from rfpose.models.transformer import CSITransformerPose
from rfpose.utils.losses import RFPoseLoss, LossConfig, MPJPE, PA_MPJPE

log = logging.getLogger(__name__)


# ===========================================================================
# Dataset — Large-scale optimized
# ===========================================================================
class CSIPoseDataset(Dataset):
    """
    Load CSI + Pose data từ MinIO Gold layer (Parquet files).

    Gold schema (output của Dagster ETL):
        - csi_amplitude:  float32[T, N_sub]   — normalized amplitude
        - csi_phase:      float32[T, N_sub]    — unwrapped phase
        - pose_coords:    float32[T, J, 3]     — joint xyz (normalized by room size)
        - pose_vis:       float32[T, J]         — joint visibility
        - session_id:     string
        - node_id:        int (chỉ có trong multi-node schema)

    
    Polars list column → numpy robust conversion (tránh dtype=object crash)
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 100,
        stride: int = 50,
        n_subcarriers: int = 114,
        n_joints: int = 17,
        n_nodes: int = 1,
        augment: bool = False,
        cache_index: bool = True,
    ):
        self.data_dir      = Path(data_dir)
        self.window_size   = window_size
        self.stride        = stride
        self.n_subcarriers = n_subcarriers
        self.n_joints      = n_joints
        self.n_nodes       = n_nodes
        self.augment       = augment
        self.cache_index   = cache_index

        self.samples = self._load_or_build_index()
        log.info(f"Dataset: {len(self.samples)} windows từ {data_dir}")

    # ------------------------------------------------------------------
    # Index: build một lần, cache JSON, tái dùng khi restart
    # ------------------------------------------------------------------
    def _index_cache_path(self) -> Path:
        """Cache key dựa trên data_dir + window_size + stride để tránh stale cache."""
        key = f"{self.data_dir}|{self.window_size}|{self.stride}|{self.n_nodes}"
        h   = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.data_dir / f".rfpose_index_{h}.json"

    def _load_or_build_index(self) -> list[dict]:
        cache_path = self._index_cache_path()

        if self.cache_index and cache_path.exists():
            log.info(f"Loading cached index: {cache_path}")
            with open(cache_path) as f:
                samples = json.load(f)
            return samples if samples else self._synthetic_index()

        samples = self._build_index()

        if self.cache_index and samples and samples[0]["file"] is not None:
            with open(cache_path, "w") as f:
                json.dump([{**s, "file": str(s["file"])} for s in samples], f)
            log.info(f"Index cached: {cache_path} ({len(samples)} windows)")

        return samples

    def _build_index(self) -> list[dict]:
        """Scan Parquet files và build index. Dùng metadata — KHÔNG load data."""
        import pyarrow.parquet as pq

        parquet_files = sorted(self.data_dir.glob("**/*.parquet"))

        if not parquet_files:
            log.warning(f"Không tìm thấy .parquet files trong {self.data_dir}")
            log.warning("Tạo synthetic data để smoke test...")
            return self._synthetic_index()

        samples = []
        for fpath in parquet_files:
            meta     = pq.read_metadata(fpath)
            n_frames = meta.num_rows

            if self.n_nodes > 1:
                n_frames = n_frames // self.n_nodes

            for start in range(0, n_frames - self.window_size + 1, self.stride):
                samples.append({
                    "file":  str(fpath),
                    "start": start,
                    "end":   start + self.window_size,
                })

        log.info(f"Index built: {len(parquet_files)} files → {len(samples)} windows")
        return samples

    def _synthetic_index(self) -> list[dict]:
        return [{"file": None, "start": i, "end": i + self.window_size} for i in range(200)]

    # ------------------------------------------------------------------
    #Robust Polars column → numpy/tensor conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _polars_col_to_tensor(
        col,
        target_shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Chuyển đổi Polars column → torch.Tensor robust.

        Vấn đề: nếu Parquet lưu cột dạng List(Float32), col.to_numpy()
        trả về dtype=object array of lists → torch.tensor().reshape() crash.

        Solution: kiểm tra dtype, nếu List thì np.stack từng row.
        """
        import numpy as np

        # Kiểm tra Polars List type
        if hasattr(col, 'dtype'):
            dt = col.dtype
            is_list = (
                str(dt).startswith("List")
                or dt == type(col.dtype).__dict__.get('List', None)
            )
        else:
            is_list = False

        if is_list:
            # List column → stack từng row
            arr = np.stack(col.to_list(), axis=0)
            return torch.tensor(arr, dtype=dtype).reshape(target_shape)
        else:
            # Flat column → reshape trực tiếp
            return torch.tensor(col.to_numpy(), dtype=dtype).reshape(target_shape)

    # ------------------------------------------------------------------
    # Lazy load: chỉ đọc đúng rows cần, không load cả file
    # ------------------------------------------------------------------
    def _load_window(self, fpath: str, start: int) -> dict[str, torch.Tensor]:
        """
        Dùng pl.scan_parquet() (lazy) + .slice() để chỉ đọc window cần thiết.
        Với Parquet row-group size = 512-1024 rows, thường chỉ đọc 1-2 row groups.
        """
        import polars as pl

        T, N, J = self.window_size, self.n_subcarriers, self.n_joints

        if self.n_nodes > 1:
            # Multi-node: file có cột node_id, interleaved theo node
            # Convention Gold schema: sort theo (frame_idx, node_id)
            row_start = start * self.n_nodes
            df = (
                pl.scan_parquet(fpath)
                .slice(row_start, T * self.n_nodes)
                .collect()
            )

            node_csi_list = []
            for node_id in range(self.n_nodes):
                node_df = df.filter(pl.col("node_id") == node_id)

                # Robust conversion cho CSI amplitude/phase
                amp_arr = self._polars_col_to_tensor(
                    node_df["csi_amplitude"], (T, N)
                )
                phase_arr = self._polars_col_to_tensor(
                    node_df["csi_phase"], (T, N)
                )
                node_csi_list.append(torch.stack([amp_arr, phase_arr], dim=-1))

            csi = torch.stack(node_csi_list, dim=0)  # (n_nodes, T, N, 2)

            # Pose chỉ lấy từ node_id=0 (ground truth từ camera)
            ref_df = df.filter(pl.col("node_id") == 0)

        else:
            # Single-node: đọc đúng T rows bắt đầu từ start
            df = (
                pl.scan_parquet(fpath)
                .slice(start, T)
                .collect()
            )

            # Robust conversion
            amp_arr = self._polars_col_to_tensor(
                df["csi_amplitude"], (T, N)
            )
            phase_arr = self._polars_col_to_tensor(
                df["csi_phase"], (T, N)
            )
            csi = torch.stack([amp_arr, phase_arr], dim=-1)  # (T, N, 2)
            ref_df = df

        # Robust conversion cho pose columns
        coords = self._polars_col_to_tensor(
            ref_df["pose_coords"], (T, J, 3)
        )
        vis = self._polars_col_to_tensor(
            ref_df["pose_vis"], (T, J)
        )

        return {"csi": csi, "coords": coords, "vis": vis}

    # ------------------------------------------------------------------
    # Synthetic data (smoke test khi chưa có real data)
    # ------------------------------------------------------------------
    def _load_synthetic(self, idx: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        T, N, J = self.window_size, self.n_subcarriers, self.n_joints

        if self.n_nodes > 1:
            amplitude = torch.abs(torch.randn(self.n_nodes, T, N)) + 0.5
            phase     = torch.rand(self.n_nodes, T, N) * 2 * 3.14159 - 3.14159
            csi = torch.stack([amplitude, phase], dim=-1)
        else:
            amplitude = torch.abs(torch.randn(T, N)) + 0.5
            phase     = torch.rand(T, N) * 2 * 3.14159 - 3.14159
            csi = torch.stack([amplitude, phase], dim=-1)

        coords = torch.randn(T, J, 3) * 0.5
        vis    = (torch.rand(T, J) > 0.2).float()

        return {"csi": csi, "coords": coords, "vis": vis}

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    def _augment(self, csi: torch.Tensor, coords: torch.Tensor) -> tuple:
        if torch.rand(1) < 0.5:
            noise_std = torch.rand(1).item() * 0.05
            csi = csi + torch.randn_like(csi) * noise_std

        if torch.rand(1) < 0.3:
            t_dim = 1 if csi.ndim == 4 else 0
            csi    = torch.flip(csi,    dims=[t_dim])
            coords = torch.flip(coords, dims=[0])

        if torch.rand(1) < 0.5:
            coords[..., 0] = -coords[..., 0]

        return csi, coords

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if sample["file"] is None:
            data = self._load_synthetic(idx)
        else:
            data = self._load_window(sample["file"], sample["start"])

        if self.augment:
            data["csi"], data["coords"] = self._augment(data["csi"], data["coords"])

        return data


# ===========================================================================
# Training utilities
# ===========================================================================
def build_model_and_tokenizer(cfg: DictConfig) -> tuple[CSITokenizer, CSITransformerPose]:
    tokenizer = CSITokenizer(
        n_subcarriers=cfg.data.n_subcarriers,
        patch_size=cfg.model.patch_size,
        d_model=cfg.model.d_model,
        max_seq_len=cfg.data.window_size + 10,
        n_nodes=cfg.data.n_nodes,
        dropout=cfg.model.dropout,
    )

    model = CSITransformerPose(
        n_patches=tokenizer.n_patches,
        d_model=cfg.model.d_model,
        spatial_heads=cfg.model.spatial_heads,
        temporal_heads=cfg.model.temporal_heads,
        n_spatial_layers=cfg.model.n_spatial_layers,
        n_temporal_layers=cfg.model.n_temporal_layers,
        n_decoder_layers=cfg.model.n_decoder_layers,
        n_decoder_temporal_layers=cfg.model.get("n_decoder_temporal_layers", 2),
        n_joints=cfg.data.n_joints,
        predict_3d=cfg.model.predict_3d,
        causal_temporal=cfg.model.get("causal_temporal", cfg.model.get("causal", False)),
        dropout=cfg.model.dropout,
        ffn_mult=cfg.model.ffn_mult,
        n_nodes=cfg.data.n_nodes,
    )

    return tokenizer, model


def build_optimizer(model: nn.Module, tokenizer: nn.Module, cfg: DictConfig):
    """AdamW với weight decay chỉ cho non-bias, non-norm parameters."""
    params_decay    = []
    params_no_decay = []

    for module in [tokenizer, model]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embed" in name:
                params_no_decay.append(param)
            else:
                params_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": params_decay,    "weight_decay": cfg.training.weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=1e-8,
    )


def build_scheduler(optimizer, cfg: DictConfig, n_steps_per_epoch: int):
    """Cosine annealing với linear warmup."""
    total_steps  = cfg.training.epochs * n_steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * n_steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    epoch: int,
    tokenizer: CSITokenizer,
    model: CSITransformerPose,
    optimizer,
    scheduler,
    metrics: dict,
    cfg: DictConfig,
    path: str,
):
    checkpoint = {
        "epoch":     epoch,
        "tokenizer": tokenizer.state_dict(),
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
        "config":    OmegaConf.to_container(cfg),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint(path: str, tokenizer, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    log.info(f"Resumed from epoch {checkpoint['epoch']}: {path}")
    return checkpoint["epoch"], checkpoint["metrics"]


# ===========================================================================
# Train / Eval loop
# ===========================================================================
def train_one_epoch(
    tokenizer:  CSITokenizer,
    model:      CSITransformerPose,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    loss_fn:    RFPoseLoss,
    device:     torch.device,
    scaler:     amp.GradScaler,
    cfg:        DictConfig,
    epoch:      int,
) -> dict[str, float]:

    tokenizer.train()
    model.train()

    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        csi       = batch["csi"].to(device, non_blocking=True)
        gt_coords = batch["coords"].to(device, non_blocking=True)
        gt_vis    = batch["vis"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=cfg.training.amp):
            tokens    = tokenizer(csi)
            out       = model(tokens)

            # model trả về "vis_logits" (raw), loss dùng BCEWithLogitsLoss
            pred_dict = {
                "coords":     out["coords"],
                "vis_logits": out["vis_logits"],  # raw logits — KHÔNG sigmoid
            }
            gt_dict = {"coords": gt_coords, "vis": gt_vis}
            loss, breakdown = loss_fn(pred_dict, gt_dict)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(tokenizer.parameters()) + list(model.parameters()),
            max_norm=cfg.training.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k, v in breakdown.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % cfg.training.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            log.info(
                f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={breakdown['loss_total']:.4f} lr={lr:.6f}"
            )

        if cfg.training.dry_run:
            break

    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def eval_one_epoch(
    tokenizer: CSITokenizer,
    model:     CSITransformerPose,
    loader:    DataLoader,
    loss_fn:   RFPoseLoss,
    device:    torch.device,
    cfg:       DictConfig,
) -> dict[str, float]:

    tokenizer.eval()
    model.eval()

    mpjpe_fn    = MPJPE()
    pa_mpjpe_fn = PA_MPJPE()

    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        csi       = batch["csi"].to(device)
        gt_coords = batch["coords"].to(device)
        gt_vis    = batch["vis"].to(device)

        tokens = tokenizer(csi)
        out    = model(tokens)

        pred_dict = {"coords": out["coords"], "vis_logits": out["vis_logits"]}
        gt_dict   = {"coords": gt_coords,     "vis": gt_vis}

        _, breakdown = loss_fn(pred_dict, gt_dict)

        mpjpe    = mpjpe_fn(out["coords"], gt_coords, gt_vis)
        pa_mpjpe = pa_mpjpe_fn(out["coords"], gt_coords)

        breakdown["mpjpe"]    = mpjpe.item()
        breakdown["pa_mpjpe"] = pa_mpjpe.item()

        for k, v in breakdown.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if cfg.training.dry_run:
            break

    return {f"val_{k}": v / n_batches for k, v in total_metrics.items()}


# ===========================================================================
# Main training function (Hydra entry point)
# ===========================================================================
@hydra.main(config_path="configs", config_name="transformer", version_base=None)
def train(cfg: DictConfig) -> None:
    log.info(f"\n{'='*60}")
    log.info("RF-WorldPose Transformer Training")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"{'='*60}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Dataset & Dataloader
    full_dataset = CSIPoseDataset(
        data_dir=cfg.data.gold_dir,
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        n_subcarriers=cfg.data.n_subcarriers,
        n_joints=cfg.data.n_joints,
        n_nodes=cfg.data.n_nodes,
        augment=cfg.data.augment,
        cache_index=True,
    )

    n_val   = int(len(full_dataset) * cfg.data.val_ratio)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(cfg.training.num_workers > 0),
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.training.num_workers > 0),
        prefetch_factor=2 if cfg.training.num_workers > 0 else None,
    )

    log.info(f"Train samples: {n_train} | Val samples: {n_val}")

    # Model
    tokenizer, model = build_model_and_tokenizer(cfg)
    tokenizer = tokenizer.to(device)
    model     = model.to(device)

    log.info(f"Tokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")
    log.info(f"Model params:     {model.count_params():,}")

    # Loss
    loss_cfg = LossConfig(
        lambda_coord=cfg.loss.lambda_coord,
        lambda_vis=cfg.loss.lambda_vis,
        lambda_bone=cfg.loss.lambda_bone,
        lambda_temporal=cfg.loss.lambda_temporal,
        lambda_symmetry=cfg.loss.lambda_symmetry,
        coord_loss_type=cfg.loss.coord_loss_type,
    )
    loss_fn = RFPoseLoss(loss_cfg).to(device)

    optimizer = build_optimizer(model, tokenizer, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # GradScaler — API mới
    scaler = amp.GradScaler(device=cfg.training.device, enabled=cfg.training.amp)

    # Resume
    start_epoch = 0
    best_mpjpe  = float("inf")

    if cfg.training.resume_from and Path(cfg.training.resume_from).exists():
        start_epoch, prev_metrics = load_checkpoint(
            cfg.training.resume_from, tokenizer, model, optimizer, scheduler
        )
        best_mpjpe  = prev_metrics.get("val_mpjpe", float("inf"))
        start_epoch += 1

    # MLflow tracking
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.training, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.loss, resolve=True))

        patience_counter = 0

        for epoch in range(start_epoch, cfg.training.epochs):
            t0 = time.time()

            train_metrics = train_one_epoch(
                tokenizer, model, train_loader,
                optimizer, scheduler, loss_fn,
                device, scaler, cfg, epoch,
            )

            val_metrics = eval_one_epoch(
                tokenizer, model, val_loader,
                loss_fn, device, cfg,
            )

            epoch_time = time.time() - t0
            all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "epoch_time": epoch_time}

            mlflow.log_metrics(all_metrics, step=epoch)

            log.info(
                f"Epoch {epoch:03d}/{cfg.training.epochs} "
                f"[{epoch_time:.1f}s] "
                f"train_loss={train_metrics['loss_total']:.4f} "
                f"val_mpjpe={val_metrics['val_mpjpe']:.4f} "
                f"val_pa_mpjpe={val_metrics['val_pa_mpjpe']:.4f}"
            )

            val_mpjpe = val_metrics["val_mpjpe"]
            if val_mpjpe < best_mpjpe:
                best_mpjpe       = val_mpjpe
                patience_counter = 0
                best_ckpt        = "checkpoints/best.pt"
                save_checkpoint(epoch, tokenizer, model, optimizer, scheduler, all_metrics, cfg, best_ckpt)
                mlflow.log_artifact(best_ckpt)
                log.info(f"  New best MPJPE: {best_mpjpe:.4f}")
            else:
                patience_counter += 1
                log.info(f"  No improvement. Patience: {patience_counter}/{cfg.training.patience}")

            if epoch % cfg.training.save_every == 0:
                save_checkpoint(
                    epoch, tokenizer, model, optimizer, scheduler,
                    all_metrics, cfg, f"checkpoints/epoch_{epoch:03d}.pt",
                )

            if patience_counter >= cfg.training.patience:
                log.info(f"Early stopping tại epoch {epoch}")
                break

            if cfg.training.dry_run:
                log.info("Dry run complete.")
                break

        # Export ONNX
        if not cfg.training.dry_run:
            log.info("Exporting model to ONNX...")
            _export_onnx(tokenizer, model, cfg, device)
            mlflow.log_artifact("checkpoints/model.onnx")

        log.info(f"Training complete. Best MPJPE: {best_mpjpe:.4f}")
        mlflow.log_metric("best_mpjpe", best_mpjpe)


def _export_onnx(
    tokenizer: CSITokenizer,
    model:     CSITransformerPose,
    cfg:       DictConfig,
    device:    torch.device,
):
    """Export tokenizer + model thành ONNX cho Triton/edge inference."""
    tokenizer.eval()
    model.eval()

    B, T, N = 1, cfg.data.window_size, cfg.data.n_subcarriers

    # Single-node input shape cho ONNX export
    dummy_csi = torch.randn(B, T, N, 2).to(device)

    class FullModel(nn.Module):
        def __init__(self, tok, mod):
            super().__init__()
            self.tokenizer = tok
            self.model     = mod

        def forward(self, csi):
            tokens = self.tokenizer(csi)
            out    = self.model(tokens)
            
            return out["coords"], out["vis_logits"]

    full_model = FullModel(tokenizer, model)

    Path("checkpoints").mkdir(exist_ok=True)
    torch.onnx.export(
        full_model,
        dummy_csi,
        "checkpoints/model.onnx",
        input_names=["csi"],
        output_names=["coords", "vis_logits"],
        dynamic_axes={
            "csi":        {0: "batch", 1: "time"},
            "coords":     {0: "batch", 1: "time"},
            "vis_logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    log.info("ONNX export complete: checkpoints/model.onnx")


# ===========================================================================
# Hydra config default
# ===========================================================================
def _write_default_config():
    config_dir  = Path("configs")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "transformer.yaml"

    if config_path.exists():
        return

    config_content = """
# configs/transformer.yaml

data:
  gold_dir: "data/gold"
  n_subcarriers: 114
  n_joints: 17
  n_nodes: 1                   # 1 = single ESP32, 4 = multi-node
  window_size: 100
  stride: 50
  val_ratio: 0.2
  augment: true

model:
  patch_size: 6
  d_model: 256
  spatial_heads: 8
  temporal_heads: 8
  n_spatial_layers: 4
  n_temporal_layers: 4
  n_decoder_layers: 3
  n_decoder_temporal_layers: 2
  ffn_mult: 4
  predict_3d: true
  causal_temporal: false
  dropout: 0.1

loss:
  lambda_coord: 1.0
  lambda_vis: 0.5
  lambda_bone: 0.3
  lambda_temporal: 0.2
  lambda_symmetry: 0.1
  coord_loss_type: "smooth_l1"

training:
  epochs: 100
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 1.0e-4
  beta1: 0.9
  beta2: 0.999
  grad_clip: 1.0
  warmup_epochs: 5
  patience: 20
  save_every: 10
  log_every: 10
  amp: true
  num_workers: 4
  seed: 42
  device: "cuda"
  resume_from: ""
  dry_run: false

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "rf-worldpose-transformer"
  run_name: "transformer-v1"
"""
    config_path.write_text(config_content.strip())
    log.info(f"Default config written to {config_path}")


if __name__ == "__main__":
    _write_default_config()
    train()
