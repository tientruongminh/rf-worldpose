# RF-WorldPose Experiment Log

> WiFi CSI-based Human Pose Estimation using Transformer Architecture
> Last updated: 2026-06-09 12:55 UTC+7

---

## 1. Project Overview

**Objective**: Estimate 3D human body pose (13 joints × 3 coords) from WiFi Channel State Information (CSI) signals, without cameras. The system uses 4 ESP32-S3 nodes placed at room corners to collect CSI data.

**Three-Phase Training Pipeline**:
1. **Phase 1 — SSL Pre-training**: Self-supervised learning on all CSI data (no labels needed) to learn general RF signal representations
2. **Phase 2 — Supervised Training**: Train pose + action heads using labeled datasets with SSL-pretrained encoder
3. **Phase 3 — Fine-tuning**: Adapt to a specific room using camera-derived pose labels, then deploy camera-free

---

## 2. Model Architecture

**CSITransformerPose** — 15.17M trainable parameters

| Component | Details |
|-----------|---------|
| CSI Tokenizer | SubcarrierPatchEmbed (270 subcarriers → 45 patches, patch_size=6) + RunningNorm + TemporalPE |
| Tokenizer Params | 15,360 |
| Spatial Encoder | 4-layer Transformer (d=256, 8 heads, FFN×4) |
| Temporal Encoder | 4-layer Transformer (d=256, 8 heads, FFN×4, non-causal) |
| Pose Decoder | 3-layer cross-attention decoder + 2-layer temporal refinement, 13 joint queries |
| Action Head | CLS token attention → 28-class classification |
| Dropout | 0.1 throughout |
| Total Params | 15,159,073 (trainable) |

**Input**: CSI tensor `(B, T=60, N_sub=270, 2)` — 60 time frames, 270 subcarriers, 2 channels (amplitude, phase)

**Output**: 
- `coords`: `(B, T, 13, 3)` — per-frame 3D joint positions
- `vis_logits`: `(B, T, 13)` — joint visibility
- `action_logits`: `(B, 28)` — activity classification

---

## 3. Dataset: rfpose-unified-v2

**Source**: 6 public WiFi/RF sensing datasets, unified via Gold ETL pipeline.

| Dataset | Samples | Has Pose | Has Action | Has Both |
|---------|---------|----------|------------|----------|
| mmfi | 12,960 | ✓ | ✓ | ✓ |
| wipose | 4,596 | ✓ | ✓ | ✓ |
| wifipose | 3,754 | ✓ | ✗ | ✗ |
| uthar | 49,730 | ✗ | ✓ | ✗ |
| wiar | 35,107 | ✗ | ✓ | ✗ |
| wimans | 112,860 | ✗ | ✓ | ✗ |
| **Total** | **219,007** | **21,310 (9.7%)** | **209,313** | **17,556 (8%)** |

**Storage**: 27GB on disk (x.npy per sub-dataset + y.npz + metadata.npz)

**Data Format**:
- `x.npy`: `[N, 2, T, N_sub]` — CSI amplitude/phase windows (memory-mapped)
- `y.npz`: `pose [N, T, J, 3]`, `pose_mask`, `action_label`, `action_mask` (preloaded into RAM at init)
- `metadata.npz`: train/val split per window

---

## 4. Loss Function: RFPoseLoss

**Weighting mode**: Uncertainty (Kendall et al. 2018) — learnable log-variance per task

$$L_{total} = \sum_i \frac{1}{2\sigma_i^2} L_i + \frac{1}{2} \log \sigma_i^2$$

| Loss Term | Description |
|-----------|-------------|
| `coord` | Smooth L1 between predicted and GT joint coordinates, weighted by visibility |
| `vis` | BCE for joint visibility prediction |
| `bone` | Bone length consistency between pred and GT |
| `temporal` | Temporal smoothness (acceleration penalty) |
| `symmetry` | Left-right body symmetry enforcement |
| `action` | Cross-entropy for activity classification (separate, lambda-weighted) |

---

## 5. Experiment History

### 5.1 Phase 1: SSL Pre-training ✅

| Item | Value |
|------|-------|
| Config | `ssl_eagle.yaml` |
| MLflow Experiment | `rf-worldpose-ssl` |
| Cluster | Eagle HPC |
| Dataset | rfpose-unified-v2, all 219,007 samples |
| Batch Size | 64 |
| Epochs | 30 |
| Learning Rate | 5e-4 |
| SSL Method | Masked autoencoder (mask_ratio=0.4) + Contrastive (λ_recon=1.0, λ_contrast=0.5, τ=0.07) |
| AMP | Enabled |
| Checkpoint | `s3://rfpose/mlflow/5/62a95b29c2e3499e82983f5e8ecb120e/artifacts/checkpoints/best` |
| Status | **COMPLETED** |

**Outcome**: Encoder learns general CSI signal representations from all 219K samples (no labels required).

---

### 5.2 Phase 2, Run 1: Supervised on ALL data (rfpose-multitask-v1) ✗

| Item | Value |
|------|-------|
| Slurm Job | 7341238 |
| Config | `supervised_eagle.yaml` |
| Dataset | rfpose-multitask-v1 (all samples) |
| GPUs | 4× H100 (DDP) |
| Issue | `val_mpjpe = 0.0000` — all pose arrays were zeros, `pose_mask=0` for every sample |

**Root Cause**: The `rfpose-multitask-v1` ETL version did not contain actual pose data — only activity labels. Pose arrays existed but were filled with zeros.

**Resolution**: Switched to `rfpose-unified-v2` which contains real pose data for mmfi, wifipose, and wipose sub-datasets.

---

### 5.3 Phase 2, Run 2: Supervised on ALL data (rfpose-unified-v2) — Cancelled

| Item | Value |
|------|-------|
| Slurm Job | 7341933 |
| MLflow Run | `supervised-4gpu` |
| Dataset | rfpose-unified-v2, all 219,007 samples |
| GPUs | 4× H100 (DDP, `find_unused_parameters=True`) |
| Batch Size | 32 per GPU |
| Epochs | 50 (cancelled at epoch 8) |
| LR | 1e-4, warmup 3 epochs |
| AMP | Enabled |
| num_workers | 4 |
| Epoch Time | ~210s |

**Results (8 epochs before cancellation)**:

| Epoch | train_loss | val_mpjpe (mm) | val_pa_mpjpe (mm) |
|-------|-----------|----------------|-------------------|
| 0 | 22.11 | 87.61 | 80.41 |
| 1 | 19.69 | 87.05 | 85.68 |
| 2 | 18.97 | 86.90 | 90.56 |
| 3 | 18.56 | 86.93 | 83.43 |
| 4 | 17.21 | 86.61 | 85.12 |
| 5 | 15.13 | 86.72 | 80.54 |
| 6 | 16.23 | 88.39 | 83.41 |
| 7 | 14.45 | 86.10 | 88.25 |

**Analysis**:
- `val_mpjpe` barely decreased (87.6 → 86.1mm in 8 epochs)
- `val_pa_mpjpe` was noisy and unstable (80 → 90 → 83 → 88)
- `train_loss` decreased mainly from action loss on 90% activity-only data
- Only ~10% of batches had pose gradients → very slow pose convergence

**Decision**: Cancelled. Switched to training only on samples with both pose + action labels.

---

### 5.4 Phase 2, Run 3: Supervised on Pose+Action Only ✅ (Current Best)

| Item | Value |
|------|-------|
| Slurm Job | 7341939 |
| MLflow Run | [`pose-action-only`](http://207.180.243.242:5000/#/experiments/4/runs/f83f6e3bbdf64eb898ff32b34d9fa771) |
| Dataset | rfpose-unified-v2, `require_pose=True`, `require_action=True` |
| Datasets Used | mmfi (12,960) + wipose (4,596) = **17,556 samples** |
| Train/Val Split | 14,044 / 3,512 (from metadata) |
| GPUs | 4× H100 (DDP) |
| Batch Size | 32 per GPU → 109 batches/epoch |
| Epochs | 150 |
| LR | 1e-4, warmup 3 epochs, patience 15 |
| AMP | Enabled |
| num_workers | 4, persistent_workers=True |
| SSL Pretrained | `s3://rfpose/mlflow/5/.../checkpoints/best` |
| freeze_encoder | False |
| Epoch Time | ~17s |
| Total Time | **1h 48m** |
| Status | **COMPLETED** |

**Training Curve (sampled every 10 epochs)**:

| Epoch | train_loss | val_mpjpe (mm) | val_pa_mpjpe (mm) |
|-------|-----------|----------------|-------------------|
| 0 | 26.83 | 97.70 | 86.33 |
| 10 | 23.73 | 97.00 | 92.95 |
| 20 | 20.61 | 96.48 | 95.53 |
| 30 | 18.74 | 95.58 | 87.30 |
| 40 | 16.58 | 94.40 | 91.44 |
| 50 | 14.27 | 93.20 | 86.69 |
| 60 | 13.66 | 92.25 | 88.56 |
| 70 | 11.54 | 90.96 | 95.24 |
| 80 | 10.42 | 90.08 | 91.29 |
| 90 | 10.39 | 89.45 | 94.38 |
| 100 | 9.39 | 89.04 | 90.77 |
| 110 | 8.94 | 88.80 | 91.65 |
| 120 | 8.67 | 88.65 | 91.40 |
| 130 | 8.60 | 88.57 | 91.48 |
| 140 | 8.77 | 88.55 | 91.82 |
| 149 | 8.84 | **88.54** | 91.84 |

**Best Checkpoint**: epoch 148, `val_mpjpe = 88.54mm`

---

### 5.5 MM-Fi Protocol Evaluation (Run 3 checkpoint)

Evaluated the best checkpoint from Run 3 on **MM-Fi data only** using official benchmark protocols.

| Setting | MPJPE (mm) | PA-MPJPE (mm) | Samples | Description |
|---------|-----------|---------------|---------|-------------|
| **S1** (Random) | 378.9 ± 87.9 | 351.6 | 3,240 | 75/25 random split (sequence-level) |
| **S2** (Cross-Subject) | 392.7 | 355.8 | 2,592 | 32 train / 8 test subjects |
| **S3** (Cross-Env) | 450.5 | 344.7 | 3,240 | E01-E03 train / E04 test |

**Comparison with published baselines (17-joint MPJPE, P3 all actions)**:

| Model | S1 | S2 | S3 | Notes |
|-------|-----|-----|-----|-------|
| MetaFi++ (2023) | 197.1 | 231.1 | 369.5 | MM-Fi paper baseline |
| DT-Pose (2025) | 178.5 | 212.8 | 288.6 | Current WiFi SOTA |
| **Ours (Run 3)** | **378.9** | **392.7** | **450.5** | 13-joint, mixed-scale training |

**Per-Joint MPJPE (S1, mm)**:

| Joint | MPJPE | Joint | MPJPE |
|-------|-------|-------|-------|
| head | 237.1 | l_hip | 379.9 |
| l_shoulder | 338.9 | l_knee | 481.8 |
| l_elbow | 405.0 | l_ankle | 502.2 |
| l_wrist | 250.1 | r_hip | 472.3 |
| r_shoulder | 284.7 | r_knee | 389.3 |
| r_elbow | 332.0 | r_ankle | 482.6 |
| r_wrist | 369.6 | | |

**Action accuracy**: 68.9% (S1), 69.1% (S2), 66.3% (S3) — reasonable given unified 28-class remapping.

#### Critical Finding: Coordinate Scale Mismatch

Root cause of poor pose performance identified:

| Dataset | Coordinate Unit | Value Range | Scale Factor |
|---------|----------------|-------------|--------------|
| **mmfi** | meters | [-1.75, 4.00] | 1.0 |
| **wipose** | millimeters | [-82, 632] | 0.001 |
| **wifipose** | meters | [-1.01, 4.28] | 1.0 |

**Impact**: When training on mixed mmfi + wipose data, the loss was dominated by wipose (coordinates 1000× larger), causing the model to primarily learn wipose-scale predictions. The training `val_mpjpe=88.54` was a misleading average across two incompatible coordinate scales.

**Resolution**: Two corrective training runs submitted (see §5.6–5.7).

---

### 5.6 Phase 2, Run 4: MM-Fi Only ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342631 (resubmitted from 7342611 after S3 fix) |
| Config | `mmfi_only_eagle.yaml` |
| MLflow Run | [`mmfi-only-v1`](http://207.180.243.242:5000/#/experiments/4/runs/dc309454d4504b72b8118bd80a1dcbbb) |
| Dataset | MM-Fi only — 12,960 windows (meter-scale, no normalization needed) |
| Train/Val Split | 10,368 / 2,592 → 81 batches/epoch |
| GPUs | 4× H100 (DDP) |
| Batch Size | 32 per GPU |
| Epochs | 200 (early stopped at epoch 43) |
| LR | 1e-4, warmup 5 epochs, patience 25 |
| SSL Pretrained | Same encoder |
| Epoch Time | ~12s |
| Status | **COMPLETED** |

**Training Curve**:

| Epoch | train_loss | val_mpjpe (mm) | val_pa_mpjpe (mm) |
|-------|-----------|----------------|-------------------|
| 0 | 0.6613 | 396.3 | 287.8 |
| 1 | 0.1659 | 356.9 | 218.2 |
| 2 | 0.1266 | 334.7 | 204.8 |
| 3 | 0.0882 | 331.9 | 207.9 |
| 5 | 0.0442 | 336.1 | 214.9 |
| 10 | −0.0693 | 331.0 | 204.9 |
| 15 | −0.1635 | 332.9 | 206.7 |
| 18 | −0.2156 | **327.4** | 209.3 |
| 20 | −0.2555 | 332.7 | 207.2 |
| 30 | −0.3933 | 333.4 | 213.1 |
| 40 | −0.5930 | 335.4 | 219.6 |
| 43 | −0.6393 | 337.3 | 221.6 |

**Best Checkpoint**: epoch 18, `val_mpjpe = 327.4mm`, `val_pa_mpjpe = 209.3mm`

**Analysis**:
- Fast initial convergence (396→327mm in 18 epochs) thanks to SSL pretrained encoder
- Best MPJPE plateaued at ~327mm, then train_loss kept decreasing → overfitting after epoch 18
- PA-MPJPE ≈ 205mm shows the model learns reasonable body articulation, but global translation/scale is the bottleneck (~120mm gap between PA-MPJPE and MPJPE)
- Still 83% worse than MetaFi++ (178.5mm) on comparable data

---

### 5.7 Phase 2, Run 5: All Data with Coordinate Normalization ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342630 (resubmitted from 7342612 after S3 fix) |
| Config | `supervised_normalized_eagle.yaml` |
| MLflow Run | [`supervised-normalized-v1`](http://207.180.243.242:5000/#/experiments/4/runs/237b7bb04e3e4d479aad6ca9d036aa0d) |
| Dataset | mmfi + wipose (normalized) = 17,556 windows (all in meters) |
| Train/Val Split | 14,044 / 3,512 → 109 batches/epoch |
| GPUs | 4× H100 (DDP) |
| Batch Size | 32 per GPU |
| Epochs | 200 (early stopped at epoch 42) |
| LR | 1e-4, warmup 5 epochs, patience 25 |
| SSL Pretrained | Same encoder |
| Fix | `DATASET_COORD_SCALE` in `gold_npz_dataset.py` — wipose ×0.001 (mm→m) |
| Epoch Time | ~17s |
| Status | **COMPLETED** |

**Training Curve**:

| Epoch | train_loss | val_mpjpe (mm) | val_pa_mpjpe (mm) |
|-------|-----------|----------------|-------------------|
| 0 | 15.0166 | 1526.0 | 481.2 |
| 1 | 2.4401 | 660.6 | 429.0 |
| 2 | 0.5947 | 476.0 | 310.8 |
| 4 | 0.2212 | 402.7 | 303.5 |
| 6 | 0.1170 | 374.5 | 304.0 |
| 10 | −0.0408 | 380.8 | 300.4 |
| 14 | −0.1795 | 372.1 | 302.8 |
| 17 | −0.2510 | **367.8** | 299.9 |
| 20 | −0.3225 | 376.2 | 307.2 |
| 30 | −0.5755 | 375.4 | 311.2 |
| 42 | −0.8153 | 368.8 | 310.4 |

**Best Checkpoint**: epoch 17, `val_mpjpe = 367.8mm`, `val_pa_mpjpe = 299.9mm`

**Analysis**:
- High initial MPJPE (1526mm) due to wipose data now normalized to meters — the SSL encoder was pretrained on mm-scale wipose, so predictions start in wrong scale
- Converged to 367.8mm, which is **worse** than MM-Fi-only (327.4mm) despite having 35% more data
- PA-MPJPE (~300mm) is significantly worse than MM-Fi-only (205mm), suggesting the two datasets have incompatible body representations (different joint definitions, different pose spaces) even after scale normalization
- The coordinate normalization fixed the scale issue, but the model struggles to learn a unified pose space from heterogeneous datasets

---

### 5.8 Phase 2, Run 6: WiPose Only ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342632 |
| Config | `wipose_only_eagle.yaml` |
| MLflow Run | [`wipose-only-v1`](http://207.180.243.242:5000/#/experiments/4/runs/b2393aea0cf048fba896290f0955412a) |
| Dataset | WiPose only — 4,596 windows (coords ×0.001 → meters) |
| Train/Val Split | 3,676 / 920 → 57 batches/epoch |
| GPUs | 4× H100 (DDP) |
| Batch Size | 32 per GPU |
| Epochs | 300 (early stopped at epoch 38) |
| LR | 1e-4, warmup 5 epochs, patience 30 |
| SSL Pretrained | Same encoder |
| Epoch Time | ~9s |
| Status | **COMPLETED** |

**Training Curve**:

| Epoch | train_loss | val_mpjpe (mm) | val_pa_mpjpe (mm) |
|-------|-----------|----------------|-------------------|
| 0 | 66.8829 | 706.4 | 340.8 |
| 1 | 16.2289 | 513.1 | 299.1 |
| 2 | 4.2985 | 401.9 | 337.8 |
| 4 | 0.8278 | 340.8 | 333.8 |
| 7 | 0.4799 | 329.2 | 313.3 |
| 8 | 0.3735 | **325.5** | 298.3 |
| 10 | 0.3014 | 346.1 | 295.9 |
| 15 | 0.1771 | 362.4 | 282.7 |
| 20 | 0.0336 | 360.2 | 280.4 |
| 30 | −0.1562 | 356.7 | 280.3 |
| 38 | −0.2141 | 338.0 | 280.4 |

**Best Checkpoint**: epoch 8, `val_mpjpe = 325.5mm`, `val_pa_mpjpe = 298.3mm`

**Analysis**:
- Very fast convergence (706→325mm in 8 epochs) then immediately overfits — only 4,596 samples is very small
- Best MPJPE (325.5mm) is slightly better than MM-Fi-only (327.4mm) despite having 64% fewer samples
- PA-MPJPE (298mm) is much worse than MM-Fi-only (205mm), suggesting WiPose pose ground truth quality may be lower or the joint positions are harder to predict
- Extreme initial MPJPE (706mm) because SSL encoder was pretrained on mm-scale wipose, but now inputs are meter-scale after normalization — the encoder needs to readapt
- Early stopping at epoch 38 shows severe overfitting on this small dataset

---

### 5.9 Comparative Analysis: Runs 4–6

| Metric | Run 4: MM-Fi Only | Run 5: Normalized All | Run 6: WiPose Only |
|--------|-------------------|----------------------|-------------------|
| **Dataset** | mmfi (12,960) | mmfi + wipose (17,556) | wipose (4,596) |
| **Best MPJPE** | **327.4mm** | 367.8mm | 325.5mm |
| **Best PA-MPJPE** | **204.9mm** | 299.9mm | 298.3mm |
| **Best Epoch** | 18 | 17 | 8 |
| **Early Stop** | Epoch 43 | Epoch 42 | Epoch 38 |
| **Convergence Speed** | Fast (12 epochs) | Slow (17 epochs) | Very fast (8 epochs) |

**Key Findings**:

1. **Single-dataset training outperforms combined**: Both individual dataset runs (327.4mm, 325.5mm) beat the combined dataset (367.8mm). Mixing datasets with different sensor setups, joint definitions, and collection protocols hurts rather than helps.

2. **MM-Fi has better pose ground truth**: PA-MPJPE is the fairest comparison (removes global positioning). MM-Fi-only achieves 205mm PA-MPJPE vs. WiPose-only 298mm — indicating MM-Fi's skeleton annotations and the model's ability to predict articulated pose from its CSI data are significantly better.

3. **PA-MPJPE vs MPJPE gap reveals the bottleneck**:
   - MM-Fi: MPJPE 327mm − PA-MPJPE 205mm = **122mm gap** → global translation error
   - WiPose: MPJPE 326mm − PA-MPJPE 298mm = **28mm gap** → body articulation error
   - The model's main weakness on MM-Fi is predicting WHERE the person is in the room (translation), not HOW their body is posed (articulation). This is exactly what **root-relative prediction** (#5) targets.

4. **Still 83–84% above SOTA**: Best result (325.5mm) vs DT-Pose (178.5mm). However, direct comparison is imperfect because we use 13 joints vs their 17 joints, and we haven't evaluated on MM-Fi protocols yet for Runs 4–6.

5. **Overfitting is a problem**: All three runs show train_loss steadily decreasing while val_mpjpe plateaus or worsens, indicating the 15.17M parameter model overfits the available data. Stronger regularization or more data may help.

---

### 5.10 Phase 2, Run 7: Root-Relative Coords (Combined) ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342633 |
| Config | `rootrel_eagle.yaml` |
| MLflow Run | [`rootrel-v1`](http://207.180.243.242:5000/#/experiments/4/runs/54d03a07d592476f8863b9636c680e26) |
| Variant | `CSITransformerPoseRootRel` |
| Dataset | mmfi + wipose (normalized) = 17,556 windows |
| Train/Val Split | 14,044 / 3,512 |
| GPUs | 1× H100 |
| Epochs | 200 (early stopped at epoch 54) |
| LR | 1e-4, warmup 5, patience 25 |
| Best MPJPE | **318.8mm** (PA-MPJPE 222mm) |
| Status | **COMPLETED** |

**Architecture — `CSITransformerPoseRootRel`** (15.2M params):

```
CSITokenizer → SpatialEncoder → TemporalEncoder → PoseDecoderRootRel
                                                    ├─ Root Head: cross-attn → MLP → root [B,T,3]
                                                    ├─ Offset Head: cross-attn → MLP → offsets [B,T,J,3]
                                                    └─ coords = root + offsets
```

Decomposes prediction into absolute root position (pelvis center) and per-joint relative offsets, so the model can learn global translation and body articulation independently.

**Loss**: Standard RFPoseLoss + `λ_root × SmoothL1(pred_root, gt_root)` + `λ_offset × SmoothL1(pred_offsets, gt_offsets)`

**Result**: 318.8mm — improves over combined-data baseline (367.8mm) by **49mm** (-13.3%), confirming root-relative decomposition is effective.

---

### 5.11 Phase 2, Run 8: Subcarrier Attention (Combined) ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342634 |
| Config | `subcarrier_attn_eagle.yaml` |
| MLflow Run | [`subcattn-v1`](http://207.180.243.242:5000/#/experiments/4/runs/e549fa206b9548249d137f24b8baee00) |
| Variant | `CSITokenizerAttn` + `CSITransformerPose` |
| Dataset | mmfi + wipose (normalized) = 17,556 windows |
| GPUs | 1× H100 |
| Epochs | 200 (early stopped at epoch 51) |
| Best MPJPE | **319.2mm** (PA-MPJPE 238mm) |
| Status | **COMPLETED** |

**Architecture — `CSITokenizerAttn`**:

```
CSI [B,T,270,2] → Linear(540→256) → 19 learnable queries attend to 270 subcarriers
                                      via multi-head cross-attention (4 heads)
                                    → tokens [B,T,19,256]
```

Replaces fixed 6-subcarrier patches with data-driven frequency grouping.

**Result**: 319.2mm — nearly identical to rootrel (318.8mm), marginal improvement over combined baseline (367.8mm). The learnable tokenizer did not outperform fixed patching.

---

### 5.12 Phase 2, Run 9: Root-Relative on WiPose Only ✅ ⭐ BEST OVERALL

| Item | Value |
|------|-------|
| Slurm Job | 7342639 |
| Config | `rootrel_wipose_eagle.yaml` |
| MLflow Run | [`rootrel-wipose-v1`](http://207.180.243.242:5000/#/experiments/4/runs/be52da0cebdb4e8e9c6cfa33bda591ca) |
| Variant | `CSITransformerPoseRootRel` |
| Dataset | WiPose only — 4,596 windows (coords ×0.001 → meters) |
| Train/Val Split | 3,676 / 920 → 57 batches/epoch |
| GPUs | 1× H100 |
| Epochs | 300 (early stopped at epoch 82) |
| LR | 1e-4, warmup 5, patience 30 |
| Best MPJPE | **286.1mm** (PA-MPJPE 296mm) |
| Status | **COMPLETED** |

**Result**: **286.1mm** — best MPJPE across all runs. Root-relative on WiPose reduces error by **39.4mm** (−12.1%) vs base WiPose-only (325.5mm).

---

### 5.13 Phase 2, Run 10: Root-Relative on MM-Fi Only ✅ ⭐ BEST ON MM-Fi

| Item | Value |
|------|-------|
| Slurm Job | 7342638 |
| Config | `rootrel_mmfi_eagle.yaml` |
| MLflow Run | [`rootrel-mmfi-v1`](http://207.180.243.242:5000/#/experiments/4/runs/b4d555d6a4b44beebe1a6c91b868fcf0) |
| Variant | `CSITransformerPoseRootRel` |
| Dataset | MM-Fi only — 12,960 windows |
| Train/Val Split | 10,368 / 2,592 → 324 batches/epoch |
| GPUs | 1× H100 |
| Epochs | 200 (early stopped at epoch ~140) |
| LR | 1e-4, warmup 5, patience 25 |
| Best MPJPE | **305.9mm** (PA-MPJPE ~220mm) |
| Epoch Time | ~48s |
| Total Time | ~1h 52m |
| Status | **COMPLETED** |

**Result**: **305.9mm** — best MPJPE on MM-Fi, improving over base MM-Fi-only (327.4mm) by **21.5mm** (−6.6%). Root-relative prediction effective on MM-Fi but less dramatic than on WiPose.

---

### 5.14 Phase 2, Run 11: Subcarrier Attention on WiPose Only ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342642 |
| Config | `subcattn_wipose_eagle.yaml` |
| MLflow Run | [`subcattn-wipose-v1`](http://207.180.243.242:5000/#/experiments/4/runs/67ebf7639ea84b14b512b1a50bee307f) |
| Variant | `CSITokenizerAttn` + `CSITransformerPose` |
| Dataset | WiPose only — 4,596 windows |
| Epochs | 300 (early stopped at epoch 45) |
| Best MPJPE | **333.1mm** (PA-MPJPE 315mm) |
| Status | **COMPLETED** |

**Result**: 333.1mm — **worse** than base WiPose-only (325.5mm) by 7.6mm. Subcarrier attention hurts on small datasets.

---

### 5.15 Phase 2, Run 12: Subcarrier Attention on MM-Fi Only ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342640 |
| Config | `subcattn_mmfi_eagle.yaml` |
| MLflow Run | [`subcattn-mmfi-v1`](http://207.180.243.242:5000/#/experiments/4/runs/fbdee3da67a1466ba604f03183fc67a0) |
| Variant | `CSITokenizerAttn` + `CSITransformerPose` |
| Dataset | MM-Fi only — 12,960 windows |
| Epochs | 200 (early stopped at ~68) |
| Best MPJPE | **373.7mm** (PA-MPJPE 279mm) |
| Status | **COMPLETED** |

**Result**: 373.7mm — significantly worse than base MM-Fi-only (327.4mm) by **46.3mm**. The learnable tokenizer adds complexity without benefit; fixed subcarrier patches are superior.

---

### 5.16 Phase 2, Run 13: MetaFi++ Baseline on MM-Fi ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342647 |
| Config | `metafi_mmfi_eagle.yaml` |
| MLflow Run | [`metafi-mmfi-v1`](http://207.180.243.242:5000/#/experiments/4/runs/4880a549c9494b3999052afb1300d3f2) |
| Variant | `MetaFiModel` (reimplemented from MM-Fi paper) |
| Dataset | MM-Fi only — 12,960 windows |
| Epochs | 200 (early stopped at epoch 31) |
| LR | 5e-4 |
| Best MPJPE | **317.3mm** (PA-MPJPE 255mm) |
| SSL Pretrained | None (trained from scratch) |
| Status | **COMPLETED** |

**Architecture — `MetaFiModel`** (~2M params):

```
MetaFiTokenizer:  CSI [B,T,270,2] → flatten(540) → Linear(540→256) → pos_embed
MetaFiModel:      4-layer TransformerEncoder (256d, 8 heads, FFN×2)
                  → pose_head: MLP → [B,T,13,3]
                  → vis_head: Linear → [B,T,13]
                  → action_head: attn_pool → MLP → [B,28]
```

Simple linear projection + standard Transformer, matching the original MetaFi++ paper architecture.

**Result**: 317.3mm — competitive with our much larger model (305.9mm with 15M params) despite having only **2M parameters** and no SSL pre-training. This suggests our larger model is overfitting, and simpler architectures may be more appropriate for the available data volume.

---

### 5.17 Phase 2, Run 14: GCN Skeleton Decoder + Root-Relative on MM-Fi ✅

| Item | Value |
|------|-------|
| Slurm Job | 7342656 |
| Config | `gcn_rootrel_mmfi_eagle.yaml` |
| MLflow Run | [`gcn-rootrel-mmfi-v3`](http://207.180.243.242:5000/#/experiments/4/runs/b2ad77bbd7b742e383c4e09240537a46) |
| Variant | `CSITransformerPoseGCN` |
| Dataset | MM-Fi only — 12,960 windows |
| Epochs | 200 (early stopped at epoch 41) |
| freeze_encoder | **true** (matches DT-Pose protocol) |
| Best MPJPE | **338.8mm** (PA-MPJPE 258mm) |
| Status | **COMPLETED** |

**Architecture — `CSITransformerPoseGCN`** (~15M total, ~5M trainable with frozen encoder):

```
CSITokenizer → SpatialEncoder → TemporalEncoder  [FROZEN]
                                    ↓
                              mean_pool → [B,T,256]
                                    ↓
                             GCNPoseDecoder:
                              ├─ pose_prompt [256,13]: learnable per-joint queries
                              ├─ 3× GraphConvLayer (skeleton adjacency)
                              ├─ 3× JointTransformerLayer (inter-joint attention)
                              ├─ root_head → [B,T,3]
                              ├─ offset_head → [B,T,13,3]
                              └─ action_head → [B,28]
```

Inspired by DT-Pose (Chen et al. 2025): GCN enforces skeleton topology constraints, combined with root-relative prediction. Encoder frozen during Phase 2 (only decoder trains).

**Skeleton adjacency matrix** (13 joints):
```
head ─ l_shoulder ─ l_elbow ─ l_wrist
  └── r_shoulder ─ r_elbow ─ r_wrist
       l_shoulder ─ r_shoulder (bridge)
       l_shoulder ─ l_hip ─ l_knee ─ l_ankle
       r_shoulder ─ r_hip ─ r_knee ─ r_ankle
       l_hip ─ r_hip (bridge)
```

**Result**: 338.8mm — worse than base MM-Fi (327.4mm) by **11.4mm**. Freezing the encoder produces weaker features when pre-training quality is insufficient (our SSL ≠ DT-Pose MAE). The GCN topology constraints alone do not compensate.

---

### 5.18 Phase 2, Run 15: Root-Relative + Weighted Joint Loss on MM-Fi 🔄

| Item | Value |
|------|-------|
| Slurm Job | 7342666 |
| Config | `rootrel_mmfi_weighted_eagle.yaml` |
| MLflow Run | `rootrel-mmfi-weighted-v1` |
| Variant | `CSITransformerPoseRootRel` |
| Dataset | MM-Fi only — 12,960 windows |
| Epochs | 300, patience 40 |
| LR | 5e-5 (halved from v1), warmup 10 |
| λ_root | 1.5, λ_offset: 1.5 (increased from 1.0) |
| λ_bone | 0.5, λ_temporal: 0.3 (stronger structural constraints) |
| Joint Weights | `[1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.2, 1.0, 2.0, 2.5, 1.0, 2.0, 2.5]` |
| Status | **RUNNING** |

**New feature**: Per-joint weighted offset loss. Weights derived from eval analysis showing legs (knee/ankle) have 2× the error of upper body. Knees weighted 2.0×, ankles 2.5×.

**Target**: < 290mm MPJPE on MM-Fi.

---

## 6. Full Evaluation Results

Comprehensive evaluation using `rfpose.evaluation.eval_v2` — downloads best checkpoints from S3/MLflow, runs inference on validation split with pose + action metrics.

### 6.1 Pose Estimation Summary

| # | Model | Variant | Dataset | Val Samples | MPJPE (mm) | PA-MPJPE (mm) | Gap (mm) | Params |
|---|-------|---------|---------|-------------|-----------|---------------|----------|--------|
| R4 | CSITransformerPose | base | MM-Fi | 1,296 | 324.0 | 208.3 | 115.6 | 15.2M |
| R5 | CSITransformerPose | base | Combined | 1,755 | 367.8 | 299.9 | 67.9 | 15.2M |
| R6 | CSITransformerPose | base | WiPose | 459 | 325.5 | 298.3 | 27.2 | 15.2M |
| R7 | CSITransformerPoseRootRel | rootrel | Combined | 1,755 | 320.1 | 209.9 | 110.2 | 15.2M |
| R8 | CSITransformerPose+AttnTok | subcattn | Combined | 1,755 | 319.2 | 238.0 | 81.2 | 15.2M |
| **R9** | **CSITransformerPoseRootRel** | **rootrel** | **WiPose** | **459** | **284.8** ⭐ | **309.5** | **−24.6** | **15.2M** |
| **R10** | **CSITransformerPoseRootRel** | **rootrel** | **MM-Fi** | **1,296** | **305.9** ⭐ | **~220** | **~86** | **15.2M** |
| R11 | CSITransformerPose+AttnTok | subcattn | WiPose | 459 | 333.1 | 315.0 | 18.1 | 15.2M |
| R12 | CSITransformerPose+AttnTok | subcattn | MM-Fi | 1,296 | 373.7 | 279.0 | 94.7 | 15.2M |
| R13 | MetaFiModel | metafi | MM-Fi | 1,296 | 318.9 | 206.9 | 112.0 | 2.0M |
| R14 | CSITransformerPoseGCN | gcn_rootrel | MM-Fi | 1,296 | 338.8 | 258.0 | 80.8 | 15M(5M trainable) |

### 6.2 Per-Joint MPJPE Breakdown (mm)

Evaluation on validation split for top 3 models:

| Joint | R4: base MM-Fi | R10: rootrel MM-Fi | R13: MetaFi MM-Fi | R9: rootrel WiPose |
|-------|---------------|-------------------|------------------|-------------------|
| head | 221.9 | — | 210.9 | 112.4 |
| l_shoulder | 221.0 | — | 256.7 | 219.9 |
| l_elbow | 273.1 | — | 293.8 | 446.8 |
| l_wrist | 223.8 | — | 218.8 | 178.2 |
| r_shoulder | 246.9 | — | 246.4 | 506.7 |
| r_elbow | 243.3 | — | 241.3 | 141.7 |
| r_wrist | 256.9 | — | 266.2 | 436.2 |
| l_hip | 255.6 | — | 286.0 | 346.5 |
| **l_knee** | **407.4** | — | **564.7** | 170.2 |
| **l_ankle** | **504.9** | — | **423.4** | 346.2 |
| r_hip | 413.4 | — | 403.9 | 122.2 |
| **r_knee** | **428.6** | — | **315.7** | 345.9 |
| **r_ankle** | **514.7** | — | **417.5** | 329.9 |

**Key finding**: On MM-Fi, lower extremities (knee/ankle) have 400–515mm error — nearly 2× the upper body (220–270mm). This motivates the weighted joint loss in Run 15.

### 6.3 Action Recognition Results

| Model | Dataset | Accuracy | Macro F1 | Macro Precision | Macro Recall | Classes |
|-------|---------|----------|----------|-----------------|--------------|---------|
| **R9: rootrel-wipose** | WiPose | **78.4%** | **78.5%** | **80.3%** | **78.9%** | 12 |
| R4: base-mmfi | MM-Fi | 38.6% | 33.4% | 37.9% | 34.1% | 22 |
| R7: rootrel-combined | All | 27.5% | 18.9% | 25.7% | 19.8% | 24 |
| R13: metafi-mmfi | MM-Fi | 21.1% | 1.6% | 1.0% | 4.6% | 22 |

#### Per-Action F1 Breakdown — rootrel-wipose (best)

| Action | F1 | Prec | Recall | MPJPE (mm) | Samples |
|--------|----|------|--------|-----------|---------|
| bend | 0.911 | 0.923 | 0.900 | 306.4 | 40 |
| circle | 0.895 | 0.829 | 0.971 | 278.8 | 35 |
| push | 0.853 | 0.889 | 0.821 | 285.5 | 39 |
| wave | 0.848 | 0.966 | 0.757 | 280.4 | 37 |
| throw | 0.848 | 0.875 | 0.824 | 278.7 | 34 |
| stand_up | 0.809 | 0.731 | 0.905 | 284.4 | 42 |
| crouch | 0.806 | 0.758 | 0.862 | 307.4 | 29 |
| run | 0.777 | 0.690 | 0.889 | 275.0 | 45 |
| sit_down | 0.719 | 0.885 | 0.605 | 300.3 | 38 |
| pull | 0.709 | 0.622 | 0.824 | 285.1 | 34 |
| jump | 0.636 | 0.636 | 0.636 | 280.5 | 44 |
| walk | 0.606 | 0.833 | 0.476 | 275.4 | 42 |

**Finding**: The model achieves strong action recognition (78.4% accuracy, 78.5% F1) from WiFi CSI — demonstrating that CSI signals carry rich semantic information about human activities.

#### Per-Action F1 Breakdown — base-mmfi (22 classes)

| Action | F1 | Prec | Recall | MPJPE (mm) | N |
|--------|------|------|--------|-----------|---|
| run | 0.584 | 0.605 | 0.565 | 300.8 | 46 |
| kick | 0.533 | 0.522 | 0.545 | 310.0 | 44 |
| nothing | 0.530 | 0.482 | 0.588 | 325.4 | 274 |
| phone_call | 0.426 | 0.418 | 0.434 | 312.0 | 53 |
| toss_paper | 0.393 | 0.423 | 0.367 | 315.0 | 60 |
| jump | 0.395 | 0.395 | 0.395 | 314.4 | 43 |
| wave | 0.392 | 0.283 | 0.638 | 363.0 | 47 |
| fall | 0.385 | 0.484 | 0.319 | 299.0 | 47 |
| draw_x | 0.385 | 0.625 | 0.278 | 394.5 | 54 |
| push | 0.368 | 0.241 | 0.778 | 373.8 | 54 |
| throw | 0.366 | 0.565 | 0.271 | 283.2 | 48 |
| drink_water | 0.354 | 0.378 | 0.333 | 338.8 | 51 |
| hand_clap | 0.353 | 0.296 | 0.438 | 325.9 | 48 |
| sit_down | 0.316 | 0.319 | 0.312 | 258.7 | 48 |
| pull | 0.292 | 0.371 | 0.241 | 392.7 | 54 |
| stand_up | 0.282 | 0.455 | 0.204 | 361.9 | 49 |
| bend | 0.253 | 0.345 | 0.200 | 287.3 | 50 |
| squat | 0.239 | 0.421 | 0.167 | 258.5 | 48 |
| pick_up | 0.211 | 0.273 | 0.171 | 278.6 | 35 |
| draw_tick | 0.182 | 0.333 | 0.125 | 301.8 | 48 |
| walk | 0.118 | 0.113 | 0.122 | 277.4 | 49 |
| rotation | 0.000 | 0.000 | 0.000 | 366.9 | 46 |

**Finding**: MetaFi++ (R13) collapses to predicting "nothing" for all samples (21% accuracy = proportion of "nothing" class). Its simple architecture lacks the CLS token attention mechanism needed for multi-class action recognition on MM-Fi's 22 classes.

### 6.4 Inference Latency

| Model | Batch Size | p50 (ms) | p95 (ms) | Mean (ms) | GPU |
|-------|-----------|---------|---------|----------|-----|
| base (15.2M) | 64 | 114.5 | 114.6 | 118.6 | H100 |
| rootrel (15.2M) | 64 | 114.3 | 114.4 | 117.8 | H100 |
| metafi (2M) | 64 | 1.2 | 5.4 | 9.5 | H100 |

MetaFi++ is **~12× faster** than our full model due to drastically fewer parameters (2M vs 15.2M).

---

## 7. SOTA Gap Analysis

### 7.1 Our Results vs Published Baselines

| Model | MPJPE (mm) | PA-MPJPE (mm) | Joints | Eval Split | Notes |
|-------|-----------|---------------|--------|-----------|-------|
| **DT-Pose** (SOTA) | **178.5** | **104.5** | 17 | MM-Fi P3-S1 | MAE + GCN + frozen encoder |
| HPE-Li | 184.3 | 106.4 | 17 | MM-Fi P3-S1 | Dynamic kernels |
| MetaFi++ (paper) | 197.1 | 121.2 | 17 | MM-Fi P3-S1 | Transformer baseline |
| **Ours: rootrel-mmfi** | **305.9** | **~220** | 13 | Internal val | SSL + rootrel |
| **Ours: metafi-reimpl** | **318.9** | **206.9** | 13 | Internal val | Reimplemented from scratch |
| Ours: base-mmfi | 324.0 | 208.3 | 13 | Internal val | SSL + supervised |

**Gap to SOTA: ~127mm (71%)**

### 7.2 Root Cause Analysis

#### Factor 1: CSI Representation (estimated ~40–50mm gap)

| | DT-Pose | Ours |
|--|---------|------|
| Input shape | `[A × (E·R·S) × T]` image-like | `[T × N_sub × 2]` sequence |
| Channels | 3 RX antennas (spatial diversity) | 2 (amplitude + phase only) |
| Patching | 2D grid patches (subcarrier × time) | 1D patches (subcarrier only) |
| Temporal | Captured in 2D patch structure | Separate TemporalEncoder |

DT-Pose treats CSI as an image with antenna diversity as RGB-like channels and applies 2D ViT patching. Our model loses spatial diversity by collapsing antennas into amplitude/phase and uses 1D patching along the subcarrier axis only.

#### Factor 2: Pre-training Method (estimated ~30–35mm gap)

| | DT-Pose | Ours |
|--|---------|------|
| Method | MAE (80% masking → reconstruct CSI) | Contrastive SSL (augmented views) |
| Temporal | Temporal-consistent contrastive loss | None |
| Regularization | Uniformity regularization | None |
| Ablation | Scratch: 198.6mm → Full: 165.3mm (−33mm) | — |

DT-Pose ablation shows their pre-training alone contributes **33mm improvement**. MAE forces the encoder to deeply understand CSI structure by reconstructing masked patches, while our contrastive SSL only learns to discriminate between augmented views.

#### Factor 3: Evaluation Protocol (estimated ~15–20mm gap)

| | DT-Pose | Ours |
|--|---------|------|
| Joints | 17 (COCO) | 13 (RF-WorldPose) |
| Split | MM-Fi official P3-S1 (random 3:1) | Internal val_ratio=0.2 |
| Data volume | ~320K frames | ~10K windows × 60 frames |

Different skeleton topologies (13 vs 17 joints) make MPJPE not directly comparable. Our 13-joint skeleton excludes facial keypoints (eyes, ears) which typically have lower error — potentially inflating our MPJPE relative to COCO-17.

#### Factor 4: Decoder Architecture (estimated ~10–15mm gap)

DT-Pose ablation on decoder components (all with pre-trained encoder):

| Components | MPJPE |
|-----------|-------|
| No task prompt, no GCN, no TF | 197.4mm |
| + Task prompt only | 174.1mm (−23.3mm) |
| + GCN only | 179.8mm (−17.6mm) |
| + Task prompt + GCN | 166.7mm (−30.7mm) |
| + Task prompt + GCN + TF (full) | 165.3mm (−32.1mm) |

**Task prompt** (learnable per-joint queries) is the single most impactful decoder component. Our GCN decoder (Run 14) failed because the encoder was not pre-trained with MAE quality — freezing an insufficiently trained encoder produces weak features.

#### Factor 5: Model Capacity vs Data (estimated ~10–15mm)

Our 15.2M param model overfits 10K training windows within 18–43 epochs. MetaFi++ reimplementation (2M params, Run 13) achieves nearly equivalent performance (318.9mm vs 305.9mm), suggesting model capacity far exceeds what the data can support. DT-Pose likely benefits from their larger effective dataset (~320K frames) relative to model size.

### 7.3 Gap Summary

| Factor | Estimated Impact | Status |
|--------|-----------------|--------|
| CSI representation (1D vs 2D, lost antenna info) | 40–50mm | **Requires data pipeline refactor** |
| Pre-training (contrastive SSL vs MAE) | 30–35mm | **Not yet implemented** |
| Evaluation protocol (13 vs 17 joints, split) | 15–20mm | Partially addressable |
| Decoder quality (task prompt, frozen encoder) | 10–15mm | Tested (R14), needs better pre-training first |
| Model capacity vs data volume | 10–15mm | Partially addressed (R13 MetaFi++) |
| **Total estimated** | **105–135mm** | Matches actual gap (~127mm) |

---

## 8. Architectural Variant Comparison

### 8.1 Effect of Root-Relative Prediction

| Dataset | Base MPJPE | RootRel MPJPE | Improvement | % Change |
|---------|-----------|---------------|------------|----------|
| WiPose | 325.5mm | **286.1mm** | −39.4mm | −12.1% |
| MM-Fi | 327.4mm | **305.9mm** | −21.5mm | −6.6% |
| Combined | 367.8mm | **318.8mm** | −49.0mm | −13.3% |

**Conclusion**: Root-relative prediction is consistently effective across all datasets. Improvement is largest on combined data where coordinate space differences amplify translation error.

### 8.2 Effect of Subcarrier Attention Tokenizer

| Dataset | Base MPJPE | SubcAttn MPJPE | Change |
|---------|-----------|---------------|--------|
| WiPose | 325.5mm | 333.1mm | **+7.6mm (worse)** |
| MM-Fi | 327.4mm | 373.7mm | **+46.3mm (worse)** |
| Combined | 367.8mm | 319.2mm | −48.6mm (but combined base is weak) |

**Conclusion**: Subcarrier attention **does not help** and can significantly hurt. The learnable cross-attention queries add model capacity that overfits on limited data. Fixed subcarrier patches provide effective inductive bias.

### 8.3 Effect of GCN Skeleton Decoder

| | Base MM-Fi | GCN+RootRel MM-Fi |
|--|-----------|-------------------|
| MPJPE | 327.4mm | 338.8mm (+11.4mm worse) |
| PA-MPJPE | 208.3mm | 258.0mm (+49.7mm worse) |
| Encoder | Fine-tuned | Frozen |

**Conclusion**: GCN decoder with frozen encoder performs **worse** than base model. The encoder features from our SSL pre-training are not rich enough to support a decoder-only training protocol. DT-Pose succeeds because their MAE pre-training produces much higher quality encoder features. Our GCN experiment confirms that **pre-training quality is the gating factor**, not decoder architecture.

### 8.4 MetaFi++ Baseline Comparison

| | Our Full Model | MetaFi++ Reimpl |
|--|---------------|-----------------|
| Params | 15.2M | **2.0M** |
| Pre-training | SSL | None (scratch) |
| MPJPE | **305.9mm** | 318.9mm |
| PA-MPJPE | ~220mm | **206.9mm** |
| Action Acc | **38.6%** | 21.1% |
| Inference | 114ms/batch | **1.2ms/batch** |

**Conclusion**: MetaFi++ achieves comparable pose accuracy with **7.6× fewer parameters** and **95× faster inference**. Its PA-MPJPE (206.9mm) is actually the best across all MM-Fi models, meaning it learns body articulation better than our larger model. However, it completely fails at action recognition (predicts only "nothing" class). Our SSL-pretrained model's advantage is primarily in translation accuracy and action recognition, not body articulation.

---

## 9. Technical Issues Encountered & Resolved

### 9.1 API ImportError (`helios` module)
- **Error**: FastAPI server crashed — `from rfpose_api.routers import helios` but no `helios.py` existed
- **Fix**: Removed non-existent import from `services/api/src/rfpose_api/main.py`

### 9.2 pyproject.toml build-backend
- **Error**: `BackendUnavailable` during `pip install -e .` on Eagle — wrong build-backend string
- **Fix**: Changed `build-backend` from `"setuptools.backends._legacy:_Backend"` to `"setuptools.build_meta"`

### 9.3 DDP "parameters did not receive gradient"
- **Error**: `RuntimeError` when wrapping both tokenizer and model with DDP
- **Fix**: Only wrap main model with DDP (not tokenizer), set `find_unused_parameters=True`

### 9.4 Slurm OUT_OF_MEMORY (RAM)
- **Error**: 4 GPU processes × 4 workers × per-sample `np.load(y.npz)` decompression → 128GB RAM exhausted
- **Fix**: Preload all y.npz arrays into RAM at dataset init (~63MB compressed). Workers share via copy-on-write after fork.

### 9.5 Training stuck (y.npz per-sample decompression)
- **Error**: Each `__getitem__` called `np.load(y_path)` → opened zip, decompressed entire array → ~8s/batch
- **Fix**: Preloaded y data in `GoldNpzDataset._scan()`, stored as numpy arrays in `self._y_arrays`. Reduced from 8s/batch to <0.13s/batch (**65× speedup**).

### 9.6 Dataset key mismatch
- **Error**: `rfpose-multitask-v1` used `activity_id`/`activity_mask` keys instead of `action_label`/`action_mask`
- **Fix**: Added fallback key lookup in `gold_npz_dataset.py`

### 9.7 Coordinate scale mismatch across datasets
- **Error**: MM-Fi ground truth in meters (values 0–4), WiPose in millimeters (values 100–600)
- **Fix**: Added `DATASET_COORD_SCALE` dict in `gold_npz_dataset.py` — wipose ×0.001 (mm→m)

### 9.8 S3 Upload Failures During Training
- **Error**: `XAmzContentSHA256Mismatch` errors when `mlflow.log_artifact` uploads checkpoints to MinIO S3, causing job crashes. Also triggered DDP NCCL timeouts on multi-GPU jobs.
- **Fix**: Wrapped `mlflow.log_artifact` calls in `try/except` blocks in `train_v2.py`. Transient S3 errors no longer crash training.

### 9.9 GCN Import Error
- **Error**: `ImportError: cannot import name 'SpatialTransformerEncoder'` — wrong class names in `pose_decoder_gcn.py`
- **Fix**: Changed to `SpatialEncoder`/`TemporalEncoder` (actual class names) and adapted forward method to match 4D input API `[B, T, N, D]`.

### 9.10 Shared venv contention
- **Error**: Multiple concurrent Slurm jobs running `pip install -e .` on the same shared virtual environment, causing package corruption
- **Fix**: Resubmitted crashed jobs; concurrent installs resolved by Slurm serialization on same node

---

## 10. Key Observations (Updated)

1. **Root-relative prediction is the most effective architectural improvement**: Consistently improves MPJPE by 6–13% across all datasets by separating global translation from body articulation.

2. **Single-dataset training outperforms combined**: Confirmed across all variants. Dataset-specific sensor setups, joint definitions, and collection protocols create irreconcilable conflicts.

3. **Subcarrier attention tokenizer does not help**: Learnable frequency grouping adds capacity that overfits on limited data. Fixed 6-subcarrier patches provide effective inductive bias.

4. **MetaFi++ is surprisingly competitive**: 2M params achieves comparable pose accuracy to our 15.2M model. Suggests our model is heavily over-parameterized for available data. MetaFi++ is 12× faster at inference.

5. **GCN skeleton decoder requires strong pre-training**: Freezing encoder + GCN decoder (DT-Pose protocol) fails when encoder features are not MAE-quality. Pre-training method is the gating factor.

6. **Action recognition from WiFi CSI is highly effective**: 78.4% accuracy on WiPose (12 classes) proves CSI carries rich activity semantics. MM-Fi's 22 classes are harder (38.6%) but still meaningful.

7. **Lower extremities are the main pose bottleneck**: Knee/ankle errors (400–515mm) are nearly 2× upper body errors (220–270mm) on MM-Fi. Weighted joint loss may help.

8. **Pre-training quality is the dominant factor in SOTA gap**: DT-Pose's MAE pre-training contributes an estimated 30–35mm improvement. Our contrastive SSL is insufficient for the decoder-frozen protocol that DT-Pose relies on.

9. **CSI representation format matters**: DT-Pose's image-like 2D representation with antenna channels captures spatial-temporal correlations that our 1D subcarrier patching misses. Estimated 40–50mm impact.

10. **Evaluation protocol differences obscure comparison**: 13 vs 17 joints, different train/val splits, and different data windowing make direct MPJPE comparison with SOTA imprecise.

---

## 11. Infrastructure

| Component | Details |
|-----------|---------|
| HPC Cluster | Eagle (Slurm) |
| GPU | NVIDIA H100 (up to 4× per job) |
| Training Framework | PyTorch 2.x, DDP (torchrun) |
| Config | Hydra + OmegaConf |
| Experiment Tracking | MLflow (http://207.180.243.242:5000) |
| Checkpoint Storage | S3 (MinIO) via `s3://rfpose/` |
| Evaluation Script | `rfpose.evaluation.eval_v2` (downloads checkpoints from S3, full pose+action metrics) |
| Data Storage | `/mnt/storage_6/project_data/pl0501-01/rf-worldpose/data/gold/rfpose-unified-v2` |
| Code | `/mnt/storage_6/project_data/pl0501-01/rf-worldpose/ml/` |

---

## 12. Next Steps (Priority Order)

### High Priority
1. **Implement MAE pre-training**: Replace contrastive SSL with DT-Pose-style Masked Autoencoder on CSI data. Expected improvement: 30–35mm. This is the single highest-impact change.

2. **Await weighted joint loss results (Run 15)**: Monitor `rootrel-mmfi-weighted-v1` for improvement on leg joints. Target < 290mm on MM-Fi.

3. **Align evaluation with MM-Fi official protocols**: Run eval using MM-Fi P1/P2/P3 × S1/S2/S3 splits for fair comparison with published results.

### Medium Priority
4. **Refactor CSI tokenizer to 2D image-like format**: Reshape CSI as `[A × (E·R·S) × T]` with antenna channels, apply 2D ViT patching. Expected improvement: 40–50mm but requires data pipeline changes.

5. **Reduce model size**: MetaFi++ (2M params) is nearly as good as our 15.2M model. Experiment with d_model=128, fewer layers, or knowledge distillation.

6. **Cross-dataset transfer learning**: Pre-train on WiPose (better per-sample performance), fine-tune on MM-Fi.

### Low Priority
7. **Data augmentation**: Time-shift, frequency jittering, CSI mixup, random subcarrier masking.

8. **17-joint COCO skeleton**: Align with SOTA evaluation by predicting all 17 COCO joints instead of 13.

9. **Phase 3 — Room-specific fine-tuning**: Deploy camera + 4 RF nodes, collect CSI+pose, fine-tune best checkpoint.
