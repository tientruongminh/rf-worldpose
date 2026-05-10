# RF-WorldPose Dataset, Training Pipeline, and Model Implementation Guide

This document defines the recommended public datasets, data normalization strategy, training stages, and model implementation plan for RF-WorldPose: a WiFi CSI sensing platform that learns to infer human pose and activity from wireless channel measurements.

The intended reader is an engineer/researcher who wants to reproduce the RF-WorldPose training stack from public data first, then adapt it to 4 ESP32-S3 CSI nodes deployed in a real room.

---

## 1. Objective

RF-WorldPose aims to learn this mapping:

```text
multi-node WiFi CSI time-series → human pose / motion / activity representation
```

The final production path is:

```text
public CSI datasets
  → CSI encoder pretraining
  → supervised pose/action fine-tuning
  → distillation from vision/radar/depth teachers
  → domain adaptation on RF-WorldPose ESP32-S3 data
  → deployment to gateway / Triton / dashboard
```

The reason for using public datasets first is simple: pose labels are expensive. Public datasets give us synchronized CSI + pose/image/depth/mmWave labels so the model can learn a general RF representation before it sees our custom ESP32-S3 room.

---

## 2. Core Training Strategy

### 2.1 Pretrain + Fine-tune in one sentence

```text
Pretrain teaches the model the language of WiFi CSI; fine-tune teaches it which CSI patterns correspond to human pose and actions.
```

### 2.2 Full training ladder

| Stage | Name | Data | Label required? | Output |
|---|---|---:|---:|---|
| A | CSI normalization | all datasets | optional | unified Silver/Gold tensors |
| B | Self-supervised pretraining | raw CSI from public + ESP32 | no | `csi_encoder_pretrained.pt` |
| C | Supervised pose fine-tuning | MM-Fi, Wi-Pose, Person-in-WiFi-3D | yes | `rfworldpose_pose_base.pt` |
| D | Multi-modal distillation | datasets with RGB/depth/mmWave/pose | yes / pseudo-label | better pose teacher/student alignment |
| E | ESP32-S3 domain adaptation | RF-WorldPose self-collected CSI | preferably pseudo-label | `room_adapter_lora.pt` |
| F | Evaluation + packaging | held-out public + self data | yes | model card + deployment artifact |

---

## 3. Dataset Inventory

### 3.1 Tier 1 datasets — primary pose training data

These are the most important datasets because they include pose-related labels or synchronized modalities that can produce pose labels.

#### 3.1.1 MM-Fi

**Purpose:** primary dataset for RF-WorldPose base training.

**Links:**

- Project: <https://ntu-aiot-lab.github.io/mm-fi>
- GitHub: <https://github.com/ybhbingo/MMFi_dataset>
- Paper: <https://arxiv.org/abs/2305.10345>
- OpenReview: <https://openreview.net/forum?id=1uAsASS1th>

**What it contains:**

```text
RGB image
Depth image
LiDAR point cloud
mmWave radar point cloud
WiFi CSI
2D/3D human pose landmarks
Action categories
40 subjects
~320k synchronized frames
```

**Why it matters:**

MM-Fi is the best first dataset because it is multi-modal and synchronized. It can train more than one objective:

```text
CSI → 2D pose
CSI → 3D pose
CSI → action
CSI ↔ RGB/depth/mmWave representation alignment
```

**Recommended RF-WorldPose usage:**

| Use | Recommended? | Notes |
|---|---:|---|
| Self-supervised CSI pretraining | yes | Use CSI windows without labels. |
| Supervised pose training | yes | Main source for 2D/3D pose. |
| Action auxiliary head | yes | Helps stabilize human-motion features. |
| Teacher-student distillation | yes | Use RGB/depth/mmWave/pose as teacher signals. |
| ESP32 direct compatibility | partial | Hardware differs; needs domain adaptation. |

**Expected local layout:**

```text
data/bronze/public/mmfi/
  raw/
  metadata/
  checksums.txt
```

**Important caution:**

MM-Fi data is large. Do not commit downloaded data to git. Store it under local/object storage and register dataset versions in the RF-WorldPose API.

---

#### 3.1.2 Wi-Pose

**Purpose:** lightweight CSI-to-skeleton supervised training and benchmarking.

**Links:**

- Dataset GitHub: <https://github.com/NjtechCVLab/Wi-PoseDataset>
- CSI-Former paper: <https://www.mdpi.com/1099-4300/25/1/20>

**What it contains:**

```text
WiFi CSI
Images
Skeleton annotations
12 actions
12 volunteers
~166,600 packets in .mat format
```

**Why it matters:**

Wi-Pose is smaller and simpler than MM-Fi, but useful for validating the model head and data loaders. Because the data is `.mat`, it is also a good early test for format converters.

**Recommended RF-WorldPose usage:**

| Use | Recommended? | Notes |
|---|---:|---|
| Data loader smoke test | yes | Good first public pose dataset conversion target. |
| CSI → 2D skeleton fine-tuning | yes | Useful for supervised signal. |
| Action classification | yes | Has action categories. |
| 3D pose | limited | Not the main strength. |
| DensePose UV | no | Not a DensePose UV dataset. |

**Expected local layout:**

```text
data/bronze/public/wipose/
  raw_mat/
  extracted/
  metadata/
```

---

#### 3.1.3 Person-in-WiFi-3D

**Purpose:** multi-person and 3D pose fine-tuning.

**Links:**

- Project page: <https://aiotgroup.github.io/Person-in-WiFi-3D/>
- CVPR paper: <https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.pdf>

**What it contains:**

```text
WiFi data
Pose annotations
Raw WiFi dataset variants
Multi-person 3D pose task
~97k WiFi samples reported in the paper
```

**Why it matters:**

RF-WorldPose eventually wants robust room-level sensing, potentially with more than one person. Person-in-WiFi-3D is the best public direction for that goal.

**Recommended RF-WorldPose usage:**

| Use | Recommended? | Notes |
|---|---:|---|
| 3D pose fine-tuning | yes | Primary reason to use it. |
| Multi-person modeling | yes | Important for room-level sensing. |
| Domain generalization | yes | Helps model avoid overfitting to MM-Fi only. |
| Early MVP | optional | Dataset access/download may be harder. |

**Expected local layout:**

```text
data/bronze/public/person_in_wifi_3d/
  raw_wifi/
  pose_annotations/
  metadata/
```

**Access caution:**

Some downloads may be hosted on BaiduNetdisk or other mirrors. Expect manual download/account friction.

---

### 3.2 Tier 2 datasets — pretraining and auxiliary learning

These datasets may not contain detailed pose labels, but they provide valuable CSI motion/activity data for representation learning.

#### 3.2.1 WiAR

**Purpose:** public activity recognition data for CSI encoder pretraining and action auxiliary tasks.

**Links:**

- GitHub: <https://github.com/linteresa/WiAR>
- Paper: <https://ieeexplore.ieee.org/document/8866726/>

**What it contains:**

```text
WiFi CSI activity recognition data
16 activities
10 volunteers
3 indoor environments
```

**Recommended usage:**

```text
self-supervised CSI pretraining
action classification auxiliary head
domain generalization tests
```

WiAR is not enough for pose by itself, but it can help the encoder learn human-motion CSI patterns.

---

#### 3.2.2 Widar 3.0

**Purpose:** large-scale gesture/activity CSI pretraining.

**Links:**

- IEEE DataPort: <https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset>
- Awesome WiFi CSI Sensing list: <https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing>

**What it contains:**

```text
RSSI + CSI
Intel 5300 NIC data
30 subcarriers
~258k gesture instances
```

**Recommended usage:**

```text
large-scale masked CSI pretraining
gesture/action auxiliary training
robustness against subject/environment variation
```

**Limitations:**

```text
not pose-labeled
hardware differs from ESP32-S3
gesture domain differs from full-body pose
```

---

#### 3.2.3 WiMANS and other CSI sensing datasets

**Purpose:** additional activity/multi-user CSI pretraining when available.

**Reference:**

- WiMANS paper page/search: <https://arxiv.org/html/2402.09430>
- Awesome WiFi CSI Sensing list: <https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing>

**Recommended usage:**

```text
optional pretraining expansion
multi-user activity auxiliary training
cross-dataset robustness evaluation
```

Use only after MM-Fi/Wi-Pose loaders are stable.

---

### 3.3 Reference paper — DensePose From WiFi

**Purpose:** architecture and research direction reference.

**Links:**

- arXiv: <https://arxiv.org/abs/2301.00250>
- PDF: <https://arxiv.org/pdf/2301.00250>

**Important note:**

This paper is highly relevant conceptually, but do not assume an official public training dataset/code is available. Use it as a design reference, not as the primary data dependency.

---

### 3.4 RF-WorldPose self-collected ESP32-S3 dataset

**Purpose:** final domain adaptation and production validation.

**Hardware target:**

```text
4 ESP32-S3 CSI nodes
1 gateway machine
controlled indoor room
optional camera only during data collection for label generation
```

**Why self data is mandatory:**

Public datasets often use Intel 5300/Atheros/router setups. RF-WorldPose uses ESP32-S3, different antenna geometry, subcarrier behavior, noise, sampling, and room multipath. A model trained only on public data will not be reliable without adaptation.

**Recommended capture plan:**

| Phase | Duration | Subjects | Actions | Label source |
|---|---:|---:|---|---|
| calibration | 10-20 min/room | 0-1 | empty room, walking grid | none/manual |
| pilot | 1-2 hours | 2-3 | stand/sit/walk/raise arms | MediaPipe/OpenPose |
| beta | 5-10 hours | 5-10 | daily actions + transitions | camera pseudo-label |
| production validation | 20+ hours | 10+ | varied clothing/body/positions | mixed/manual QA |

**Privacy rule:**

Video should be used only to generate training labels, then deleted or stored separately with strict access controls. Production inference should not require cameras.

---

## 4. Unified Data Lake Layout

RF-WorldPose should preserve dataset provenance and gradually normalize raw data into trainable tensors.

```text
data/
  bronze/
    public/
      mmfi/
      wipose/
      person_in_wifi_3d/
      wiar/
      widar3/
    self/
      room001_esp32/
  silver/
    csi_windows/
    pose_labels/
    action_labels/
    sync_tables/
  gold/
    pretrain_csi/
    pose2d_train/
    pose3d_train/
    action_train/
    domain_adaptation/
```

### 4.1 Bronze

Bronze stores raw downloaded or captured files with minimal mutation.

Rules:

```text
never overwrite raw data
store checksums
store original license/readme
store download source URL
store dataset version
```

Example metadata:

```json
{
  "dataset": "mmfi",
  "version": "public-2023",
  "source_url": "https://ntu-aiot-lab.github.io/mm-fi",
  "local_path": "data/bronze/public/mmfi/raw",
  "license": "check-upstream",
  "modalities": ["wifi_csi", "rgb", "depth", "lidar", "mmwave", "pose2d", "pose3d"],
  "checksum_manifest": "checksums.txt"
}
```

### 4.2 Silver

Silver stores decoded and synchronized intermediate representations.

Recommended tables/files:

```text
csi_window.parquet
pose2d.parquet
pose3d.parquet
action.parquet
sync_index.parquet
subject_environment.parquet
```

Suggested CSI window schema:

| Field | Type | Description |
|---|---|---|
| `sample_id` | string | stable ID |
| `dataset` | string | mmfi/wipose/self/etc. |
| `subject_id` | string/null | subject identity if available |
| `environment_id` | string/null | room/session ID |
| `start_ts` | int/float | window start timestamp |
| `end_ts` | int/float | window end timestamp |
| `sample_rate_hz` | float | CSI sample rate |
| `csi_real` | tensor/ref | real component or storage ref |
| `csi_imag` | tensor/ref | imaginary component or storage ref |
| `amplitude` | tensor/ref | amplitude tensor |
| `phase` | tensor/ref | phase tensor |
| `num_subcarriers` | int | subcarrier count |
| `num_rx` | int | receiver antenna count if known |
| `num_tx` | int | transmitter count if known |
| `node_ids` | list | ESP32/public receiver IDs |

### 4.3 Gold

Gold stores ML-ready fixed-shape examples.

Recommended tensor format:

```text
X_csi: float32 [T, N, S, C]
  T = time steps
  N = nodes/receivers
  S = subcarriers
  C = channels, e.g. amplitude, phase, real, imag

Y_pose2d: float32 [T_label, J, 3]
  J = joints
  3 = x, y, confidence

Y_pose3d: float32 [T_label, J, 4]
  4 = x, y, z, confidence

Y_action: int64 or multi-hot
```

For ESP32-S3 4-node mode:

```text
X_csi: [T, 4, S, C]
```

If a public dataset has one receiver instead of four nodes, use:

```text
N = 1
node_mask = [1, 0, 0, 0]
```

This lets the model learn with missing-node masking.

---

## 5. Data Normalization and Feature Engineering

### 5.1 Raw CSI representation

For each CSI sample:

```text
complex CSI H = I + jQ
amplitude = sqrt(I^2 + Q^2)
phase = atan2(Q, I)
```

Recommended input channels:

```text
amplitude_log
phase_unwrapped_or_sanitized
real_normalized
imag_normalized
optional: phase_difference
optional: temporal_derivative
```

### 5.2 Preprocessing steps

Recommended order:

1. Validate packet/session integrity.
2. Drop corrupted CSI frames.
3. Align timestamps.
4. Resample to fixed rate.
5. Remove static phase offsets if possible.
6. Normalize per session/environment.
7. Slice windows.
8. Attach labels using nearest timestamp or interpolation.

### 5.3 Windowing

Initial recommended settings:

```text
window_length: 1.0-2.0 seconds
stride: 0.25-0.5 seconds
sample_rate: normalize to 50-100 Hz depending dataset
```

For fast movement/action:

```text
shorter windows: 0.5-1.0 seconds
```

For stable pose:

```text
longer windows: 2.0-4.0 seconds
```

### 5.4 Cross-dataset normalization

Each dataset has different hardware. Do not blindly concatenate raw tensors.

Use dataset-aware normalization:

```text
per-dataset mean/std
per-environment mean/std
subcarrier masking
node/antenna masking
dataset embedding token
hardware embedding token
```

---

## 6. Model Architecture

### 6.1 High-level architecture

```text
CSI input
  → CSI stem
  → temporal/subcarrier encoder
  → node fusion / antenna fusion
  → shared latent representation
  → task heads
      → 2D pose head
      → 3D pose head
      → action head
      → confidence/uncertainty head
```

### 6.2 Recommended baseline model

Start with a practical baseline rather than an over-complex DensePose model.

```text
CSIStem:
  1D/2D convolution over time/subcarrier
  layer norm
  positional encoding

CSIEncoder:
  temporal transformer or conformer blocks
  subcarrier attention
  node attention

Fusion:
  cross-node attention
  missing-node mask support

Heads:
  MLP pose2d head
  MLP pose3d head
  classifier action head
  confidence head
```

### 6.3 Input/output contract

Input batch:

```python
batch = {
    "x_csi": FloatTensor[B, T, N, S, C],
    "node_mask": BoolTensor[B, N],
    "dataset_id": LongTensor[B],
    "hardware_id": LongTensor[B],
    "subject_id": Optional[LongTensor[B]],
}
```

Output:

```python
output = {
    "pose2d": FloatTensor[B, T_out, J, 3],
    "pose3d": FloatTensor[B, T_out, J, 4],
    "action_logits": FloatTensor[B, num_actions],
    "uncertainty": FloatTensor[B, T_out, J, 1],
    "embedding": FloatTensor[B, D],
}
```

### 6.4 Joint convention

Use COCO 17 keypoints as the first target because many pose tools support it.

```text
0 nose
1 left_eye
2 right_eye
3 left_ear
4 right_ear
5 left_shoulder
6 right_shoulder
7 left_elbow
8 right_elbow
9 left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
```

For 3D datasets, store a mapping table if the source uses a different skeleton.

---

## 7. Self-supervised Pretraining

### 7.1 Goal

Learn a robust RF representation without requiring pose labels.

### 7.2 Recommended pretraining tasks

#### Task 1 — Masked CSI modeling

Randomly mask time/subcarrier patches and reconstruct them.

```text
input: partially masked CSI window
prediction: missing CSI amplitude/phase/real/imag patches
loss: L1/Huber/MSE reconstruction loss
```

This is the RF equivalent of masked language/image modeling.

#### Task 2 — Temporal contrastive learning

Positive pairs:

```text
overlapping windows from same session/time neighborhood
same action under augmented CSI
```

Negative pairs:

```text
different time/session/action/environment
```

Loss:

```text
InfoNCE / supervised contrastive if action labels exist
```

#### Task 3 — Future window prediction

Predict next latent representation from past CSI.

```text
past windows → future embedding
```

Useful because human pose and motion are temporally smooth.

### 7.3 Augmentations

Use RF-safe augmentations:

```text
time jitter
subcarrier dropout
node dropout
small gaussian noise
amplitude scaling
phase offset perturbation
time masking
frequency/subcarrier masking
```

Avoid augmentations that destroy physical meaning completely.

### 7.4 Pretrain command target

Future implementation target:

```bash
python -m ml.rfpose.training.pretrain \
  --config configs/pretrain_csi.yaml \
  --datasets mmfi,widar3,wiar,self_esp32 \
  --output artifacts/csi_encoder_pretrained.pt
```

---

## 8. Supervised Fine-tuning

### 8.1 Pose fine-tuning objective

Primary losses:

```text
pose2d_loss = Huber(pred_xy, target_xy) weighted by keypoint confidence
pose3d_loss = MPJPE / Huber(pred_xyz, target_xyz)
temporal_loss = smoothness(pred_t - pred_t-1)
confidence_loss = calibration loss for predicted uncertainty
```

Optional losses:

```text
action_loss = cross entropy
bone_length_loss = skeleton consistency
left_right_symmetry_loss = weak regularizer
```

### 8.2 Multi-dataset training

Mix datasets carefully:

```text
batch sampler balances dataset sources
missing labels are masked out
pose2d-only datasets do not affect pose3d loss
activity-only datasets do not affect pose loss
```

Example loss composition:

```text
L = λ2d * L_pose2d
  + λ3d * L_pose3d
  + λact * L_action
  + λtemp * L_temporal
  + λbone * L_bone
```

### 8.3 Fine-tune command target

```bash
python -m ml.rfpose.training.finetune_pose \
  --config configs/finetune_pose.yaml \
  --init artifacts/csi_encoder_pretrained.pt \
  --datasets mmfi,wipose,person_in_wifi_3d \
  --output artifacts/rfworldpose_pose_base.pt
```

---

## 9. Multi-modal Distillation

### 9.1 Why distillation matters

WiFi pose labels are sparse and noisy. Vision/depth/mmWave modalities can act as teachers during training.

Teacher examples:

```text
MediaPipe Pose
OpenPose
Detectron2 DensePose
MM-Fi RGB/depth pose labels
mmWave/radar pose estimates
```

Student:

```text
CSI-only RF-WorldPose model
```

### 9.2 Distillation targets

```text
2D keypoints
3D keypoints
joint heatmaps
body-part segmentation / DensePose UV if available
latent embedding alignment
motion/action logits
```

### 9.3 Distillation loss

```text
L_distill_pose = Huber(student_pose, teacher_pose)
L_distill_heatmap = KL/MSE(student_heatmap, teacher_heatmap)
L_distill_embed = cosine distance(student_embedding, teacher_embedding)
```

Existing repo hook:

```text
ml/rfpose/training/distill.py
```

This should evolve from scaffold to full multi-modal teacher/student training.

---

## 10. ESP32-S3 Domain Adaptation

### 10.1 Problem

Public CSI hardware differs from ESP32-S3:

```text
subcarrier count differs
sampling rate differs
antenna geometry differs
phase stability differs
packet loss differs
room multipath differs
```

### 10.2 Recommended approach

Use adapters/LoRA rather than retraining the full model.

```text
freeze base CSI encoder
train domain adapter / LoRA layers
train small normalization layers
optionally fine-tune pose head
```

Existing repo hook:

```text
ml/rfpose/models/lora.py
```

### 10.3 Self-collection label pipeline

```text
ESP32-S3 CSI nodes record CSI
camera records video only for training session
MediaPipe/OpenPose extracts pseudo pose labels
timestamp sync aligns CSI windows with pose frames
pseudo-label confidence filters bad frames
train domain adapter
```

### 10.4 Minimal capture protocol

For first adapter:

```text
room: 1
nodes: 4
subjects: 2-3
activities: stand, sit, walk, raise arms, turn, squat
recording: 60-120 minutes total
labels: MediaPipe 2D pose from phone/laptop camera
```

For stronger adapter:

```text
subjects: 5-10
rooms: 2-3
recording: 5-10 hours
include empty-room calibration and multi-position grid walk
```

---

## 11. Evaluation

### 11.1 Metrics

Pose metrics:

```text
MPJPE for 3D pose
PCK for 2D keypoints
OKS-style keypoint score
bone length error
temporal jitter
```

Action metrics:

```text
accuracy
macro F1
confusion matrix
```

System metrics:

```text
latency per window
throughput windows/sec
packet drop tolerance
missing node tolerance
cross-room generalization
cross-subject generalization
```

### 11.2 Evaluation splits

Use multiple splits, not only random split.

```text
same-subject random split
cross-subject split
cross-environment split
cross-dataset split
ESP32 room holdout split
```

The most important production metric is not random validation accuracy. It is:

```text
performance on unseen room/subject/hardware condition
```

---

## 12. Implementation Plan in This Repo

### 12.1 Existing components

Relevant existing repo paths:

```text
ml/rfpose/models/lora.py
ml/rfpose/training/distill.py
ml/rfpose/packaging/model_card.py
pipelines/dagster/rfpose_pipelines/etl/bronze_to_silver.py
pipelines/dagster/rfpose_pipelines/etl/silver_to_gold.py
services/api/scripts/register_dataset.py
helios_runner/rfpose_helios/submit.py
```

### 12.2 Missing implementation modules

Recommended new modules:

```text
ml/rfpose/data/datasets/mmfi.py
ml/rfpose/data/datasets/wipose.py
ml/rfpose/data/datasets/person_in_wifi_3d.py
ml/rfpose/data/datasets/wiar.py
ml/rfpose/data/datasets/widar3.py
ml/rfpose/data/transforms/csi.py
ml/rfpose/data/transforms/pose.py
ml/rfpose/models/csi_encoder.py
ml/rfpose/models/rfworldpose.py
ml/rfpose/training/pretrain.py
ml/rfpose/training/finetune_pose.py
ml/rfpose/eval/pose_metrics.py
configs/pretrain_csi.yaml
configs/finetune_pose.yaml
configs/domain_adapt_esp32.yaml
```

### 12.3 Minimal model skeleton

```python
class RFWorldPose(nn.Module):
    def __init__(self, num_nodes=4, num_subcarriers=64, channels=4, d_model=256):
        super().__init__()
        self.stem = CSIStem(channels, d_model)
        self.encoder = CSITemporalEncoder(d_model=d_model)
        self.node_fusion = NodeFusion(d_model=d_model)
        self.pose2d_head = PoseHead(d_model, joints=17, dims=3)
        self.pose3d_head = PoseHead(d_model, joints=17, dims=4)
        self.action_head = nn.Linear(d_model, NUM_ACTIONS)

    def forward(self, x_csi, node_mask=None, dataset_id=None, hardware_id=None):
        z = self.stem(x_csi)
        z = self.encoder(z)
        z = self.node_fusion(z, node_mask=node_mask)
        return {
            "pose2d": self.pose2d_head(z),
            "pose3d": self.pose3d_head(z),
            "action_logits": self.action_head(z.mean(dim=1)),
            "embedding": z.mean(dim=1),
        }
```

### 12.4 Training config example

```yaml
seed: 42
precision: bf16
window:
  seconds: 2.0
  stride_seconds: 0.5
  sample_rate_hz: 50
input:
  num_nodes: 4
  num_subcarriers: 64
  channels:
    - amplitude_log
    - phase_sanitized
    - real
    - imag
model:
  d_model: 256
  layers: 8
  heads: 8
  dropout: 0.1
train:
  batch_size: 64
  epochs: 100
  lr: 0.0003
  weight_decay: 0.05
loss:
  pose2d: 1.0
  pose3d: 1.0
  action: 0.2
  temporal: 0.1
  bone: 0.05
datasets:
  - name: mmfi
    weight: 0.5
  - name: wipose
    weight: 0.2
  - name: person_in_wifi_3d
    weight: 0.2
  - name: self_esp32
    weight: 0.1
```

---

## 13. Download and Registration Workflow

### 13.1 Download manually

Because most public datasets have their own license/access process, downloads should be manual or semi-automatic.

Recommended steps:

1. Visit dataset page.
2. Accept license/terms if required.
3. Download raw archive to local storage.
4. Extract under `data/bronze/public/<dataset>/raw`.
5. Generate checksums.
6. Register dataset in RF-WorldPose API.

### 13.2 Register dataset

Expected future/working command pattern:

```bash
python services/api/scripts/register_dataset.py \
  --name mmfi \
  --version public-2023 \
  --uri s3://rfpose/bronze/public/mmfi \
  --description "MM-Fi public multimodal WiFi CSI human pose dataset"
```

### 13.3 Checksums

```bash
find data/bronze/public/mmfi/raw -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > data/bronze/public/mmfi/checksums.txt
```

---

## 14. Practical First Milestones

### Milestone 1 — Dataset docs and download manifests

```text
Add dataset manifests for MM-Fi, Wi-Pose, Person-in-WiFi-3D.
Document manual download links and expected layout.
```

### Milestone 2 — MM-Fi loader

```text
Parse MM-Fi CSI and pose labels.
Produce Gold tensors.
Run one batch through model skeleton.
```

### Milestone 3 — Pretraining baseline

```text
Masked CSI reconstruction on MM-Fi CSI.
Save encoder checkpoint.
Track in MLflow.
```

### Milestone 4 — Pose fine-tune baseline

```text
Train CSI → 2D/3D pose on MM-Fi subset.
Evaluate PCK/MPJPE.
```

### Milestone 5 — ESP32 adapter

```text
Collect pilot data from 4 nodes.
Generate MediaPipe pseudo-labels.
Train LoRA adapter.
Compare base vs adapted model.
```

---

## 15. Recommended immediate next action

Start with MM-Fi.

```text
1. Download MM-Fi.
2. Put raw data under data/bronze/public/mmfi/raw.
3. Implement MM-Fi converter to Silver/Gold.
4. Train masked CSI pretraining baseline.
5. Fine-tune pose head on MM-Fi labels.
```

Do not start by trying to train full DensePose UV. Start with:

```text
CSI → 17-keypoint 2D/3D pose
```

Once keypoints work, add richer body surface/DensePose-like outputs if enough labels exist.

---

## 16. Summary

RF-WorldPose should not rely on a single dataset. The strongest route is:

```text
MM-Fi for main multimodal pose training
+ Wi-Pose for CSI skeleton validation
+ Person-in-WiFi-3D for 3D/multi-person robustness
+ WiAR/Widar for CSI pretraining scale
+ self-collected ESP32-S3 data for real deployment adaptation
```

The key technical idea is to separate:

```text
general RF representation learning
from
pose-specific supervised learning
from
room/hardware-specific adaptation
```

That makes the system research-grade while still practical for a 4-node ESP32-S3 deployment.
