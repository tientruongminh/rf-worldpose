# RF-WorldPose ML — Training Guide

## Quick Start (5 phút đọc, submit job đầu tiên)

### Bước 1: Viết config YAML

Tạo file `ml/configs/my_experiment.yaml`:

```yaml
# Kế thừa base config (tất cả defaults từ transformer_gold.yaml)
defaults:
  - transformer_gold

data:
  datasets: ["uthar", "wiar"]     # null = dùng tất cả 6 dataset
  val_ratio: 0.2

model:
  d_model: 256                     # kích thước embedding
  n_spatial_layers: 4
  n_temporal_layers: 4

training:
  epochs: 30
  batch_size: 32
  lr: 1.0e-4
  patience: 10                     # early stopping sau N epoch không cải thiện

mlflow:
  experiment_name: "rf-worldpose-transformer"
  run_name: "my-experiment-v1"     # tên hiện trên MLflow UI
```

> Chỉ cần ghi những giá trị muốn thay đổi — phần còn lại kế thừa từ `transformer_gold.yaml`.

### Bước 2: Đăng ký module mapping

Thêm config name vào `eagle_runner/rfpose_eagle/submit.py`:

```python
CONFIG_TO_MODULE = {
    ...
    "my_experiment": "rfpose.training.transformer_train",
}
```

### Bước 3: Submit trên Web Portal

1. Mở **http://207.180.243.242:8082**
2. Bấm **"+ New Job"**
3. Chọn config `my_experiment` từ dropdown
4. Bấm **Submit Training Job**

### Bước 4: Theo dõi trên MLflow

1. Mở **http://207.180.243.242:5000**
2. Sidebar trái → chọn experiment `rf-worldpose-transformer`
3. Click run name → tab **Metrics** → chọn metric → xem chart
4. Tab **Artifacts** → download checkpoint

---

## Repo Structure

```
ml/
├── configs/                        # Hydra YAML configs
│   ├── transformer_gold.yaml       # Base config (mọi config khác kế thừa từ đây)
│   ├── transformer_eagle.yaml      # Full training trên Eagle HPC
│   ├── ssl_pretrain.yaml           # SSL pretrain base
│   ├── ssl_eagle.yaml              # SSL trên Eagle
│   ├── finetune_room.yaml          # Fine-tune cho phòng mới
│   ├── demo.yaml                   # Demo 2 epoch (test pipeline)
│   ├── quick_test.yaml             # Quick test ~10 min
│   └── eval_demo.yaml              # Evaluation job (submit từ Portal)
├── rfpose/
│   ├── data/
│   │   ├── gold_npz_dataset.py     # GoldNpzDataset — load x.npy + y.npz
│   │   └── silver_csi_dataset.py   # Raw CSI parquet (ETL)
│   ├── models/
│   │   ├── csi_tokenizer.py        # CSITokenizer — patch + embed + pos-enc
│   │   └── transformer.py          # SpatialEncoder, TemporalEncoder, PoseDecoder
│   ├── training/
│   │   ├── ssl_pretrain.py         # Phase 1 — SSL encoder pretraining
│   │   └── transformer_train.py    # Phase 2/3 — supervised pose+action
│   ├── evaluation/
│   │   ├── eval.py                 # Eval on test set → JSON report + MLflow
│   │   ├── eval_job.py             # Hydra wrapper — submit eval từ Portal
│   │   └── eval_gate.py            # Quality gate pass/fail
│   ├── export/
│   │   └── onnx.py                 # ONNX export
│   └── utils/
│       └── losses.py               # RFPoseLoss, MPJPE, PA-MPJPE
└── pyproject.toml
```

## Services

| Service | URL | Chức năng |
|---------|-----|-----------|
| **Portal** | http://207.180.243.242:8082 | Submit/cancel/monitor jobs |
| **MLflow** | http://207.180.243.242:5000 | Metrics chart, artifacts, compare runs |
| **API** | http://207.180.243.242:8080/docs | REST API (Swagger) |
| **MinIO** | http://207.180.243.242:9003 | S3 storage (checkpoints, models) |

---

## Hướng dẫn chi tiết

### 1. Viết Training Code

Hiện có 2 training script sẵn:

| Script | Dùng khi |
|--------|----------|
| `ssl_pretrain.py` | Train encoder không cần label (Phase 1) |
| `transformer_train.py` | Train full model có label pose + action (Phase 2/3) |

**Nếu muốn thay đổi model/loss/data pipeline**, sửa trực tiếp `transformer_train.py`.

**Nếu muốn viết training script hoàn toàn mới:**

1. Tạo file `ml/rfpose/training/my_train.py`
2. Cần có `@hydra.main()` decorator ở cuối file
3. Load data bằng `GoldNpzDataset` hoặc tự viết dataset
4. Log metrics bằng `mlflow.log_metrics(...)` mỗi epoch
5. Save checkpoint bằng `torch.save(...)` + `mlflow.log_artifact(...)`

Template tối thiểu:

```python
import hydra, mlflow, torch, logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="my_config", version_base=None)
def train(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params({...})  # log hyperparams

        for epoch in range(cfg.training.epochs):
            # ... training loop ...
            # ... validation ...

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mpjpe": val_mpjpe,
            }, step=epoch)

            # Save best checkpoint
            torch.save(checkpoint, "checkpoints/best.pt")
            mlflow.log_artifact("checkpoints/best.pt")

if __name__ == "__main__":
    train()
```

### 2. Viết Config YAML

Tất cả config nằm trong `ml/configs/`. Mỗi config có 4 section:

```yaml
# (optional) Kế thừa từ config khác
defaults:
  - transformer_gold

# Data: chọn dataset, split ratio
data:
  gold_dir: "..."                  # tự động set bởi sbatch
  datasets: ["uthar", "wiar"]     # null = tất cả
  val_ratio: 0.2
  n_subcarriers: 270
  n_joints: 13
  window_size: 60

# Model: kiến trúc
model:
  patch_size: 6
  d_model: 256                     # nhỏ hơn = train nhanh, lớn hơn = chính xác hơn
  spatial_heads: 8
  temporal_heads: 8
  n_spatial_layers: 4              # số layer encoder
  n_temporal_layers: 4
  n_decoder_layers: 3
  dropout: 0.1
  num_actions: 28

# Training: hyperparameters
training:
  epochs: 50
  batch_size: 32                   # H100 95GB: batch 64 OK, A100 40GB: batch 32
  lr: 1.0e-4
  warmup_epochs: 3
  patience: 15                     # early stopping
  save_every: 5                    # save checkpoint mỗi N epoch
  amp: true                        # mixed precision (nhanh hơn 2x)
  ssl_pretrained: ""               # path tới SSL checkpoint (Phase 1)
  pretrained_from: ""              # path tới full checkpoint (Phase 2→3)
  freeze_encoder: false

# Loss: weighting
loss:
  weighting_mode: "uncertainty"    # "static" hoặc "uncertainty" (Kendall auto-balance)
  lambda_coord: 1.0
  lambda_action: 0.5

# MLflow: tracking
mlflow:
  tracking_uri: "http://207.180.243.242:5000"
  experiment_name: "rf-worldpose-transformer"
  run_name: "my-run-name"
```

**Tips khi thay đổi config:**
- Chỉ ghi giá trị muốn override, phần còn lại kế thừa từ `defaults`
- `datasets: null` = dùng tất cả 6 dataset (219K samples)
- `datasets: ["uthar"]` = chỉ 1 dataset (49K samples, nhanh hơn)
- Tăng `d_model` / `n_*_layers` = model mạnh hơn nhưng chậm hơn
- `patience: 10` = tự dừng sau 10 epoch không cải thiện

### 3. Đăng ký Config (bắt buộc cho job mới)

Thêm vào `eagle_runner/rfpose_eagle/submit.py`:

```python
CONFIG_TO_MODULE = {
    "transformer_gold":  "rfpose.training.transformer_train",
    "transformer_eagle": "rfpose.training.transformer_train",
    "ssl_pretrain":      "rfpose.training.ssl_pretrain",
    "ssl_eagle":         "rfpose.training.ssl_pretrain",
    "demo":              "rfpose.training.transformer_train",
    "eval_demo":         "rfpose.evaluation.eval_job",         # ← evaluation job
    "my_new_config":     "rfpose.training.transformer_train",  # ← thêm dòng này
}
```

### 4. Submit Job trên Web Portal

**URL: http://207.180.243.242:8082**

1. Bấm **"+ New Job"** (góc phải trên)
2. **Training Config**: chọn config từ dropdown
3. **Job ID**: tự động generate, hoặc tự đặt (vd: `train-v3-full`)
4. **Submitted By**: chọn tên
5. Tick **Auto-submit to Eagle HPC** (mặc định bật)
6. Bấm **Submit Training Job**
7. Redirect tới trang Job Detail

**Trên trang Job Detail:**
- **Refresh**: cập nhật trạng thái Slurm (PENDING → RUNNING → COMPLETED)
- **Cancel**: hủy job trên Eagle

**Trạng thái job:**
| Status | Ý nghĩa |
|--------|---------|
| created | Đã tạo trong DB, chưa gửi Eagle |
| submitted | Đã gửi lên Slurm queue, đang chờ GPU |
| running | Đang train trên GPU |
| completed | Train xong |
| failed | Lỗi (xem Slurm log) |
| cancelled | Đã hủy |

### 5. Monitor trên MLflow

**URL: http://207.180.243.242:5000**

**Experiments:**
| Experiment | Chứa |
|-----------|-------|
| `rf-worldpose-ssl` | SSL pretrain runs |
| `rf-worldpose-transformer` | Supervised + eval runs |

**Xem metrics:**
1. Sidebar trái → chọn experiment
2. Click tên run (vd: `demo-e2e-02`)
3. Tab **Metrics** → chọn metric → xem chart theo epoch

**Metrics quan trọng:**

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| `val_mpjpe` | Sai số vị trí khớp (mm) | Càng thấp càng tốt |
| `val_pa_mpjpe` | MPJPE sau alignment | Càng thấp càng tốt |
| `val_action_acc` | Accuracy nhận dạng action | Càng cao càng tốt (0→1) |
| `loss_total` | Tổng loss | Giảm đều = OK |
| `val_loss_total` | Val loss | Giảm rồi tăng = overfitting |

**So sánh runs:**
1. Tick nhiều runs trong danh sách
2. Bấm **Compare** → xem chart chồng lên nhau

**Download checkpoint:**
1. Click run → tab **Artifacts**
2. Click `best.pt` → Download

### 6. Evaluation (submit như 1 job trên Portal)

Evaluation cũng là 1 job — submit y hệt training, chạy trên Eagle GPU, kết quả log lên MLflow.

#### Bước 1: Tạo eval config

Copy `ml/configs/eval_demo.yaml` và sửa checkpoint:

```yaml
eval:
  checkpoint: "mlflow://RUN_ID/best.pt"     # ← paste MLflow Run ID
  device: "cuda"
  batch_size: 64
  num_workers: 4

data:
  datasets: null                             # null = eval trên tất cả dataset

# Section training bắt buộc (sbatch truyền vào, eval bỏ qua)
training:
  device: "cuda"
  epochs: 1
  batch_size: 64
  dry_run: false
  checkpoint_dir: ""

mlflow:
  tracking_uri: "http://207.180.243.242:5000"
  experiment_name: "rf-worldpose-transformer"
  run_name: "eval-my-model"                  # ← tên hiện trên MLflow
```

**Lấy Run ID:** Mở MLflow → click run đã train → copy Run ID từ URL hoặc info panel.

**Checkpoint source hỗ trợ:**
| Format | Ví dụ |
|--------|-------|
| `mlflow://RUN_ID/best.pt` | Tự download từ MLflow S3 (khuyến khích) |
| `/path/to/best.pt` | File trực tiếp trên Eagle |

#### Bước 2: Đăng ký (nếu config mới)

```python
CONFIG_TO_MODULE = {
    ...
    "eval_demo":     "rfpose.evaluation.eval_job",
    "eval_my_model": "rfpose.evaluation.eval_job",    # ← thêm config mới
}
```

> Config `eval_demo` đã đăng ký sẵn. Nếu chỉ sửa checkpoint trong `eval_demo.yaml` thì không cần đăng ký lại.

#### Bước 3: Submit trên Portal

1. Mở **http://207.180.243.242:8082**
2. Bấm **"+ New Job"**
3. Chọn config `eval_demo` (hoặc eval config mới)
4. Submit → chạy trên Eagle GPU → kết quả tự log MLflow

#### Kết quả trên MLflow

Sau khi job COMPLETED, mở MLflow → click run:
- **Params**: `eval_checkpoint`, `eval_datasets`, `eval_n_samples`
- **Metrics**: `test_mpjpe`, `test_pa_mpjpe`, `test_accuracy`, `test_macro_f1`, `test_latency_p50`
- **Artifacts**: `eval_report.json` (download để xem chi tiết per-action)

**Metrics giải thích:**

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| `test_mpjpe` | Sai số vị trí khớp trung bình | Càng thấp càng tốt |
| `test_pa_mpjpe` | MPJPE sau Procrustes alignment | Càng thấp càng tốt |
| `test_accuracy` | Accuracy action classification | Càng cao càng tốt |
| `test_macro_f1` | F1 trung bình mỗi action class | > 0.5 cơ bản, > 0.7 tốt |
| `test_latency_p50` | Inference latency (ms/batch) | < 20ms OK |

#### Chạy eval thủ công (không qua Portal)

```bash
python -m rfpose.evaluation.eval \
  --checkpoint checkpoints/best.pt \
  --gold-dir /path/to/gold/rfpose-multitask-v1 \
  --output eval_report.json \
  --mlflow
```

#### Quality gate (CI/CD)

```bash
python -m rfpose.evaluation.eval_gate \
  --report eval_report.json \
  --max-mpjpe 0.10 \
  --min-macro-f1 0.50
# exit 0 = pass, exit 2 = fail
```

---

## Chạy nhiều job cùng lúc

Portal + Eagle hỗ trợ chạy song song. Mỗi job có riêng:
- Job ID + Slurm Job ID
- Checkpoint directory
- MLflow run (tên khác nhau)

**Ví dụ submit 3 job thí nghiệm:**

1. Tạo 3 config:
   - `configs/exp_small.yaml` — d_model=128, 2 layers
   - `configs/exp_medium.yaml` — d_model=256, 4 layers  
   - `configs/exp_large.yaml` — d_model=512, 6 layers

2. Đăng ký trong `submit.py`

3. Submit lần lượt trên Portal (mỗi job tự tạo Job ID khác nhau)

4. So sánh trên MLflow: tick 3 runs → Compare → chart `val_mpjpe`

---

## Training Pipeline — 3 Phases

```
Phase 1: SSL pretrain     → csi_encoder_pretrained.pt (encoder only)
Phase 2: Supervised train → best.pt (full model: encoder + decoder + heads)
Phase 3: Room fine-tune   → room_best.pt (adapt to new environment)
```

### Phase 1 — SSL (không cần label)

Học cấu trúc tín hiệu WiFi CSI từ 219K windows.

Config: `ssl_eagle.yaml`
Script: `ssl_pretrain.py`
Output: `csi_encoder_pretrained.pt`

Watch: `epoch/val_recon` trên MLflow → plateau = dừng.

### Phase 2 — Supervised (cần label pose + action)

Train full model có label. Nếu có Phase 1 checkpoint:

```yaml
training:
  ssl_pretrained: "checkpoints/ssl-v1/csi_encoder_pretrained.pt"
```

Config: `transformer_eagle.yaml`
Script: `transformer_train.py`
Output: `best.pt`, `model.onnx`

Watch: `val_mpjpe` → auto-stops khi plateau.

### Phase 3 — Room fine-tune

Adapt model cho phòng/hardware cụ thể:

```yaml
defaults:
  - finetune_room

training:
  pretrained_from: "checkpoints/train-v1/best.pt"
  freeze_encoder: true
```

---

## Data format (Gold v2)

```
data/gold/rfpose-multitask-v1/
├── uthar/              49,730 samples
│   ├── x.npy           (N, 2, 60, 270) float32 — CSI [amplitude, phase]
│   ├── y.npz            pose, pose_mask, action_label, action_mask
│   └── metadata.npz     split (train/val/test) per sample
├── wiar/               35,107 samples
├── wimans/            112,860 samples
├── label_maps.json     unified action label → index
└── manifest.json       per-dataset stats
```

Total: ~197K labeled samples, 28 action classes, 13 joints.

---

## Troubleshooting

### Job FAILED ngay lập tức

**Xem log:**
- Portal → click job → xem error message
- Hoặc SSH vào Eagle xem trực tiếp:

```bash
# Thay {job_id} và {slurm_id} bằng giá trị thật
cat ~/pl0501-01/project_data/rf-worldpose/logs/rfpose-{job_id}-{slurm_id}.err
```

**Nguyên nhân phổ biến:**

| Lỗi | Fix |
|-----|-----|
| `ModuleNotFoundError` | Code chưa sync — submit lại từ Portal (tự rsync) |
| `Key 'xxx' is not in struct` | Config thiếu field — thêm vào YAML hoặc dùng `+key=value` |
| `CUDA out of memory` | Giảm `batch_size` trong config |
| `No such file or directory: gold_dir` | Sai path data — check `data.gold_dir` trong config |

### MLflow hiện RUNNING nhưng job đã xong

Job bị cancel/crash trước khi `mlflow.end_run()` được gọi → run bị orphan.

**Fix từng run (MLflow UI):**
1. Mở http://207.180.243.242:5000
2. Click run → nút **⋮** (menu) → **Delete** hoặc dùng API

**Fix tất cả orphan runs (API):**

```bash
# Trên VPS (ssh root@207.180.243.242)
curl -X POST http://localhost:5000/api/2.0/mlflow/runs/update \
  -H "Content-Type: application/json" \
  -d '{"run_id": "PASTE_RUN_ID_HERE", "status": "FINISHED"}'
```

**Fix hàng loạt (script):**

```bash
# Trên VPS — close tất cả RUNNING runs trong experiment 4
python3 -c "
import urllib.request, json
base = 'http://localhost:5000'
for eid in ['4', '5']:
    data = json.dumps({'experiment_ids': [eid], 'filter_string': \"attributes.status = 'RUNNING'\"}).encode()
    req = urllib.request.Request(f'{base}/api/2.0/mlflow/runs/search', data=data)
    req.add_header('Content-Type', 'application/json')
    for r in json.loads(urllib.request.urlopen(req).read()).get('runs', []):
        rid = r['info']['run_id']
        upd = json.dumps({'run_id': rid, 'status': 'FINISHED'}).encode()
        req2 = urllib.request.Request(f'{base}/api/2.0/mlflow/runs/update', data=upd, method='POST')
        req2.add_header('Content-Type', 'application/json')
        urllib.request.urlopen(req2)
        print(f'Closed: {rid[:12]}')
"
```

### Checkpoint bị mất sau rsync

Sbatch submit tự chạy `rsync --delete` từ VPS → Eagle. Nếu checkpoint nằm trong `ml/`
thì sẽ bị xóa vì VPS không có file đó.

**Phòng tránh:** Checkpoint đã được lưu ngoài `ml/` (trong `checkpoints/{job_id}/`).

**Nếu đã mất — recover từ MLflow:**
1. Mở MLflow → click run → tab **Artifacts**
2. Download `best.pt`

Hoặc dùng API:

```bash
# Trên Eagle
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://207.180.243.242:5000')
client = mlflow.tracking.MlflowClient()
client.download_artifacts('PASTE_RUN_ID', 'best.pt', '/path/to/save/')
"
```

### Code mới không được nhận trên Eagle

Sbatch template tự chạy `pip install -e .` mỗi lần submit. Nếu vẫn lỗi:

```bash
# SSH vào Eagle
ssh tiencd1234@eagle.man.poznan.pl
cd ~/pl0501-01/project_data/rf-worldpose
source .venv/bin/activate
pip install -e ml/
```

### Cancel job trên Eagle

**Cách 1 — Portal:** click job → nút **Cancel**

**Cách 2 — API:**

```bash
curl -X POST http://207.180.243.242:8080/api/v1/hpc/training-jobs/{job_id}/cancel
```

**Cách 3 — SSH trực tiếp:**

```bash
ssh tiencd1234@eagle.man.poznan.pl "scancel {slurm_job_id}"
```
