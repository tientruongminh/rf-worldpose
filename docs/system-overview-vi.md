# RF-WorldPose — Tài Liệu Chi Tiết Hệ Thống

> **Phiên bản:** 1.0 · **Ngày:** 2026-05-10
>
> Nền tảng nhận dạng con người qua sóng WiFi CSI — từ firmware cảm biến, gateway xử lý tín hiệu, data lake, ML pipeline, đến model serving và monitoring.

---

## Mục Lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Firmware — Cảm biến ESP32-S3](#3-firmware--cảm-biến-esp32-s3)
4. [Edge Gateway — Cổng thu nhận (Rust)](#4-edge-gateway--cổng-thu-nhận-rust)
5. [Control Plane API — Trung tâm điều khiển (FastAPI)](#5-control-plane-api--trung-tâm-điều-khiển-fastapi)
6. [Data Lake — Kho dữ liệu 3 tầng](#6-data-lake--kho-dữ-liệu-3-tầng)
7. [ETL Pipeline — Đường ống xử lý dữ liệu (Dagster)](#7-etl-pipeline--đường-ống-xử-lý-dữ-liệu-dagster)
8. [ML — Mô hình trí tuệ nhân tạo (PyTorch)](#8-ml--mô-hình-trí-tuệ-nhân-tạo-pytorch)
9. [Helios GH200 — Siêu máy tính huấn luyện](#9-helios-gh200--siêu-máy-tính-huấn-luyện)
10. [Model Serving — Triển khai suy luận](#10-model-serving--triển-khai-suy-luận)
11. [Dashboard — Giao diện quản trị (Next.js)](#11-dashboard--giao-diện-quản-trị-nextjs)
12. [Infrastructure — Hạ tầng dịch vụ](#12-infrastructure--hạ-tầng-dịch-vụ)
13. [Database Schema](#13-database-schema)
14. [Luồng vận hành end-to-end](#14-luồng-vận-hành-end-to-end)
15. [Cách chạy local](#15-cách-chạy-local)
16. [Cấu trúc thư mục](#16-cấu-trúc-thư-mục)

---

## 1. Tổng quan dự án

### Bài toán

Nhận biết sự hiện diện, hành động, và tư thế cơ thể con người thông thường phụ thuộc vào camera. RF-WorldPose giải quyết bài toán khó hơn: **suy luận tất cả thông tin đó chỉ từ tín hiệu WiFi**, không cần camera trong quá trình sử dụng thực tế.

### Nguyên lý hoạt động

Khi sóng WiFi truyền qua không gian, cơ thể con người làm thay đổi đặc tính kênh truyền (Channel State Information — CSI). Bằng cách phân tích sự thay đổi biên độ và pha trên từng subcarrier của tín hiệu WiFi, mô hình AI có thể học được:

- **Có người hay không** (presence detection)
- **Đang làm gì** (activity recognition — đi, đứng, ngồi, nằm, vẫy tay...)
- **Tư thế cơ thể** (skeleton pose — 17 keypoints 3D)
- **Bề mặt cơ thể** (DensePose — mục tiêu tương lai)

### Tại sao không chỉ dùng camera?

| Tiêu chí | Camera | WiFi CSI |
|---|---|---|
| Quyền riêng tư | Xâm phạm | Không ghi hình |
| Hoạt động trong tối | Không (trừ IR) | Có |
| Xuyên tường/vật cản | Không | Có (sóng RF) |
| Chi phí phần cứng | Trung bình | Rất thấp (ESP32 ~$5) |
| Độ chính xác hiện tại | Rất cao | Đang nghiên cứu |

Camera vẫn được sử dụng trong giai đoạn huấn luyện (teacher) để tạo label — nhưng khi triển khai thực tế, hệ thống chỉ cần WiFi.

---

## 2. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────┐
│  4x ESP32-S3 CSI Nodes                              │
│  Thu tín hiệu WiFi, đóng gói binary, stream UDP    │
└────────────────────┬────────────────────────────────┘
                     │ UDP packets (binary, CRC32)
                     ▼
┌─────────────────────────────────────────────────────┐
│  Rust Edge Gateway                                  │
│  Decode, validate CRC, buffer SQLite, publish NATS  │
│  Upload batch → MinIO Bronze                        │
│  Edge ONNX inference (tuỳ chọn)                     │
└─────────┬──────────┬──────────┬─────────────────────┘
          │          │          │
          ▼          ▼          ▼
┌─────────────┐ ┌────────┐ ┌──────────────────────────┐
│ NATS        │ │ SQLite │ │ MinIO/S3                  │
│ JetStream   │ │ Buffer │ │ Bronze Data Lake          │
│ (realtime)  │ │(local) │ │ (immutable raw)           │
└─────────────┘ └────────┘ └──────────┬───────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────┐
│  Dagster ETL Pipeline                               │
│  Bronze → Silver (decode, parquet)                  │
│  Silver → Gold (join labels, sliding window)        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI Control Plane                              │
│  Quản lý: deployments, nodes, sessions, datasets,  │
│  training jobs, model versions                      │
│  PostgreSQL metadata store                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Helios GH200 Training Backend                      │
│  Slurm batch: train, eval, export ONNX             │
│  4x NVIDIA GH200 96GB (ARM aarch64)                │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  MLflow Model Registry                              │
│  Candidate → Staging → Production → Archived        │
│  Eval gates trước khi promote                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Inference Deployment                               │
│  Edge: ONNX Runtime trên gateway                    │
│  Cloud: Triton + TensorRT                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Monitoring & Observability                         │
│  Prometheus + Grafana + Loki                        │
│  Alerts: node offline, drop rate, drift, latency    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Firmware — Cảm biến ESP32-S3

**Thư mục:** `firmware/esp32-csi-node/`

### Vai trò

4 con chip ESP32-S3 đặt trong phòng, mỗi con hoạt động như một "antenna" thu sóng WiFi CSI. Chúng liên tục bắt tín hiệu, đóng gói thành packet nhị phân, và gửi qua UDP đến gateway.

### Packet format (32 bytes header + payload + 4 bytes CRC)

```
┌────────────────────────────────────────────────────┐
│ Header (32 bytes, little-endian, packed)            │
├────────────────────────────────────────────────────┤
│ magic            u32   0xC5110001                   │
│ protocol_version u8    1                            │
│ node_id          u8    0-255 (ID chip)              │
│ header_len       u16   32                           │
│ seq              u32   Số thứ tự tăng dần           │
│ timestamp_us     u64   Microsecond timestamp        │
│ rssi             i8    Cường độ tín hiệu (dBm)     │
│ noise_floor      i8    Nền nhiễu (dBm)             │
│ channel          u8    Kênh WiFi                    │
│ flags            u8    Reserved                     │
│ n_subcarriers    u16   Số subcarrier (thường 56)    │
│ firmware_version u16   Phiên bản firmware           │
│ payload_len      u32   Kích thước payload (bytes)   │
├────────────────────────────────────────────────────┤
│ Payload: n_subcarriers × 2 giá trị int16           │
│ Mỗi subcarrier = (I, Q) — In-phase, Quadrature    │
├────────────────────────────────────────────────────┤
│ CRC32            u32   Checksum toàn bộ phía trên  │
└────────────────────────────────────────────────────┘
```

### File chính

| File | Mô tả |
|---|---|
| `main/main.c` | Entry point: init NVS, WiFi STA mode, khởi động CSI collector |
| `main/csi_packet.h` | Định nghĩa struct `rfpose_csi_header_t` (32 bytes packed), `rfpose_csi_frame_t`, hàm encode/CRC |
| `main/csi_collector.h` | Header cho CSI callback và UDP streamer |
| `provision.py` | Script Python để provision WiFi credentials qua serial |
| `test/` | Unit test C cho packet encoding (native, không cần ESP32) |

### Các hằng số quan trọng

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `RFPOSE_CSI_MAGIC` | `0xC5110001` | Magic number nhận diện packet |
| `RFPOSE_CSI_HEADER_LEN` | `32` | Kích thước header cố định |
| `RFPOSE_CSI_PROTOCOL_VERSION` | `1` | Phiên bản protocol |
| `RFPOSE_CSI_MAX_SUBCARRIERS` | `256` | Giới hạn tối đa subcarrier |

### Quy tắc firmware

- Không dùng Arduino cho production
- Không gửi JSON raw (phải là binary packed)
- Mỗi node phải có ID duy nhất
- Mỗi packet phải có `seq` và `timestamp_us`
- OTA update phải có signing và rollback

---

## 4. Edge Gateway — Cổng thu nhận (Rust)

**Thư mục:** `gateway/rf-gateway/`

### Vai trò

Gateway là trạm thu nhận trung tâm, viết bằng Rust/Tokio cho hiệu năng cao. Nó thực hiện 6 chức năng:

1. **Nhận packet UDP** từ 4 ESP32 trên port 5006
2. **Validate** — kiểm tra magic number, CRC32, chiều dài packet
3. **Theo dõi health** — đếm packet, phát hiện drop qua sequence number
4. **Buffer local** — lưu vào SQLite chống mất dữ liệu khi mạng lỗi
5. **Publish realtime** — gửi lên NATS JetStream
6. **Upload Bronze** — batch upload lên MinIO/S3 theo định kỳ

### Module chi tiết

#### `src/packet/mod.rs` — Packet decoder

Decode binary packet từ ESP32, trích xuất tất cả fields, tính amplitude từ I/Q:

```
amplitude[i] = sqrt(I[i]² + Q[i]²)
```

Validate: magic number, header length, CRC32, payload alignment, subcarrier count.

**Struct chính:**

| Struct | Mô tả |
|---|---|
| `CsiPacket` | Packet đã decode: node_id, seq, timestamp_us, rssi, iq, amplitude, crc32 |
| `NodeHealth` | Trạng thái node: packets_received, packets_dropped_est, last_rssi |

#### `src/buffer/mod.rs` — SQLite local buffer

Bảng `csi_packets` trong SQLite:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `received_at_ms` | INTEGER | Thời điểm gateway nhận |
| `node_id` | INTEGER | ID node ESP32 |
| `seq` | INTEGER | Sequence number |
| `timestamp_us` | INTEGER | Timestamp từ ESP32 |
| `rssi` | INTEGER | Cường độ tín hiệu |
| `channel` | INTEGER | Kênh WiFi |
| `n_subcarriers` | INTEGER | Số subcarrier |
| `packet_json` | TEXT | Toàn bộ packet serialized JSON |
| `uploaded_at_ms` | INTEGER | NULL = chưa upload, có giá trị = đã upload |

Ba operations: `insert_packet`, `pending_packets` (lấy batch chưa upload), `mark_uploaded`.

#### `src/nats/mod.rs` — NATS publisher

Publish lên 2 topic:
- `csi.raw.{deployment_id}.node-{NN}` — mỗi packet CSI
- `node.health.{deployment_id}` — health report mỗi 50 packets

#### `src/uploader/mod.rs` — S3/MinIO Bronze uploader

Upload batch packets lên MinIO theo path chuẩn:

```
bronze/deployment={id}/date=YYYY-MM-DD/csi_raw/batch-{timestamp}-{uuid}.json
```

Mỗi batch là JSON với schema `rfpose.bronze.csi_batch.v1`.

Background task chạy mỗi 30 giây (configurable), upload tối đa 500 packets/batch.

#### `src/metrics.rs` — Prometheus metrics

3 counter atomic: `packets_ok`, `packets_bad`, `packets_uploaded`.

#### `src/inference.rs` — Edge inference (stub)

Placeholder cho ONNX Runtime inference trên gateway. Nhận một window CSI packets, trả về `InferenceOutput {action, confidence}`.

### Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `RFPOSE_GATEWAY_BIND` | `0.0.0.0:5006` | Địa chỉ UDP bind |
| `RFPOSE_DEPLOYMENT_ID` | `room01` | ID deployment/phòng |
| `RFPOSE_GATEWAY_SQLITE` | `rf-gateway-buffer.sqlite` | Đường dẫn SQLite |
| `NATS_URL` | (tuỳ chọn) | URL NATS server |
| `S3_BUCKET` | (tuỳ chọn) | Bucket MinIO/S3 |
| `S3_ENDPOINT_URL` | (tuỳ chọn) | Endpoint MinIO |
| `RFPOSE_UPLOAD_INTERVAL_SECS` | `30` | Chu kỳ upload (giây) |
| `RFPOSE_UPLOAD_BATCH_SIZE` | `500` | Số packet mỗi batch |
| `RFPOSE_ONNX_MODEL` | (tuỳ chọn) | Đường dẫn model ONNX |

---

## 5. Control Plane API — Trung tâm điều khiển (FastAPI)

**Thư mục:** `services/api/`

### Vai trò

API trung tâm quản lý toàn bộ vòng đời dữ liệu và model. Chạy tại port **8080**, Swagger UI tại `/docs`.

### Các router (endpoint groups)

#### 5.1 Deployments — Quản lý triển khai

Một "deployment" là một phòng/vị trí triển khai cảm biến.

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/deployments` | Tạo/cập nhật deployment |
| `GET` | `/api/v1/deployments/{id}/status` | Xem trạng thái + danh sách nodes |
| `PUT` | `/api/v1/deployments/{id}/nodes/{node_id}` | Đăng ký/cập nhật node |

#### 5.2 Recording Sessions — Phiên thu thập

Một session là khoảng thời gian thu thập dữ liệu (ví dụ: 10 phút thu CSI + video).

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/recording-sessions` | Tạo session mới (status = `recording`) |
| `POST` | `/api/v1/recording-sessions/{id}/finish` | Kết thúc session, gắn `bronze_uri` |

#### 5.3 Datasets — Quản lý tập dữ liệu

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/datasets` | Đăng ký dataset version |

#### 5.4 Training Jobs — Quản lý huấn luyện

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/training-jobs` | Tạo training job (status = `created`) |
| `GET` | `/api/v1/training-jobs/{id}` | Xem trạng thái job |
| `POST` | `/api/v1/training-jobs/{id}/mark-submitted` | Đánh dấu đã submit lên Slurm |

#### 5.5 Models — Quản lý model

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/models` | Đăng ký model version mới |
| `POST` | `/api/v1/models/{id}/promote` | Promote: candidate → staging → production |
| `POST` | `/api/v1/models/{id}/rollback` | Rollback model |

Khi promote lên `production`, model production cũ tự động chuyển thành `archived`.

#### 5.6 Helios — HPC job submission

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/v1/helios/submit` | Submit training job lên Helios Slurm |
| `GET` | `/api/v1/helios/jobs/{id}/status` | Kiểm tra trạng thái Slurm job |

### Pydantic Schemas

| Schema | Dùng cho |
|---|---|
| `DeploymentCreate` | Tạo deployment: id, name, room_id, metadata |
| `NodeUpsert` | Đăng ký node: id, deployment_id, hardware_revision, firmware_version, position, status |
| `RecordingSessionCreate` | Tạo session: id, deployment_id, label, metadata |
| `DatasetVersionCreate` | Đăng ký dataset: id, source_sessions, preprocess_version, teacher_version, artifact_uri, stats |
| `TrainingJobCreate` | Tạo training job: id, dataset_version, train_config, backend |
| `ModelVersionCreate` | Đăng ký model: id, name, artifact_uri, metrics, hash |

---

## 6. Data Lake — Kho dữ liệu 3 tầng

**Lưu trữ:** MinIO/S3 (bucket `rfpose`)

### Mô hình Medallion Architecture

```
Bronze (raw)  →  Silver (cleaned)  →  Gold (ML-ready)
```

#### Bronze — Dữ liệu thô

```
s3://rfpose/bronze/
  deployment=room01/
    date=2026-05-02/
      csi_raw/
        batch-1714663800123-abc123.json    ← Gateway upload
      node_health/
      events/
      video/
      metadata.json
```

Mỗi file Bronze là JSON batch:

```json
{
  "schema": "rfpose.bronze.csi_batch.v1",
  "deployment_id": "room01",
  "uploaded_at": "2026-05-02T10:30:00Z",
  "packets": [
    {
      "id": 1,
      "received_at_ms": 1714663800123,
      "node_id": 1,
      "seq": 42,
      "timestamp_us": 1714663800000000,
      "packet_json": "{...}"
    }
  ]
}
```

**Quy tắc:** Bronze là immutable — chỉ ghi thêm, không bao giờ sửa.

#### Silver — Dữ liệu đã xử lý

```
s3://rfpose/silver/
  deployment=room01/
    session=001/
      csi_decoded.parquet      ← Polars DataFrame
      pose_labels.parquet      ← Teacher labels
      events.parquet
      quality_report.json      ← Thống kê chất lượng
```

Mỗi row trong `csi_decoded.parquet`:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `deployment_id` | string | ID deployment |
| `node_id` | int | ID node |
| `seq` | int | Sequence number |
| `timestamp_us` | int | Timestamp |
| `rssi` | int | Cường độ tín hiệu |
| `noise_floor` | int | Nền nhiễu |
| `channel` | int | Kênh WiFi |
| `n_subcarriers` | int | Số subcarrier |
| `amplitude` | list[float] | Biên độ mỗi subcarrier |
| `firmware_version` | int | Phiên bản firmware |
| `source_file` | string | File Bronze gốc |

`quality_report.json` chứa: tổng rows, danh sách node_ids, ước tính packet drops.

#### Gold — Dữ liệu sẵn sàng huấn luyện

```
s3://rfpose/gold/
  dataset_version=rfpose-room01-v0007/
    train/               ← Training split
    val/                 ← Validation split
    test/                ← Test split
    manifest.json        ← Metadata
    stats.json           ← Thống kê dataset
    normalization.json   ← Mean/std cho normalize
```

Mỗi sample Gold là một **sliding window**: [N_nodes × T_frames × S_subcarriers × C_channels].

---

## 7. ETL Pipeline — Đường ống xử lý dữ liệu (Dagster)

**Thư mục:** `pipelines/dagster/`

### Dagster UI: http://localhost:3004

### Assets & ETL modules

| Module | File | Mô tả |
|---|---|---|
| Bronze → Silver | `etl/bronze_to_silver.py` | Đọc JSON Bronze, decode từng packet, tính quality report, xuất Parquet |
| Silver → Gold | `etl/silver_to_gold.py` | Cắt sliding window, chia train/val/test, tính normalization stats |
| Join Labels | `etl/join_labels.py` | Ghép CSI data với pose labels từ camera teacher |
| Data Lake Assets | `assets/data_lake.py` | Dagster asset definitions cho Bronze, Silver, Gold |
| Training Jobs | `jobs/training_jobs.py` | Job definition cho Dagster-triggered training |
| Hard Case Sensor | `sensors/hard_case_sensor.py` | Tự động phát hiện và thu thập mẫu khó (low confidence, drift) |

### Bronze → Silver chi tiết

```python
bronze_to_silver(bronze_root, silver_out):
    1. Duyệt tất cả file JSON trong bronze_root
    2. Với mỗi packet: decode fields, extract amplitude
    3. Tạo Polars DataFrame, xuất Parquet
    4. Tính quality_report: row count, node list, seq drop estimate
    5. Ghi quality_report.json cạnh file Parquet
```

### Quality gates

Dữ liệu phải đạt các tiêu chí trước khi vào Gold:
- `node_count == 4` — đủ 4 node
- `packet_drop_rate < threshold` — tỷ lệ mất packet thấp
- `fps_stability` — tốc độ thu ổn định
- `rssi_range` — cường độ tín hiệu hợp lệ
- `timestamp_monotonic` — timestamp luôn tăng
- `label_confidence > threshold` — nhãn teacher đủ tin cậy

---

## 8. ML — Mô hình trí tuệ nhân tạo (PyTorch)

**Thư mục:** `ml/rfpose/`

### 8.1 Kiến trúc model: RFWorldPose

```
Input: [B, N_nodes, T_frames, S_subcarriers, C_channels]
  ví dụ: [16, 4, 60, 56, 2]  (batch 16, 4 nodes, 60 frames, 56 subcarriers, I/Q)

  ┌──────────────────┐
  │ CSI Tokenizer    │  Linear(channels → dim) + pool subcarriers
  │                  │  [B, N*T, D] = [16, 240, 128]
  └────────┬─────────┘
           │ + positional bias (node_embed + time_embed)
           ▼
  ┌──────────────────┐
  │ RF Graph         │  Transformer Encoder
  │ Transformer      │  4 layers, 4 heads, GELU, pre-norm
  │                  │  dim=128, ffn=512, dropout=0.1
  └────────┬─────────┘
           │ mean pooling → LayerNorm
           ▼
  ┌──────────────────┐
  │ Task Heads       │
  │                  │
  │ action_head      │  Linear(128 → 6)   → phân loại hành động
  │ presence_head    │  Linear(128 → 1)   → phát hiện người
  │ keypoint_head    │  Linear(128 → 51)  → 17 keypoints × 3 toạ độ
  │ embedding        │  Pooled vector      → dùng cho downstream tasks
  └──────────────────┘
```

### 8.2 LoRA — Tinh chỉnh nhẹ

**File:** `models/lora.py`

LoRA (Low-Rank Adaptation) cho phép fine-tune model cho phòng mới mà chỉ cần thêm rất ít tham số:

- Đóng băng toàn bộ base model
- Thêm adapter LoRA (rank=8, alpha=16) vào các task heads
- Chỉ cần dữ liệu ít (vài trăm sample) để adapt

```
Output = Base(x) + LoRA_B(LoRA_A(dropout(x))) × (alpha/rank)
```

Áp dụng cho: `action_head`, `presence_head`, `keypoint_head`.

### 8.3 Training

| File | Mô tả |
|---|---|
| `training/train.py` | Training script chính: AdamW + CosineAnnealing LR, cross-entropy loss, best checkpoint |
| `training/train_hydra.py` | Training với Hydra config (yaml-driven, reproducible) |
| `training/distill.py` | Knowledge distillation: nén model lớn thành nhỏ cho edge |

**Training flow:**
1. Load dataset Gold từ `CsiWindowDataset`
2. Khởi tạo `RFWorldPose` model
3. Train với `AdamW(lr=3e-4, weight_decay=1e-4)` + `CosineAnnealingLR`
4. Mỗi epoch: train → validate → lưu best checkpoint
5. Xuất `best.pt`, `history.json`, `metrics.json`

### 8.4 Evaluation

| File | Mô tả |
|---|---|
| `evaluation/eval.py` | Đánh giá model trên test set |
| `evaluation/eval_gate.py` | Tự động kiểm tra model có đạt ngưỡng chất lượng hay không |

**Eval gates** (model phải đạt tất cả mới được promote):
- `presence_f1 >= threshold`
- `action_acc >= threshold`
- `keypoint_error <= threshold`
- `latency_p95 <= threshold`

### 8.5 Export & Packaging

| File | Mô tả |
|---|---|
| `export/onnx.py` | Xuất PyTorch model sang ONNX format cho inference |
| `packaging/model_card.py` | Tạo model card markdown với metadata, metrics, dataset info |

**Artifact package:**
```
model.pt                ← PyTorch checkpoint
model.onnx              ← ONNX Runtime format
normalization.json      ← Mean/std cho preprocessing
model_config.yaml       ← Hyperparameters
metrics.json            ← Evaluation metrics
model_card.md           ← Documentation tự động
SHA256 manifest         ← Hash verification
```

### 8.6 Dataset

**File:** `data/window_dataset.py`

`CsiWindowDataset` — PyTorch Dataset cho sliding window CSI:

| Config | Mặc định | Mô tả |
|---|---|---|
| `num_nodes` | 4 | Số node ESP32 |
| `window_frames` | 60 | Số frame trong 1 window (~3 giây ở 20Hz) |
| `n_subcarriers` | 56 | Số subcarrier WiFi |
| `channels` | 2 | I/Q channels |
| `num_classes` | 6 | Số loại hành động |

---

## 9. Helios GH200 — Siêu máy tính huấn luyện

**Thư mục:** `helios_runner/`

### Vai trò

Helios là cụm siêu máy tính HPC tại Cyfronet (Ba Lan) với node NVIDIA Grace Hopper GH200. Dự án sử dụng nó làm **batch training backend** — không phải production server.

### Thông số

| Thuộc tính | Giá trị |
|---|---|
| Login node | `login01.helios.cyfronet.pl` |
| Partition | `plgrid-gpu-gh200` |
| GPU | 4× NVIDIA GH200 96GB HBM3 |
| CPU | Grace ARM aarch64 |
| OS | Rocky Linux 9 |
| Scheduler | Slurm |
| Time limit | 48 giờ |

### Integration flow

```
1. Control Plane tạo training_job
2. Helios submitter render sbatch template
3. SSH → login01.helios.cyfronet.pl
4. sbatch → plgrid-gpu-gh200 queue
5. GH200 job pull dataset từ MinIO/S3
6. torchrun train trên 4 GH200
7. Eval + export ONNX
8. Upload artifacts → S3/MLflow
9. Register model candidate
```

### Quy tắc HPC quan trọng

- `$SCRATCH` là temporary — không bao giờ là source of truth
- Phải checkpoint thường xuyên (48h time limit)
- Build/runtime phải tương thích aarch64 ARM
- Tất cả artifacts phải upload về S3/MLflow sau khi xong

---

## 10. Model Serving — Triển khai suy luận

**Thư mục:** `services/inference/`

### Hai chế độ

#### Edge mode — Trên gateway

```
CSI stream → online preprocess → ring buffer 3s
  → inference mỗi 100-200ms → temporal smoother
  → confidence gate → output
```

- Dùng ONNX Runtime
- Ưu điểm: latency thấp, hoạt động offline, bảo mật dữ liệu
- Nhược điểm: giới hạn model size

#### Cloud mode — Trên server

- Dùng NVIDIA Triton + TensorRT
- Ưu điểm: model lớn, batched inference, GPU acceleration
- Nhược điểm: cần network, latency cao hơn

### Output format

```json
{
  "deployment_id": "room01",
  "model_version": "rfworldpose-v0.4.12",
  "timestamp": 1714663800.123,
  "presence": true,
  "action": "walking",
  "confidence": 0.82,
  "keypoints": [[0.5, 0.3, 1.2], ...],
  "quality": {
    "nodes_online": 4,
    "packet_drop_rate": 0.01,
    "drift_score": 0.08
  }
}
```

---

## 11. Dashboard — Giao diện quản trị (Next.js)

**Thư mục:** `dashboard/`

### Tech stack

- **Next.js** — React framework
- **Three.js** — 3D visualization (skeleton rendering)
- **TypeScript** — Type-safe frontend

### Màn hình dự kiến

| Màn hình | Mô tả |
|---|---|
| Live Skeleton | Hiển thị 3D skeleton realtime từ inference output |
| Node Health | Trạng thái 4 node ESP32: online/offline, RSSI, packet rate |
| CSI Raw Graph | Biểu đồ sóng CSI amplitude theo thời gian |
| Recording Control | Bắt đầu/kết thúc phiên thu thập |
| Dataset Sessions | Danh sách sessions, quality status |
| Training Jobs | Danh sách training jobs, status, metrics |
| Model Registry | Model versions, promote/rollback |
| Alerts | Cảnh báo hệ thống |

---

## 12. Infrastructure — Hạ tầng dịch vụ

### Docker Compose Stack (local dev)

**File:** `infra/docker-compose/docker-compose.yml`

| Service | Image | Port | Vai trò |
|---|---|---|---|
| **PostgreSQL 16** | `postgres:16` | 5432 | Metadata store: deployments, sessions, datasets, training jobs, models |
| **NATS 2.10** | `nats:2.10` | 4222, 8222 | Message broker: CSI realtime stream, health events |
| **MinIO** | `minio/minio` | 9000, 9003 | Object storage S3-compatible: Bronze/Silver/Gold data lake, model artifacts |
| **FastAPI** | `python:3.11-slim` | 8080 | Control plane API |
| **MLflow** | `ghcr.io/mlflow/mlflow` | 5000 | Experiment tracking, model registry |
| **Dagster** | `python:3.11-slim` | 3004 | ETL pipeline orchestration |
| **Prometheus** | `prom/prometheus` | 9090 | Metrics collection |
| **Grafana** | `grafana/grafana` | 3002 | Dashboard visualization |
| **Loki** | `grafana/loki:2.9.8` | 3100 | Log aggregation |

### Kubernetes (production)

**Thư mục:** `infra/k8s/base/` — Base manifests cho multi-service deployment.

### Monitoring

**Thư mục:** `infra/monitoring/prometheus/`

Metrics thu thập:
- ESP32 node health (online/offline, RSSI)
- Gateway packet rate, drop rate
- CSI drift detection
- Model confidence, inference latency
- Training job status
- GPU utilization (GH200)
- Storage usage

Alerts cảnh báo:
- Node offline > 60 giây
- Packet drop > 10%
- Inference latency p95 > 200ms
- Drift score cao bất thường
- Disk < 15%
- Model confidence sụp đổ

---

## 13. Database Schema

**File:** `infra/postgres/migrations/001_initial.sql`

### Bảng `deployments` — Triển khai

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID triển khai (vd: `room01`) |
| `name` | TEXT | Tên hiển thị |
| `room_id` | TEXT | ID phòng |
| `status` | TEXT | `created`, `active`, `archived` |
| `metadata` | JSONB | Thông tin bổ sung |
| `created_at` | TIMESTAMPTZ | Thời điểm tạo |

### Bảng `nodes` — Node cảm biến

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID node (vd: `node-01`) |
| `deployment_id` | TEXT FK | Thuộc deployment nào |
| `hardware_revision` | TEXT | Phiên bản phần cứng |
| `firmware_version` | TEXT | Phiên bản firmware |
| `position` | JSONB | Vị trí đặt `{x, y, z}` |
| `status` | TEXT | `online`, `offline`, `unknown` |
| `last_seen_at` | TIMESTAMPTZ | Lần cuối liên lạc |

### Bảng `recording_sessions` — Phiên thu thập

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID session |
| `deployment_id` | TEXT FK | Thuộc deployment nào |
| `label` | TEXT | Nhãn (vd: `walking`, `sitting`) |
| `status` | TEXT | `created`, `recording`, `finished` |
| `started_at` | TIMESTAMPTZ | Bắt đầu thu |
| `ended_at` | TIMESTAMPTZ | Kết thúc thu |
| `bronze_uri` | TEXT | Đường dẫn S3 tới dữ liệu Bronze |
| `quality_status` | TEXT | `unknown`, `pass`, `fail` |

### Bảng `dataset_versions` — Phiên bản dataset

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID (vd: `rfpose-room01-v0007`) |
| `source_sessions` | JSONB | Danh sách session IDs |
| `preprocess_version` | TEXT | Phiên bản pipeline xử lý |
| `teacher_version` | TEXT | Phiên bản model teacher (ViTPose, etc.) |
| `artifact_uri` | TEXT | Đường dẫn S3 |
| `stats` | JSONB | Thống kê dataset |
| `quality_report_uri` | TEXT | Báo cáo chất lượng |

### Bảng `training_jobs` — Job huấn luyện

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID job |
| `dataset_version` | TEXT FK | Dataset dùng để train |
| `train_config` | TEXT | Cấu hình training (Hydra config) |
| `backend` | TEXT | `helios-slurm` hoặc `local` |
| `slurm_job_id` | TEXT | Slurm job ID trên Helios |
| `slurm_partition` | TEXT | `plgrid-gpu-gh200` |
| `status` | TEXT | `created` → `submitted` → `running` → `completed`/`failed` |
| `artifact_uri` | TEXT | Đường dẫn model artifacts |
| `eval_report_uri` | TEXT | Báo cáo evaluation |

### Bảng `model_versions` — Phiên bản model

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | TEXT PK | ID (vd: `rfworldpose-v0.4.12`) |
| `name` | TEXT | Tên model |
| `status` | TEXT | `candidate` → `staging` → `production` → `archived` |
| `dataset_version` | TEXT FK | Dataset đã train |
| `training_job_id` | TEXT FK | Job đã train |
| `artifact_uri` | TEXT | Đường dẫn artifacts S3 |
| `metrics` | JSONB | `{accuracy, f1, keypoint_error, ...}` |
| `hash` | TEXT | SHA256 hash cho verification |
| `promoted_at` | TIMESTAMPTZ | Thời điểm promote |

### Quan hệ giữa các bảng

```
deployments ──< nodes
deployments ──< recording_sessions
                recording_sessions ──> dataset_versions (qua source_sessions JSONB)
                                       dataset_versions ──< training_jobs
                                                           training_jobs ──< model_versions
```

---

## 14. Luồng vận hành end-to-end

### Flow 1: Thu thập dữ liệu

```
1. Operator tạo deployment (POST /api/v1/deployments)
2. Đăng ký 4 nodes (PUT /api/v1/deployments/{id}/nodes/{node_id})
3. Tạo recording session (POST /api/v1/recording-sessions)
4. ESP32 bắt đầu stream CSI → Gateway
5. Gateway validate, buffer, upload Bronze → MinIO
6. Kết thúc session (POST /api/v1/recording-sessions/{id}/finish)
7. Dagster chạy ETL: Bronze → Silver (decode, quality check)
8. Camera teacher tạo pose labels
9. Join CSI + labels (Silver)
10. Silver → Gold: sliding window, split train/val/test
11. Đăng ký dataset version (POST /api/v1/datasets)
```

### Flow 2: Huấn luyện model

```
1. Tạo training job (POST /api/v1/training-jobs)
2. Helios submitter render Slurm sbatch script
3. SSH vào Helios, sbatch
4. GH200 pull dataset từ MinIO
5. torchrun train (PyTorch, DDP, mixed precision)
6. Evaluation trên test set
7. Export ONNX
8. Upload artifacts → MinIO + MLflow
9. Đăng ký model candidate (POST /api/v1/models)
10. Eval gate: kiểm tra metrics ≥ thresholds
11. Promote staging → production (POST /api/v1/models/{id}/promote)
```

### Flow 3: Triển khai & sử dụng

```
1. Gateway nhận thông báo model mới
2. Download model.onnx + normalization.json từ MinIO
3. Verify SHA256 hash
4. Warmup model (chạy dummy inference)
5. Canary: chạy song song model cũ và mới
6. Monitor confidence + latency
7. Nếu OK → full switch
8. Nếu fail → rollback (POST /api/v1/models/{id}/rollback)
```

### Flow 4: Feedback loop

```
1. Production phát hiện mẫu khó (low confidence, drift)
2. Hard case sensor (Dagster) lưu các mẫu này
3. Upload lên MinIO/S3
4. Tạo hard_case_dataset
5. Fine-tune LoRA adapter trên Helios
6. Eval against golden test set
7. Deploy improved model
```

---

## 15. Cách chạy local

### Yêu cầu

- Docker + Docker Compose
- `psql` (PostgreSQL client)
- Python 3.11+
- Rust toolchain (cho gateway)

### Bước 1: Khởi động infrastructure

```bash
cp .env.example .env
./scripts/dev_up.sh
```

### Bước 2: Chạy migrations

```bash
export DATABASE_URL=postgresql://rfpose:rfpose@localhost:5432/rfpose
./scripts/run_migrations.sh
```

### Bước 3: Khởi tạo MinIO

```bash
export MINIO_ENDPOINT=http://localhost:9000
export MINIO_ROOT_USER=rfpose
export MINIO_ROOT_PASSWORD=rfpose-secret
export S3_BUCKET=rfpose
./scripts/init_minio.sh
```

### Bước 4: Chạy Gateway

```bash
cd gateway/rf-gateway
RFPOSE_DEPLOYMENT_ID=room01 \
RFPOSE_GATEWAY_BIND=0.0.0.0:5006 \
RFPOSE_GATEWAY_SQLITE=/tmp/rfpose-gateway.sqlite \
NATS_URL=nats://localhost:4222 \
S3_BUCKET=rfpose \
S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=rfpose \
AWS_SECRET_ACCESS_KEY=rfpose-secret \
cargo run
```

### Bước 5: Gửi CSI giả lập

```bash
python tools/mock_sender/send_mock_csi.py --node-id 1 --count 100
```

### Truy cập các service

| Service | URL |
|---|---|
| API Swagger | http://localhost:8080/docs |
| MLflow | http://localhost:5000 |
| Dagster | http://localhost:3004 |
| MinIO Console | http://localhost:9003 |
| Grafana | http://localhost:3002 |
| Prometheus | http://localhost:9090 |
| NATS Monitor | http://localhost:8222 |

---

## 16. Cấu trúc thư mục

```
rf-worldpose/
├── firmware/
│   └── esp32-csi-node/          # ESP32-S3 CSI firmware (C, ESP-IDF)
│       ├── main/
│       │   ├── main.c           # Entry point
│       │   ├── csi_packet.h     # Packet format definition
│       │   └── csi_collector.h  # CSI callback + UDP streamer
│       ├── provision.py         # WiFi provisioning script
│       └── test/                # Native C unit tests
│
├── gateway/
│   └── rf-gateway/              # Rust edge gateway
│       ├── src/
│       │   ├── main.rs          # UDP listener, main loop
│       │   ├── packet/mod.rs    # Binary packet decoder + CRC
│       │   ├── buffer/mod.rs    # SQLite local buffer
│       │   ├── nats/mod.rs      # NATS JetStream publisher
│       │   ├── uploader/mod.rs  # S3/MinIO Bronze uploader
│       │   ├── metrics.rs       # Prometheus metrics
│       │   └── inference.rs     # Edge ONNX inference stub
│       └── tests/               # Integration tests
│
├── services/
│   ├── api/                     # FastAPI control plane
│   │   └── src/rfpose_api/
│   │       ├── main.py          # FastAPI app + routers
│   │       ├── routers/         # deployments, sessions, datasets, training, models, helios
│   │       ├── schemas/         # Pydantic models
│   │       └── db/              # PostgreSQL connection
│   ├── inference/               # Model serving service
│   ├── ingest/                  # Ingestion service
│   └── model-registry/          # Model registry service
│
├── pipelines/
│   └── dagster/                 # Dagster ETL pipeline
│       └── rfpose_pipelines/
│           ├── etl/             # bronze_to_silver, silver_to_gold, join_labels
│           ├── assets/          # Dagster asset definitions
│           ├── jobs/            # Training job definitions
│           └── sensors/         # Hard case detection sensor
│
├── ml/
│   └── rfpose/                  # PyTorch ML code
│       ├── models/
│       │   ├── rf_worldpose.py  # RFWorldPose Transformer model
│       │   └── lora.py          # LoRA adapter
│       ├── training/
│       │   ├── train.py         # Standard training loop
│       │   ├── train_hydra.py   # Hydra config-driven training
│       │   └── distill.py       # Knowledge distillation
│       ├── evaluation/
│       │   ├── eval.py          # Model evaluation
│       │   └── eval_gate.py     # Automated quality gates
│       ├── export/
│       │   └── onnx.py          # ONNX export
│       ├── packaging/
│       │   └── model_card.py    # Model card generation
│       └── data/
│           └── window_dataset.py # PyTorch Dataset
│
├── helios_runner/               # HPC Slurm job submitter
│   ├── rfpose_helios/           # Submit, status, cancel modules
│   └── templates/               # Slurm sbatch templates
│
├── dashboard/                   # Next.js web UI
│
├── infra/
│   ├── docker-compose/          # Local dev stack (9 services)
│   ├── postgres/migrations/     # SQL migrations
│   ├── monitoring/prometheus/   # Prometheus config
│   └── k8s/base/                # Kubernetes manifests
│
├── tools/
│   └── mock_sender/             # Synthetic CSI packet generator
│       └── send_mock_csi.py
│
├── data/                        # Sample/stub data
├── libs/                        # Shared libraries
├── scripts/                     # Dev scripts (dev_up, migrations, init_minio)
├── docs/                        # Documentation
├── .env.example                 # Environment template
├── Makefile                     # make up/down/fmt/test
└── README.md                    # Project overview
```
