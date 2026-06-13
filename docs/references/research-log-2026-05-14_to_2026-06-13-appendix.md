# Appendix — RF-WorldPose research log 2026-05-14 → 2026-06-13 (đầy đủ)

Main doc: [research-log-30d-2026-05-14_to_2026-06-13.md](../research-log-30d-2026-05-14_to_2026-06-13.md)

---

## A. Proto1 data contract

| Field | Value |
|-------|-------|
| Gold | `data/gold/rfpose-humanlike-v2-proto1/mmfi` |
| Split | Protocol 1 via `scripts/assign_subject_splits.py --protocol 1` |
| train / val / test | **10356 / 1500 / 1104** |
| Joints | 17 H36M |
| Subcarriers | 342 |
| Window | 60 |
| Preprocess | `normalize_csi`, `center_pose`, `root_joint=0` |
| CSI norm (reference) | mean `[39.955, 0.00646]`, std `[4.787, 0.585]` |
| Gold pose ckpt | `checkpoints/wimose-mmfi17j-proto1-v1/best.pt` |
| Best val MPJPE | **157.7 mm** (job **7401153**, epoch 7) |

### Action distribution (Proto1)

| Split | n | classes seen | majority `"nothing"` (id 25) | random@28 |
|-------|---|--------------|-------------------------------|-----------|
| train | 10356 | 22 | 22.8% (2364) | 3.6% |
| val | 1500 | 21 | **18.4%** (276) | 3.6% |
| test | 1104 | 22 | 21.7% (240) | 3.6% |

Top train after nothing: push 456, stand_up 444, wave 432, run 408.

MMFi A22–A27 → unified label `"nothing"`.

---

## B. All checkpoints (Eagle snapshot 2026-06-13)

| Checkpoint | ep | val MPJPE | val action acc | Ghi chú |
|------------|-----|-----------|----------------|---------|
| wimose-mmfi17j-proto1-v1 | 7 | **157.7 mm** | — | **Gold pose** |
| wimose-mmfi17j-proto1-action-v1 | 5 | 155.4 mm | — | multitask λ=0.5, cancelled lineage |
| wimose-mmfi17j-proto1-action-weighted-v1 best_action | 4 | 156.0 mm | **6.3%** | cancelled 7403230 |
| wimose-mmfi17j-proto1-action-weighted-v1 best | 3 | 155.4 mm | 3.7% | pose ckpt |
| wimose-mmfi17j-proto1-action-only-v1 | (running) | ~155.6 mm | **2.5%** ep2 | job 7403259 |
| wimose-mmfi17j-clean-v1 | 31 | 169.3 mm | — | split khác |
| wimose-mmfi17j-unif-v1 | 6 | 174.3 mm | — | |
| wimose-mmfi17j-gcn-v1 | 31 | 174.3 mm | — | |
| wimose-mmfi17j-fk-antcollapse-v1 | 19 | 281.2 mm | — | dead-end |
| wimose-mmfi17j-v1 | 1 | 318.5 mm | — | pre-humanlike |
| wimose-wipose18j-stable-v1 | 2 | 297.7 mm | — | COMPLETED 7400269 |
| rootrel-mmfi-v1 | 107 | ~304 mm eval | **91.28%** | unified-v2, 13j, 22 class eval |
| rootrel-wipose-v1 | — | ~289 mm viz | 85% viz only | 20 mẫu |

Eval JSON rootrel: `eval_results/rootrel-mmfi-v1-action-val.json` (job 7403253).

---

## C. WiMose MMFI jobs (Proto1 + variants)

| Job | State | Time | Best / last val MPJPE | Fail / note |
|-----|-------|------|----------------------|-------------|
| 7398936 | OOM | 8m | — | F1 |
| 7398974 | CANCELLED | 48m | ep6 523.3 mm | early |
| 7400250 | FAILED | 10h57m | **169.3 mm** ep61 | F2 SIGABRT, ckpt OK |
| 7401153 | FAILED | 7h37m | **157.7 mm** ep37 | F2 SIGABRT, **ckpt OK** |
| 7403052 | FAILED | 3h51m | 174.3 mm ep36 | unif |
| 7403044 | CANCELLED | 5h42m | 176.8 mm ep56 | GCN |
| 7403204 | RUNNING→? | >6h | 281 best; ~304 ep44 | F8 FK |
| 7403216 | CANCELLED | 1h38m | pose 155.4; acc **19.3%** | F10 linear CE |
| 7403230 | **CANCELLED** | 2h55m | pose 155.0; acc **6.3%** best | F7 weighted |
| 7403259 | **RUNNING** | >1h | ep2 pose 155.6; acc **2.5%** | action-only |

Epoch time Proto1: ~716 s (2×H100) hoặc ~725–1569 s (1×H100).

---

## D. Action experiments — epoch tables đầy đủ

### D1. 7403216 — linear CE + freeze + λ_action=0.5 (cancel ep8)

| Ep | val MPJPE | tr acc | val acc | Ghi chú |
|----|-----------|--------|---------|---------|
| 1 | 156.5 mm | 23.2% | **18.7%** | ≈ majority ngay ep1 |
| 5 | 155.4 mm | 22.8% | 18.5% | |
| 7 | 156.6 mm | 22.0% | **19.3%** | best acc |
| 8 | 156.1 mm | 22.2% | 18.4% | = majority exactly |

**Kết luận:** acc flat từ ep1 → predict gần như luôn `"nothing"` (18.4% val).

### D2. 7403230 — weighted CE λ=1.5 + freeze (cancel ~ep10)

| Ep | s/ep | val MPJPE | val acc | best_action |
|----|------|-----------|---------|-------------|
| 1 | 1666 | 156.0 mm | 6.3% | ✓ |
| 2 | 791 | 155.6 mm | 5.0% | |
| 3 | 725 | 155.4 mm | 3.7% | |
| 4 | 726 | 155.7 mm | 6.3% | ✓ |
| 5 | 725 | 155.0 mm | 4.6% | |
| 6–9 | ~725 | ~155.5 mm | 2.7–5.2% | regression |
| ~10 | — | — | — | **CANCELLED** user |

best_action.pt @ ep4: **6.3%**. Pose ~155 mm flat.

**Tại sao val MPJPE “giảm”?** Backbone + pose head **freeze** — không học pose. MPJPE ~155–156 mm ≈ ckpt gốc; nhích ~1 mm do BN running stats drift khi `model.train()` hoặc nhiễu metric, **không phải** cải thiện pose.

### D3. 7403259 — action_only + weighted CE + freeze (đang chạy)

Config: `wimose_mmfi17j_proto1_action_only_eagle.yaml`

| Ep | tr loss (CE) | val loss | val MPJPE | tr acc | val acc |
|----|--------------|----------|-----------|--------|---------|
| 1 | **3.020** | 3.185 | 156.2 mm | 5.9% | **4.9%** ✓ best |
| 2 | **3.040** | 3.286 | 155.6 mm | 6.0% | **2.5%** |

Trainable params: **57,372** (chỉ `action_head`).

**Ý nghĩa train loss ~3.0:** pure weighted CE trên 28 class. Random uniform ≈ **ln(28) ≈ 3.33**. Loss ~3.0 + acc ~5% = **gần random**, chưa học phân loại. Không so trực tiếp với ~4.5 job cũ (có thêm pose loss).

---

## E. rootrel multitask — phát hiện action 91% (2026-06-13)

### E1. Nguồn nhầm lẫn “85%”

| Nguồn | Metric | Thực chất |
|-------|--------|-----------|
| Job 7359079, 7359084 | Action Acc **85%** | `visualize_pose.py`, **20 mẫu random** |
| `viz_output/results.json` | 17/20 đúng | checkpoint `rootrel-mmfi-v1/best.pt` |
| MPJPE viz | ~289–290 mm | pose **không tốt** |

### E2. Eval thật — job 7403253

| | |
|---|---|
| Script | `eval_v2.py` |
| Checkpoint | `rootrel-mmfi-v1/best.pt` (ep 107) |
| Data | `gold/rfpose-unified-v2`, datasets=mmfi |
| Split | **val 1296 mẫu** |
| **Action accuracy** | **91.28%** |
| Macro F1 | **90.95%** |
| Macro precision / recall | 91.30% / 90.80% |
| Classes seen | 22 |
| MPJPE | ~304 mm |

**Kết luận:** rootrel **multitask train** (`lambda_action=0.5`, job 7342638) **học action tốt** trên unified-v2 — nhưng **không áp dụng** cho Proto1 WiMose (17 joints, 28 class, gold khác, arch khác).

---

## F. Team & legacy jobs

### F1. Proto1 aligned (2026-06-13)

| Job | Model | GPU | State | Error |
|-----|-------|-----|-------|-------|
| 7403207 | Ngân SSL | 1 | CANCELLED | — |
| 7403208 | Ngọc transformer | 1 | CANCELLED | — |
| 7403209 | ViT2D | 1 | FAILED | F4 ListConfig |
| 7403213 | Ngân SSL | 2 | FAILED | F3 DDP batch 0 |
| 7403214 | Ngọc transformer | 2 | FAILED | F3 DDP idx 102–118 |
| 7403215 | ViT2D | 2 | FAILED | F3 DDP |

Pre-crash: train=10356 val=1500, CSI norm OK, 1 batch ep0 rồi crash.

### F2. Pre-Proto1 (không so với 157.7 mm)

| Job | Model | Best metric |
|-----|-------|-------------|
| 7359067 | Ngân SSL | **607.8 mm** (PA 430.5) |
| ngoc logs | Transformer | 319.6 mm |
| 7351683 | ViT2D MMFI | 314.3 mm |
| 7342638 | rootrel unified | 305.9 mm train val |
| 7341939 | supervised 4gpu | 88.5 mm (scale suspect) |
| 7399066 | ViT2D clean | PA ~183.6 mm ep6, cancel |

---

## G. WiMAE / Diffusion / FK

| Job | State | Notes |
|-----|-------|-------|
| 7403053–7403065 | FAIL/OOM | early MAE |
| 7403195 | RUNNING | ep52 va_recon **0.538** |
| 7403196 | PENDING | diffusion afterok MAE |
| 7403204 | FK anti-collapse | val ~281–335 mm — không beat 157.7 |

---

## H. Code & config changes (2026-06-13 chiều, Eagle rsync)

| File | Thay đổi |
|------|----------|
| `train_wimose.py` | `action_only` mode; optimizer chỉ trainable params; checkpoint `task=action_classification` |
| `train_v2.py` | **Fix** load `pretrained_from`; `action_only` loss; early stop `val_action_acc` → `best_action.pt` |
| `wimose_mmfi17j_proto1_action_only_eagle.yaml` | Proto1 action-only config |
| `rootrel_mmfi_action_only_eagle.yaml` | unified-v2 action-only (chưa submit) |
| `scripts/wimose_mmfi_proto1_action_only.sbatch` | job **7403259** |
| `scripts/eval_wimose_action_only_test.sbatch` | job **7403260** pending |

**Bug đã fix:** `train_v2.py` đọc `pretrained_from` nhưng trước đó **không gọi** `load_pretrained_full` — chỉ load SSL encoder.

---

## I. Hành trình chronological narrative

### Tuần 1 — 28/05–05/06: Data foundation

- Loaders WiFiPose, adapters 6 dataset (Ngân).
- ETL Dagster: idempotent, parallel, stream fix OOM VPS 17GB.
- Silver redesign (1 row/sample, binary CSI `.npy`).
- Silver-Unified + memmap Gold X (tránh OOM stack).
- EDA MMFI, WiPose, WIAR (Hương).
- CSI tokenizer + transformer logic (Ngọc).

**Chưa train ML production.**

### 30/05–08/06: Platform

- Portal Eagle Quick Submit, Config/Model Registry, Inference API.
- CI/CD GitHub Actions, webhook deploy, Grafana, MLflow DNS fix.
- PR#2: ML pipeline end-to-end lần đầu.

### 08–10/06: Multi-arch v1 experiments

- Transformer Ngọc, ViT2D Hương (PR#4), SSL CNN Ngân (PR#6–9).
- Jobs 7359067 (SSL **607 mm**), 7351683 (ViT2D **314 mm**), ngoc transformer **~320 mm**.
- **rootrel-mmfi-v1** train (7342638): multitask pose+action, val_mpjpe ~**306 mm**; action val **không log** trong epoch summary.
- Viz jobs 7359079: **85% action** trên 20 mẫu — metric misleading, lưu lại đến 13/06 mới hiểu.

### 11–12/06: WiMose humanlike pivot

- Bulk merge **3562b01** / PR#3: WiMoseNet, humanlike gold v2, xóa legacy rootrel eval.
- WiMose MMFI: OOM (7398936) → long run clean **169.3 mm** (7400250, FAILED SIGABRT).
- **Proto1 align:** assign_subject_splits protocol 1 → **7401153** → **157.7 mm ep7** (FAILED SIGABRT, ckpt OK).
- Thử GCN/unif **174.3 mm** — không beat baseline.
- WiPose 18j ~296–314 mm.

### 13/06 sáng: Team Proto1 + action attempts

- Submit Ngân/Ngọc/Hương Proto1: **7403213–15 FAILED** DDP unused params.
- Action fine-tune WiMose pose ckpt:
  - **7403216** linear CE → **~19%** fake majority.
  - **7403230** weighted CE → **6.3%** best, regression.
- FK job 7403204, WiMAE 7403195 chạy song song — chưa competitive.

### 13/06 trưa–chiều: Clarify action + action-only

- Eval rootrel full val **7403253** → **91.28%** — tách khỏi viz 85%.
- Implement action_only + fix train_v2 pretrained.
- Submit **7403259**, cancel **7403230** (user confirm).
- Ep1–2 action-only: acc **4.9% → 2.5%** — pattern freeze-only fail tiếp tục.

---

## J. Giải thích metric — CE loss & accuracy

### Cross-Entropy (action-only train loss ~3.0)

\[
L = -\log p(y_{\text{true}}), \quad \text{mean over batch}
\]

Với class weights: mỗi sample nhân weight theo label (class hiếm weight cao).

| Mức | Loss (xấp xỉ) | Ý nghĩa |
|-----|---------------|---------|
| Random 28 class (uniform) | **3.33** | ln(28) |
| Job 7403259 ep1 train | **3.02** | ≈ random |
| Model học tốt (acc 50%+) | 1.5–2.5 | CE giảm rõ |
| Perfect | → 0 | |

**Accuracy vs majority:** trên Proto1 val, luôn predict `"nothing"` → **18.4%**. Mọi run freeze-only **18–19%** hoặc thấp hơn (weighted/action-only **2–6%**) đều **không phải** action recognition usable.

**Metric nên báo:** macro-F1, per-class recall, confusion matrix — không chỉ accuracy với class imbalance.

---

## K. Full commit list (83)

```
25b870a|2026-05-28|Add WiFiPose loader support
012d808|2026-05-28|add data loaders for each dataset and adapters
2c664d9|2026-05-29|add preprocessing pipeline
d64723c|2026-05-30|feat: add training portal, Helios integration, VPS deploy scripts
ff8a133|2026-05-30|feat: Eagle HPC integration + Quick Submit portal
... (see report-full-appendix-jobs-commits.md for full table)
3562b01|2026-06-13|feat(ml): add WiMoSE/WiPose/ViT2D models and humanlike experiments
e63fb9e|2026-06-13|Merge pull request #3
```

---

## L. Key commit 3562b01 (2026-06-13)

Bulk ML merge: WiMoseNet, WiPoseNet, ViT2D, MAE, diffusion, humanlike configs, Proto1, action head, `eval_wimose_action`, `gold_batch_prep`, `ddp_helpers`, sbatch scripts.

**Deleted:** legacy rootrel/metafi eval paths, `docs/experiment_log.md`.

---

## M. Related docs

| Doc | Nội dung |
|-----|----------|
| [report-team-full](../report-team-full-2026-05-14_to_2026-06-13.md) | Commit theo người, PR, timeline hệ thống |
| [report-eagle-jobs-raw.txt](report-eagle-jobs-raw.txt) | 201 job sacct raw |
| [failure-taxonomy.md](../../.cursor/skills/rf-worldpose-research-log/references/failure-taxonomy.md) | F1–F10 taxonomy |
| [proto1-baseline.md](../../.cursor/skills/rf-worldpose-research-log/references/proto1-baseline.md) | Hợp đồng thí nghiệm |
