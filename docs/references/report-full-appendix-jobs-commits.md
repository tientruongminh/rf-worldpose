# Phụ lục — Commit đầy đủ & Job Eagle

Báo cáo chính: [report-team-full-2026-05-14_to_2026-06-13.md](../report-team-full-2026-05-14_to_2026-06-13.md)

---

## A. Toàn bộ 83 commit (thứ tự thời gian)

| # | Hash | Ngày | Author | Message |
|---|------|------|--------|---------|
| 1 | 25b870a | 2026-05-28 | Ngan | Add WiFiPose loader support |
| 2 | 012d808 | 2026-05-28 | Ngan | add data loaders for each dataset and adapters |
| 3 | 2c664d9 | 2026-05-29 | Ngan | add preprocessing pipeline |
| 4 | d64723c | 2026-05-30 | tientruongminh | feat: add training portal, Helios integration, VPS deploy scripts |
| 5 | ff8a133 | 2026-05-30 | tientruongminh | feat: Eagle HPC integration + Quick Submit portal |
| 6 | d9453a3 | 2026-05-30 | tientruongminh | fix: TemplateResponse API for newer Starlette |
| 7 | b3a9a74 | 2026-05-30 | tientruongminh | fix: update labels from Helios to Eagle |
| 8 | df34deb | 2026-05-30 | tientruongminh | feat(portal): add dropdown presets for training config and team members |
| 9 | 9824189 | 2026-05-30 | tientruongminh | feat(portal): add Config Registry with CRUD, dynamic presets in Submit Job |
| 10 | d82bdbb | 2026-05-30 | tientruongminh | feat(portal): add Model Registry, Inference API, enhanced job detail with logs |
| 11 | 4eae645 | 2026-05-30 | tientruongminh | fix(inference): lazy-import numpy/onnxruntime to avoid startup crash |
| 12 | ebf2396 | 2026-05-30 | tientruongminh | refactor: separate portal from API, clean architecture |
| 13 | c09d827 | 2026-05-30 | tientruongminh | ci/cd: add GitHub Actions CI+CD, fix NATS healthcheck with alpine image |
| 14 | 219cceb | 2026-05-30 | tientruongminh | chore: remove temp scripts |
| 15 | 22f0ec6 | 2026-05-30 | tientruongminh | fix(dagster): remove OpExecutionContext annotation for newer dagster |
| 16 | a85218e | 2026-05-30 | tientruongminh | ci: test CD workflow trigger |
| 17 | 15e0cbc | 2026-05-30 | tientruongminh | ci: re-test CD with repository secrets |
| 18 | 9fdf7b0 | 2026-05-30 | tientruongminh | fix(cd): add command_timeout for longer deploys |
| 19 | f190c13 | 2026-05-30 | tientruongminh | fix(cd): use sshpass instead of appleboy action |
| 20 | 43221dd | 2026-05-30 | tientruongminh | fix(ci): simplify CI to compile+compose check only |
| 21 | e825b73 | 2026-05-30 | tientruongminh | fix(cd): remove if condition, simplify deploy |
| 22 | 46c392c | 2026-05-30 | tientruongminh | fix(cd): inline deploy command, fix YAML heredoc issue |
| 23 | d30bf26 | 2026-05-30 | tientruongminh | chore: add deploy.ps1 script |
| 24 | 213c3d0 | 2026-05-30 | tientruongminh | feat: add GitHub webhook auto-deploy listener |
| 25 | ce613e0 | 2026-05-30 | tientruongminh | test: verify webhook auto-deploy |
| 26 | ab75e75 | 2026-05-30 | tientruongminh | fix: use paramiko instead of ssh binary for HPC connections |
| 27 | 83524ce | 2026-05-30 | tientruongminh | fix(mlflow): allow all hosts to fix DNS rebinding error |
| 28 | 5aad870 | 2026-05-30 | tientruongminh | feat(grafana): provision datasources and overview dashboard |
| 29 | bd4bf6a | 2026-05-30 | Ngan | add build_gold.py |
| 30 | c7cec37 | 2026-06-02 | Ngoc Kim | Implement CSI tokenizer |
| 31 | e1fc45b | 2026-06-02 | Ngoc Kim | Cập nhật logic cho csi_tokenizer và transformer |
| 32 | dc500ec | 2026-06-02 | Ngoc Kim | Xóa file rác tạo nhầm |
| 33 | ff5665a | 2026-06-03 | Ngoc Kim | Remove Zone.Identifier junk files |
| 34 | d7ffad0 | 2026-06-03 | Ngan | add etl |
| 35 | e13cd27 | 2026-06-03 | Ngan | feat: add idempotent checks and progress logging to ETL pipeline |
| 36 | c4688cb | 2026-06-03 | Ngan | feat: add S3 idempotent check for bronze_to_silver and silver_to_gold |
| 37 | e0d8cf4 | 2026-06-03 | Ngan | add log and change path |
| 38 | 2bd7e0b | 2026-06-03 | Ngan | add postgre |
| 39 | 6d17cbf | 2026-06-03 | Ngan | fix: resolve merge conflict in data_lake.py dataset_registry_entry |
| 40 | fed2fc3 | 2026-06-03 | Ngan | fix: resolve all merge conflicts in bronze_to_silver and silver_to_gold |
| 41 | d509653 | 2026-06-03 | Ngan | fix: rename LOGGER to log for consistency |
| 42 | 54374e3 | 2026-06-03 | Ngan | feat: add percentage progress to ETL logging |
| 43 | a16a9cf | 2026-06-03 | Ngan | feat: add parallel processing for ETL pipeline |
| 44 | 77a97ca | 2026-06-03 | Ngan | feat: add granular per-step logging throughout ETL pipeline |
| 45 | 023168d | 2026-06-04 | Ngan | fix: enable logging.basicConfig at module level |
| 46 | adb3859 | 2026-06-04 | Ngan | fix: reduce parallel workers from 4 to 2 to avoid OOM |
| 47 | c9533c1 | 2026-06-04 | Ngan | fix: stream rows to disk instead of holding all in memory |
| 48 | dd3622a | 2026-06-04 | Ngan | fix: use streaming parquet conversion |
| 49 | 9453bf2 | 2026-06-04 | Ngan | refactor: redesign Silver layer |
| 50 | 7c36308 | 2026-06-05 | tientruongminh | feat: add Silver-Unified step |
| 51 | 9ed0435 | 2026-06-05 | tientruongminh | refactor: catalog-only Silver-Unified |
| 52 | e44663b | 2026-06-05 | tientruongminh | fix: use memmap for Gold X array |
| 53 | d453140 | 2026-06-05 | MaiThuHuong | feat: add eda folder |
| 54 | 0340d14 | 2026-06-08 | tientruongminh | feat: complete ML pipeline — training, evaluation, HPC submission, MLflow |
| 55 | 0fa571c | 2026-06-08 | tientruongminh | Merge remote-tracking branch origin/main into feature2 |
| 56 | c6be79e | 2026-06-08 | lang_du_coder | Merge pull request #2 |
| 57 | eb916a7 | 2026-06-08 | MaiThuHuong | Merge branch main into feature3 |
| 58 | a135775 | 2026-06-09 | tientruongminh | fix: correct encoder import names in GCN model |
| 59 | 71679d8 | 2026-06-09 | tientruongminh | feat(ml): complete training pipeline with multi-architecture experiments |
| 60 | 4350eb6 | 2026-06-09 | Ngoc Kim | upload model & training & loss & evaluation |
| 61 | 13feed0 | 2026-06-09 | Ngoc Kim | register config module for ngoc exp |
| 62 | 0295405 | 2026-06-09 | Ngoc Kim | register config fix typo |
| 63 | ac3c538 | 2026-06-09 | Ngoc Kim | hotfix: append eagle_runner path to sys.path in hpc router |
| 64 | 9b23469 | 2026-06-09 | MaiThuHuong | feat: model CSIViT2DPose and config run with dataset wipose, mmfi |
| 65 | defeda4 | 2026-06-09 | Mai Thu Huong | Merge pull request #4 |
| 66 | 35d55c2 | 2026-06-10 | buibaongan | add ssl_cnn model |
| 67 | e4d049b | 2026-06-10 | buibaongan | Merge pull request #6 |
| 68 | 1ed47e6 | 2026-06-10 | buibaongan | Add SSL CNN training with MLflow checkpoint artifacts |
| 69 | 9dfa192 | 2026-06-10 | buibaongan | Merge branch main into cnn |
| 70 | 2b2f223 | 2026-06-10 | buibaongan | Merge pull request #7 |
| 71 | b6e9e13 | 2026-06-10 | buibaongan | Fix API startup without helios router |
| 72 | ab7f34a | 2026-06-10 | buibaongan | Merge pull request #8 |
| 73 | 36cf7f4 | 2026-06-10 | tientruongminh | untracked files on feature/ml-training-experiments (stash) |
| 74 | e6aefac | 2026-06-10 | tientruongminh | index on feature/ml-training-experiments (stash) |
| 75 | 72c57dd | 2026-06-10 | tientruongminh | WIP on feature/ml-training-experiments (stash) |
| 76 | 83c8f53 | 2026-06-10 | buibaongan | change gold data path |
| 77 | be07f72 | 2026-06-10 | buibaongan | Merge pull request #9 |
| 78 | 7ed2cca | 2026-06-10 | MaiThuHuong | add model viT2d after augmenta |
| 79 | 6bd2672 | 2026-06-10 | Ngoc Kim | update config |
| 80 | 4efacf6 | 2026-06-10 | Ngoc Kim | add v3 model |
| 81 | 3562b01 | 2026-06-13 | tientruongminh | feat(ml): add WiMoSE/WiPose/ViT2D models and humanlike experiments |
| 82 | ff32f09 | 2026-06-13 | tientruongminh | Merge origin/main into feature/ml-training-experiments |
| 83 | e63fb9e | 2026-06-13 | lang_du_coder | Merge pull request #3 |

---

## B. Commit theo thành viên (copy nhanh)

### Tiến — 37 commits
d64723c … e63fb9e (xem bảng trên #4–28, #50–52, #54–59, #73–75, #81–83)

### Ngân — 20 commits (#1–3, #29, #34–49)

### Ngọc — 10 commits (#30–33, #60–63, #79–80)

### Hương — 4 commits (#53, #57, #64–65, #78)

### buibaongan (Ngân PR) — 9 commits (#66–72, #76–77)

---

## C. 201 job Eagle — danh sách đầy đủ

File raw (JobID, JobName, State, Elapsed, ExitCode):

→ [report-eagle-jobs-raw.txt](report-eagle-jobs-raw.txt)

### C.1 Phân loại theo tên job

| Nhóm | Số job ước tính | COMPLETED nổi bật | FAILED/CANCEL nổi bật |
|------|-----------------|-------------------|------------------------|
| Demo / test / quick | ~25 | demo_tien, py-check, simple-test | quick-test 01–06 |
| Setup / jupyter / nb | ~10 | rfpose-setup, run-nb | jupyter-gpu |
| SSL pretrain v2 | 4 | — | cancel dài 4h+ |
| Supervised / rootrel / GCN legacy | ~25 | rootrel-mmfi 7342638, gcn v3, supervised 7341939 | nhiều fail sớm |
| MAE (legacy + wimae) | ~15 | mae-4gpu-v2, mae-ft-* | wimae 7403053–65 fail |
| ViT2D (Hương) | ~15 | 7351682, 7351683 | patch_size, DDP proto1 |
| SSL CNN (Ngân) | ~8 | **7359067** 4h03m | proto1 DDP |
| Transformer (Ngọc) | ~10 | legacy ~320mm | proto1 DDP |
| WiMose / humanlike (Tiến) | ~35 | **7401153**, 7400269, 7403201 | OOM, SIGABRT, cancel |
| Team Proto1 2gpu | 6 | — | **7403213–15 all FAILED** |
| Action / eval / viz | ~10 | viz-3model, viz-17j | eval test chưa xong |
| ETL humanlike | 4 | 7398885 | early fail |

### C.2 Job có metric ML ghi nhận (COMPLETED hoặc ckpt OK)

| Job | Name | Metric |
|-----|------|--------|
| 7401153 | wimose-mmfi-proto1 | **157.7 mm val** (FAILED Slurm) |
| 7400250 | wimose-mmfi-clean-fast | 169.3 mm |
| 7400269 | wimose-wipose-stable | 297.7 mm |
| 7359067 | rfpose-ssl-cnn-ngan | 607.8 mm |
| ngoc logs | rfpose-ngoc-transformer | 319.6 mm |
| 7351683 | rfpose-vit2d-mmfi-v3 | 314.3 mm |
| 7342638 | rfpose-rootrel-mmfi-v1 | 305.9 mm |
| 7342666 | rootrel-mmfi-weighted-v1 | 3h34m COMPLETED |
| 7342905 | rfpose-mae-4gpu-v2 | 1h48m |
| 7350916–918 | rfpose-mae-ft-* | 11–44 min |
| 7359725 | wipose-paper | 4h54m |
| 7398916_1 | humanlike-train-v2 | 3h24m |

---

## D. Thay đổi repo lớn (commit 3562b01)

**Thêm:** WiMoseNet, WiPoseNet, ViT2D pose, MAE, diffusion, humanlike configs, Proto1, action eval, EDA, sbatch scripts, gold_batch_prep, ddp_helpers

**Xóa:** rootrel/metafi eval v2, legacy train_v2, csi_tokenizer_attn, transformer_rootrel, experiment_log.md

**Thay config:** từ `*_eagle.yaml` rootrel → `wimose_*`, `humanlike_*`, team `ssl_cnn`, `ngoc_transformer`, `vit2d_mmfi`

---

*Generated 2026-06-13. Re-export jobs: `ssh eagle sacct ...` → `report-eagle-jobs-raw.txt`*
