# Helios GH200 Slurm Backend

- Login: `login01.helios.cyfronet.pl`
- Partition: `plgrid-gpu-gh200`
- Node: 4x NVIDIA GH200 96GB, Grace ARM aarch64, Rocky 9
- Time limit: 48h

Use `helios_runner/templates/train_gh200.sbatch` as the canonical job template. Do not treat `$SCRATCH` as source of truth; sync datasets/artifacts with MinIO/S3.
