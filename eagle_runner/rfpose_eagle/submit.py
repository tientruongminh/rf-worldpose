from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rfpose_eagle.registry import get_preset, resolve_train_module

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "train_eagle.sbatch"

@dataclass(frozen=True)
class EagleJobSpec:
    job_id: str
    config_name: str = "wimose_mmfi17j_proto1_eagle"
    train_module: str = ""
    dataset_version: str = ""
    project_root: str = "pl0501-01/project_data/rf-worldpose"
    partition: str = ""
    gpu_type: str = ""
    gpus: int = 0
    cpus: int = 0
    mem: str = ""
    time_limit: str = ""
    mlflow_tracking_uri: str = "http://207.180.243.242:5000"
    epochs: int = 50
    batch_size: int = 32
    dry_run: bool = False


def _ssh_opts(ssh_key: str = "") -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    if ssh_key:
        opts += ["-i", ssh_key]
    return opts


def render_sbatch(spec: EagleJobSpec) -> str:
    text = TEMPLATE.read_text()
    preset = get_preset(spec.config_name)
    module = resolve_train_module(spec.config_name, spec.train_module)
    dataset_version = spec.dataset_version or preset.dataset_version
    partition = spec.partition or preset.partition
    gpu_type = spec.gpu_type or preset.gpu_type
    gpus = spec.gpus or preset.gpus
    cpus = spec.cpus or preset.cpus
    mem = spec.mem or preset.mem
    time_limit = spec.time_limit or preset.time_limit
    replacements = {
        "{{ partition }}": partition,
        "{{ gpu_type }}": gpu_type,
        "{{ gpus }}": str(gpus),
        "{{ cpus }}": str(cpus),
        "{{ mem }}": mem,
        "{{ time_limit }}": time_limit,
        "{{ project_root }}": spec.project_root,
        "{{ dataset_version }}": dataset_version,
        "{{ mlflow_tracking_uri }}": spec.mlflow_tracking_uri,
        "{{ job_id }}": spec.job_id,
        "{{ epochs }}": str(spec.epochs),
        "{{ batch_size }}": str(spec.batch_size),
        "{{ dry_run }}": "true" if spec.dry_run else "false",
        "{{ config_name }}": spec.config_name,
        "{{ train_module }}": module,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def sync_code(
    repo_root: str,
    ssh_host: str,
    eagle_root: str,
    *,
    ssh_key: str = "",
) -> None:
    """Rsync ml/ and eagle_runner/ to Eagle."""
    opts = _ssh_opts(ssh_key)
    subprocess.run(
        ["ssh", *opts, ssh_host, "mkdir", "-p",
         f"{eagle_root}/ml", f"{eagle_root}/data/gold",
         f"{eagle_root}/logs", f"{eagle_root}/checkpoints"],
        check=True, timeout=30,
    )
    for subdir in ["ml", "eagle_runner"]:
        src = f"{repo_root}/{subdir}/"
        dst = f"{ssh_host}:{eagle_root}/{subdir}/"
        cmd = ["rsync", "-az", "--delete",
               "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", "*.pyc"]
        if ssh_key:
            cmd += ["-e", f"ssh {' '.join(opts)}"]
        cmd += [src, dst]
        subprocess.run(cmd, check=True, timeout=120)


def submit_training_job(
    spec: EagleJobSpec,
    *,
    ssh_host: str = "eagle",
    ssh_key: str = "",
    remote_dir: str = "~/rfpose-jobs",
    repo_root: str = "",
    sync: bool = True,
    dry_run: bool = False,
) -> str:
    """Render sbatch, optionally sync code, submit via Slurm."""
    script_name = f"{spec.job_id}.sbatch"
    rendered = render_sbatch(spec)
    if dry_run:
        return rendered

    opts = _ssh_opts(ssh_key)

    if sync and repo_root:
        sync_code(repo_root, ssh_host, spec.project_root, ssh_key=ssh_key)

    local_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / script_name
    local_tmp.write_text(rendered)

    mkdir_cmd = f"mkdir -p {remote_dir} {remote_dir}/logs"
    subprocess.run(
        ["ssh", *opts, ssh_host, mkdir_cmd],
        check=True, timeout=30,
    )
    scp_cmd = ["scp"]
    if ssh_key:
        scp_cmd += ["-i", ssh_key]
    scp_cmd += ["-o", "StrictHostKeyChecking=accept-new",
                str(local_tmp), f"{ssh_host}:{remote_dir}/{script_name}"]
    subprocess.run(scp_cmd, check=True, timeout=60)

    if remote_dir.startswith("~/"):
        cd_path = "$HOME/" + shlex.quote(remote_dir[2:])
    else:
        cd_path = shlex.quote(remote_dir)
    script_q = shlex.quote(script_name)
    cmd = f"cd {cd_path} && sbatch --parsable {script_q}"
    out = subprocess.check_output(
        ["ssh", *opts, ssh_host, cmd], text=True, timeout=30,
    ).strip()
    return out
