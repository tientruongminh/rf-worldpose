from __future__ import annotations
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "train_gh200.sbatch"

@dataclass(frozen=True)
class HeliosJobSpec:
    job_id: str
    dataset_version: str
    train_config: str
    account: str
    partition: str = "plgrid-gpu-gh200"
    s3_bucket: str = "rfpose"
    s3_endpoint_url: str = "http://minio:9000"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    time_limit: str = "24:00:00"


def render_sbatch(spec: HeliosJobSpec) -> str:
    text = TEMPLATE.read_text()
    replacements = {
        "{{ partition }}": spec.partition,
        "{{ account }}": spec.account,
        "{{ time_limit | default(\"24:00:00\") }}": spec.time_limit,
        "{{ dataset_version }}": spec.dataset_version,
        "{{ train_config }}": spec.train_config,
        "{{ s3_bucket }}": spec.s3_bucket,
        "{{ s3_endpoint_url }}": spec.s3_endpoint_url,
        "{{ mlflow_tracking_uri }}": spec.mlflow_tracking_uri,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def submit_training_job(spec: HeliosJobSpec, *, login: str, remote_dir: str = "~/rfpose-jobs", dry_run: bool = False) -> str:
    script_name = f"{spec.job_id}.sbatch"
    rendered = render_sbatch(spec)
    if dry_run:
        return rendered

    local_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / script_name
    local_tmp.write_text(rendered)
    subprocess.run(["ssh", login, "mkdir", "-p", remote_dir, f"{remote_dir}/logs"], check=True)
    subprocess.run(["scp", str(local_tmp), f"{login}:{remote_dir}/{script_name}"], check=True)
    cmd = f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}"
    out = subprocess.check_output(["ssh", login, cmd], text=True).strip()
    return out
