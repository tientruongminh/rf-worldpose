from __future__ import annotations
import os
import shlex
import subprocess
from dataclasses import dataclass, field
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
        '{{ time_limit | default("24:00:00") }}': spec.time_limit,
        "{{ dataset_version }}": spec.dataset_version,
        "{{ train_config }}": spec.train_config,
        "{{ s3_bucket }}": spec.s3_bucket,
        "{{ s3_endpoint_url }}": spec.s3_endpoint_url,
        "{{ mlflow_tracking_uri }}": spec.mlflow_tracking_uri,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _ssh_opts(ssh_key: str = "") -> list[str]:
    """Build common SSH option flags."""
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if ssh_key:
        opts += ["-i", ssh_key]
    return opts


def submit_training_job(
    spec: HeliosJobSpec,
    *,
    login: str,
    ssh_key: str = "",
    remote_dir: str = "~/rfpose-jobs",
    dry_run: bool = False,
) -> str:
    script_name = f"{spec.job_id}.sbatch"
    rendered = render_sbatch(spec)
    if dry_run:
        return rendered

    opts = _ssh_opts(ssh_key)

    local_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / script_name
    local_tmp.write_text(rendered)

    subprocess.run(
        ["ssh", *opts, login, "mkdir", "-p", remote_dir, f"{remote_dir}/logs"],
        check=True, timeout=30,
    )
    subprocess.run(
        ["scp", *opts, str(local_tmp), f"{login}:{remote_dir}/{script_name}"],
        check=True, timeout=60,
    )
    cmd = f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}"
    out = subprocess.check_output(
        ["ssh", *opts, login, cmd], text=True, timeout=30,
    ).strip()
    return out


def test_connection(login: str, ssh_key: str = "") -> dict:
    """Quick SSH connectivity test — returns hostname + queue info."""
    opts = _ssh_opts(ssh_key)
    try:
        host = subprocess.check_output(
            ["ssh", *opts, login, "hostname"], text=True, timeout=15,
        ).strip()
        queue = subprocess.check_output(
            ["ssh", *opts, login, "squeue -u $USER --format='%.8i %.9P %.20j %.2t %.10M' --noheader | head -5"],
            text=True, timeout=15,
        ).strip()
        return {"ok": True, "hostname": host, "queue_preview": queue}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SSH connection timed out (15s)"}
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "error": f"SSH failed (exit {exc.returncode}): {exc.stderr or exc.stdout or ''}"}
    except FileNotFoundError:
        return {"ok": False, "error": "ssh binary not found on this system"}
