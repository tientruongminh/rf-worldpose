"""Unified SSH / Slurm executor — uses paramiko instead of ssh binary."""
from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import paramiko

log = logging.getLogger(__name__)

TEMPLATE = (
    Path(__file__).resolve().parents[4]
    / "helios_runner"
    / "templates"
    / "train_gh200.sbatch"
)


@dataclass(frozen=True)
class HpcJobSpec:
    job_id: str
    dataset_version: str
    train_config: str
    account: str = ""
    partition: str = ""
    s3_bucket: str = "rfpose"
    s3_endpoint_url: str = "http://minio:9000"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    time_limit: str = "24:00:00"
    script_path: str = ""
    git_repo: str = ""
    git_branch: str = "main"


def _connect(login: str, ssh_key: str = "", timeout: int = 15) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    parts = login.split("@", 1)
    if len(parts) == 2:
        user, host = parts
    else:
        user, host = "root", parts[0]

    kwargs: dict = {"hostname": host, "username": user, "timeout": timeout}
    if ssh_key and os.path.isfile(ssh_key):
        kwargs["key_filename"] = ssh_key
    else:
        password = os.environ.get("HPC_PASSWORD", "")
        if password:
            kwargs["password"] = password

    ssh.connect(**kwargs)
    return ssh


def _exec(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0 and err:
        log.warning("cmd=%s exit=%d stderr=%s", cmd[:80], exit_code, err[:200])
    return out


def render_sbatch(spec: HpcJobSpec) -> str:
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
        "{{ script_path }}": spec.script_path or "training/train.py",
        "{{ git_repo }}": spec.git_repo or "https://github.com/tientruongminh/rf-worldpose",
        "{{ git_branch }}": spec.git_branch or "main",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def submit_training_job(
    spec: HpcJobSpec,
    *,
    login: str,
    ssh_key: str = "",
    remote_dir: str = "~/rfpose-jobs",
    dry_run: bool = False,
) -> str:
    rendered = render_sbatch(spec)
    if dry_run:
        return rendered

    script_name = f"{spec.job_id}.sbatch"
    ssh = _connect(login, ssh_key)
    try:
        _exec(ssh, f"mkdir -p {remote_dir} {remote_dir}/logs")
        sftp = ssh.open_sftp()
        remote_path = f"{remote_dir}/{script_name}"
        with sftp.file(remote_path, "w") as f:
            f.write(rendered)
        sftp.close()
        return _exec(ssh, f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}")
    finally:
        ssh.close()


def submit_script(
    *, login: str, ssh_key: str = "", remote_dir: str, script_name: str
) -> str:
    ssh = _connect(login, ssh_key)
    try:
        return _exec(ssh, f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}")
    finally:
        ssh.close()


def test_connection(login: str, ssh_key: str = "") -> dict:
    if not login:
        return {"ok": False, "error": "HPC_LOGIN not configured"}
    try:
        ssh = _connect(login, ssh_key)
        host = _exec(ssh, "hostname")
        queue = _exec(ssh, "squeue -u $USER --format='%.8i %.9P %.20j %.2t %.10M' --noheader 2>/dev/null | head -5")
        ssh.close()
        return {"ok": True, "hostname": host, "queue_preview": queue}
    except paramiko.AuthenticationException:
        return {"ok": False, "error": "SSH authentication failed — check credentials"}
    except paramiko.SSHException as exc:
        return {"ok": False, "error": f"SSH error: {exc}"}
    except TimeoutError:
        return {"ok": False, "error": "SSH connection timed out (15s)"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def list_remote_scripts(
    login: str, ssh_key: str = "", remote_dir: str = ""
) -> list[str]:
    if not remote_dir or not login:
        return []
    try:
        ssh = _connect(login, ssh_key)
        out = _exec(ssh, f"ls {shlex.quote(remote_dir)}/*.sh {shlex.quote(remote_dir)}/*.sbatch 2>/dev/null")
        ssh.close()
        return [Path(f).name for f in out.splitlines() if f.strip()]
    except Exception:
        return []


def slurm_status(login: str, slurm_job_id: str, *, ssh_key: str = "") -> str:
    ssh = _connect(login, ssh_key)
    try:
        cmd = f"sacct -j {slurm_job_id} --format=JobID,State,Elapsed,ExitCode --parsable2 --noheader | head -1"
        return _exec(ssh, cmd)
    finally:
        ssh.close()


def slurm_job_detail(login: str, slurm_job_id: str, *, ssh_key: str = "") -> dict:
    fmt = "JobID,State,Elapsed,ExitCode,Start,End,MaxRSS,MaxVMSize,TotalCPU,NodeList"
    cmd = f"sacct -j {slurm_job_id} --format={fmt} --parsable2 --noheader | head -1"
    try:
        ssh = _connect(login, ssh_key)
        raw = _exec(ssh, cmd)
        ssh.close()
        parts = raw.split("|")
        keys = fmt.split(",")
        return dict(zip(keys, parts)) if len(parts) >= len(keys) else {"raw": raw}
    except Exception as exc:
        return {"error": str(exc)}


def fetch_slurm_logs(
    login: str,
    slurm_job_id: str,
    *,
    ssh_key: str = "",
    remote_dir: str = "~/rfpose-jobs",
    tail: int = 100,
) -> dict:
    result: dict[str, str] = {"stdout": "", "stderr": ""}
    try:
        ssh = _connect(login, ssh_key)
        for suffix, key in [("out", "stdout"), ("err", "stderr")]:
            cmd = (
                f"tail -n {tail} {shlex.quote(remote_dir)}/logs/*{slurm_job_id}.{suffix} "
                f"2>/dev/null || echo '(no log file found)'"
            )
            result[key] = _exec(ssh, cmd)
        ssh.close()
    except Exception as exc:
        result["stdout"] = f"(fetch failed: {exc})"
    return result


def cancel_job(login: str, slurm_job_id: str, *, ssh_key: str = "") -> None:
    ssh = _connect(login, ssh_key)
    try:
        _exec(ssh, f"scancel {slurm_job_id}")
    finally:
        ssh.close()
