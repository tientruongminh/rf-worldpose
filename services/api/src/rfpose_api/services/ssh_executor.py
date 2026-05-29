"""Unified SSH / Slurm executor — consolidates all HPC remote operations."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def _ssh_opts(ssh_key: str = "") -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if ssh_key:
        opts += ["-i", ssh_key]
    return opts


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
    script_name = f"{spec.job_id}.sbatch"
    rendered = render_sbatch(spec)
    if dry_run:
        return rendered
    opts = _ssh_opts(ssh_key)
    local_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / script_name
    local_tmp.write_text(rendered)
    subprocess.run(
        ["ssh", *opts, login, "mkdir", "-p", remote_dir, f"{remote_dir}/logs"],
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["scp", *opts, str(local_tmp), f"{login}:{remote_dir}/{script_name}"],
        check=True,
        timeout=60,
    )
    cmd = f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}"
    return subprocess.check_output(
        ["ssh", *opts, login, cmd], text=True, timeout=30
    ).strip()


def submit_script(
    *, login: str, ssh_key: str = "", remote_dir: str, script_name: str
) -> str:
    opts = _ssh_opts(ssh_key)
    cmd = f"cd {shlex.quote(remote_dir)} && sbatch --parsable {shlex.quote(script_name)}"
    return subprocess.check_output(
        ["ssh", *opts, login, cmd], text=True, timeout=30
    ).strip()


def test_connection(login: str, ssh_key: str = "") -> dict:
    opts = _ssh_opts(ssh_key)
    try:
        host = subprocess.check_output(
            ["ssh", *opts, login, "hostname"], text=True, timeout=15
        ).strip()
        queue = subprocess.check_output(
            [
                "ssh",
                *opts,
                login,
                "squeue -u $USER --format='%.8i %.9P %.20j %.2t %.10M' --noheader 2>/dev/null | head -5",
            ],
            text=True,
            timeout=15,
        ).strip()
        return {"ok": True, "hostname": host, "queue_preview": queue}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SSH connection timed out (15s)"}
    except subprocess.CalledProcessError as exc:
        return {
            "ok": False,
            "error": f"SSH failed (exit {exc.returncode}): {exc.stderr or exc.stdout or ''}",
        }
    except FileNotFoundError:
        return {"ok": False, "error": "ssh binary not found on this system"}


def list_remote_scripts(
    login: str, ssh_key: str = "", remote_dir: str = ""
) -> list[str]:
    if not remote_dir:
        return []
    opts = _ssh_opts(ssh_key)
    try:
        out = subprocess.check_output(
            [
                "ssh",
                *opts,
                login,
                f"ls {shlex.quote(remote_dir)}/*.sh {shlex.quote(remote_dir)}/*.sbatch 2>/dev/null",
            ],
            text=True,
            timeout=15,
        ).strip()
        return [Path(f).name for f in out.splitlines() if f.strip()]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []


# ── Slurm status queries ──────────────────────────────────


def slurm_status(login: str, slurm_job_id: str, *, ssh_key: str = "") -> str:
    cmd = f"sacct -j {slurm_job_id} --format=JobID,State,Elapsed,ExitCode --parsable2 --noheader | head -1"
    return subprocess.check_output(
        ["ssh", *_ssh_opts(ssh_key), login, cmd], text=True, timeout=15
    ).strip()


def slurm_job_detail(login: str, slurm_job_id: str, *, ssh_key: str = "") -> dict:
    fmt = "JobID,State,Elapsed,ExitCode,Start,End,MaxRSS,MaxVMSize,TotalCPU,NodeList"
    cmd = f"sacct -j {slurm_job_id} --format={fmt} --parsable2 --noheader | head -1"
    try:
        raw = subprocess.check_output(
            ["ssh", *_ssh_opts(ssh_key), login, cmd], text=True, timeout=15
        ).strip()
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
    opts = _ssh_opts(ssh_key)
    result: dict[str, str] = {"stdout": "", "stderr": ""}
    for suffix, key in [("out", "stdout"), ("err", "stderr")]:
        cmd = (
            f"tail -n {tail} {shlex.quote(remote_dir)}/logs/*{slurm_job_id}.{suffix} "
            f"2>/dev/null || echo '(no log file found)'"
        )
        try:
            out = subprocess.check_output(
                ["ssh", *opts, login, cmd], text=True, timeout=15
            ).strip()
            result[key] = out
        except Exception as exc:
            result[key] = f"(fetch failed: {exc})"
    return result


# ── Job cancellation ───────────────────────────────────────


def cancel_job(login: str, slurm_job_id: str, *, ssh_key: str = "") -> None:
    subprocess.run(
        ["ssh", *_ssh_opts(ssh_key), login, f"scancel {slurm_job_id}"],
        check=True,
        timeout=15,
    )
