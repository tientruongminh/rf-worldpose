from __future__ import annotations
import subprocess


def _ssh_opts(ssh_key: str = "") -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if ssh_key:
        opts += ["-i", ssh_key]
    return opts


def cancel_job(login: str, slurm_job_id: str, *, ssh_key: str = "") -> None:
    subprocess.run(
        ["ssh", *_ssh_opts(ssh_key), login, f"scancel {slurm_job_id}"],
        check=True, timeout=15,
    )
