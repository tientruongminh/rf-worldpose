from __future__ import annotations
import subprocess


def _ssh_opts(ssh_key: str = "") -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10"]
    if ssh_key:
        opts += ["-i", ssh_key]
    return opts


def slurm_status(login: str, slurm_job_id: str, *, ssh_key: str = "") -> str:
    cmd = f"sacct -j {slurm_job_id} --format=JobID,State,Elapsed,ExitCode --parsable2 --noheader | head -1"
    return subprocess.check_output(
        ["ssh", *_ssh_opts(ssh_key), login, cmd], text=True, timeout=15,
    ).strip()
