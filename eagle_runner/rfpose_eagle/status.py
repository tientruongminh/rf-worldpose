from __future__ import annotations

import subprocess


def slurm_status(ssh_host: str, slurm_job_id: str) -> str:
    cmd = (
        f"sacct -j {slurm_job_id} "
        "--format=JobID,State,Elapsed,ExitCode --parsable2 --noheader | head -1"
    )
    return subprocess.check_output(["ssh", ssh_host, cmd], text=True).strip()


def cancel_job(ssh_host: str, slurm_job_id: str) -> None:
    subprocess.run(["ssh", ssh_host, f"scancel {slurm_job_id}"], check=True)
