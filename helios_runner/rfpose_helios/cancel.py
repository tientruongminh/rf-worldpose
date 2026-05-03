from __future__ import annotations
import subprocess

def cancel_job(login: str, slurm_job_id: str) -> None:
    subprocess.run(["ssh", login, f"scancel {slurm_job_id}"], check=True)
