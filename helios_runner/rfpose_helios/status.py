from __future__ import annotations
import subprocess

def slurm_status(login: str, slurm_job_id: str) -> str:
    cmd = f"sacct -j {slurm_job_id} --format=JobID,State,Elapsed,ExitCode --parsable2 --noheader | head -1"
    return subprocess.check_output(["ssh", login, cmd], text=True).strip()
