from __future__ import annotations
import shlex
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


def slurm_job_detail(login: str, slurm_job_id: str, *, ssh_key: str = "") -> dict:
    """Fetch extended sacct info for a Slurm job."""
    fmt = "JobID,State,Elapsed,ExitCode,Start,End,MaxRSS,MaxVMSize,TotalCPU,NodeList"
    cmd = f"sacct -j {slurm_job_id} --format={fmt} --parsable2 --noheader | head -1"
    try:
        raw = subprocess.check_output(
            ["ssh", *_ssh_opts(ssh_key), login, cmd], text=True, timeout=15,
        ).strip()
        parts = raw.split("|")
        keys = fmt.split(",")
        return dict(zip(keys, parts)) if len(parts) >= len(keys) else {"raw": raw}
    except Exception as exc:
        return {"error": str(exc)}


def fetch_slurm_logs(
    login: str, slurm_job_id: str, *,
    ssh_key: str = "", remote_dir: str = "~/rfpose-jobs",
    tail: int = 100,
) -> dict:
    """Fetch stdout/stderr log tails from HPC."""
    opts = _ssh_opts(ssh_key)
    result = {"stdout": "", "stderr": ""}
    for suffix, key in [("out", "stdout"), ("err", "stderr")]:
        cmd = f"tail -n {tail} {shlex.quote(remote_dir)}/logs/*{slurm_job_id}.{suffix} 2>/dev/null || echo '(no log file found)'"
        try:
            out = subprocess.check_output(
                ["ssh", *opts, login, cmd], text=True, timeout=15,
            ).strip()
            result[key] = out
        except Exception as exc:
            result[key] = f"(fetch failed: {exc})"
    return result
