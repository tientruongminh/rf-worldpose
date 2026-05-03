"""Helios Slurm submitter skeleton.

Renders sbatch templates, uploads to login01.helios.cyfronet.pl, submits via sbatch --parsable,
and records slurm_job_id in the control-plane database.
"""
from __future__ import annotations

def submit_training_job(*, dataset_version: str, train_config: str) -> str:
    # TODO: render Jinja template, scp to Helios, ssh sbatch --parsable, return job id.
    raise NotImplementedError("Helios submission is not configured yet")
