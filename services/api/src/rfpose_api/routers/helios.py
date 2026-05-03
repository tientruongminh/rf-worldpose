from fastapi import APIRouter, HTTPException
from rfpose_api.db.connection import connect
from rfpose_api.config import settings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[5] / 'helios_runner'))
from rfpose_helios.submit import HeliosJobSpec, submit_training_job

router = APIRouter(prefix='/api/v1/helios', tags=['helios'])

@router.post('/training-jobs/{job_id}/submit')
def submit_job(job_id: str, dry_run: bool = True):
    with connect() as conn, conn.cursor() as cur:
        cur.execute('SELECT * FROM training_jobs WHERE id=%s', (job_id,)); job=cur.fetchone()
        if not job: raise HTTPException(404,'training job not found')
        spec=HeliosJobSpec(job_id=job_id,dataset_version=job['dataset_version'],train_config=job['train_config'],account=settings.helios_account,partition=settings.helios_partition,s3_bucket=settings.s3_bucket,s3_endpoint_url=settings.s3_endpoint_url,mlflow_tracking_uri=settings.mlflow_tracking_uri)
        result=submit_training_job(spec, login=settings.helios_login, dry_run=dry_run)
        if dry_run: return {'dry_run': True, 'sbatch': result}
        cur.execute("UPDATE training_jobs SET status='submitted', slurm_job_id=%s, submitted_at=now() WHERE id=%s RETURNING *", (result, job_id))
        return cur.fetchone()
