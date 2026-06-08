ALTER TABLE training_jobs
  ADD COLUMN IF NOT EXISTS submitted_by TEXT,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS slurm_state TEXT;

CREATE INDEX IF NOT EXISTS idx_training_jobs_submitted_by ON training_jobs(submitted_by);
