CREATE TABLE IF NOT EXISTS deployments (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  room_id TEXT,
  status TEXT NOT NULL DEFAULT 'created',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nodes (
  id TEXT PRIMARY KEY,
  deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
  hardware_revision TEXT,
  firmware_version TEXT,
  position JSONB NOT NULL DEFAULT '{}'::jsonb,
  status TEXT NOT NULL DEFAULT 'unknown',
  last_seen_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recording_sessions (
  id TEXT PRIMARY KEY,
  deployment_id TEXT NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
  label TEXT,
  status TEXT NOT NULL DEFAULT 'created',
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  bronze_uri TEXT,
  quality_status TEXT NOT NULL DEFAULT 'unknown',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_versions (
  id TEXT PRIMARY KEY,
  source_sessions JSONB NOT NULL DEFAULT '[]'::jsonb,
  preprocess_version TEXT NOT NULL,
  teacher_version TEXT,
  artifact_uri TEXT NOT NULL,
  stats JSONB NOT NULL DEFAULT '{}'::jsonb,
  quality_report_uri TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by TEXT
);

CREATE TABLE IF NOT EXISTS training_jobs (
  id TEXT PRIMARY KEY,
  dataset_version TEXT NOT NULL REFERENCES dataset_versions(id),
  train_config TEXT NOT NULL,
  backend TEXT NOT NULL DEFAULT 'helios-slurm',
  slurm_job_id TEXT,
  slurm_partition TEXT NOT NULL DEFAULT 'plgrid-gpu-gh200',
  status TEXT NOT NULL DEFAULT 'created',
  artifact_uri TEXT,
  eval_report_uri TEXT,
  logs_uri TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  submitted_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'candidate',
  dataset_version TEXT REFERENCES dataset_versions(id),
  training_job_id TEXT REFERENCES training_jobs(id),
  artifact_uri TEXT NOT NULL,
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
  eval_report_uri TEXT,
  hash TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  promoted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_nodes_deployment ON nodes(deployment_id);
CREATE INDEX IF NOT EXISTS idx_sessions_deployment ON recording_sessions(deployment_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_models_status ON model_versions(status);
