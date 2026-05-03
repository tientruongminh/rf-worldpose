# Database Schema Draft

## deployments

- id text primary key
- name text
- room_id text
- status text
- created_at timestamptz

## nodes

- id text primary key
- deployment_id text
- hardware_revision text
- firmware_version text
- position jsonb
- status text
- last_seen_at timestamptz

## recording_sessions

- id text primary key
- deployment_id text
- label text
- started_at timestamptz
- ended_at timestamptz
- bronze_uri text
- quality_status text
- metadata jsonb

## dataset_versions

- id text primary key
- source_sessions jsonb
- preprocess_version text
- teacher_version text
- artifact_uri text
- stats jsonb
- quality_report_uri text
- created_at timestamptz

## training_jobs

- id text primary key
- dataset_version text
- train_config text
- backend text default 'helios-slurm'
- slurm_job_id text
- slurm_partition text default 'plgrid-gpu-gh200'
- status text
- artifact_uri text
- eval_report_uri text
- logs_uri text
- created_at timestamptz
- finished_at timestamptz

## model_versions

- id text primary key
- name text
- status text
- dataset_version text
- training_job_id text
- artifact_uri text
- metrics jsonb
- eval_report_uri text
- hash text
- created_at timestamptz
- promoted_at timestamptz
