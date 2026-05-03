#!/usr/bin/env bash
set -euo pipefail

# Gateway metrics + inference scaffold
cat > gateway/rf-gateway/src/metrics.rs <<'RS'
use std::sync::atomic::{AtomicU64, Ordering};
#[derive(Default)]
pub struct GatewayMetrics { pub packets_ok: AtomicU64, pub packets_bad: AtomicU64, pub packets_uploaded: AtomicU64 }
impl GatewayMetrics { pub fn render_prometheus(&self) -> String { format!("rfpose_packets_ok {}\nrfpose_packets_bad {}\nrfpose_packets_uploaded {}\n", self.packets_ok.load(Ordering::Relaxed), self.packets_bad.load(Ordering::Relaxed), self.packets_uploaded.load(Ordering::Relaxed)) } }
RS
cat > gateway/rf-gateway/src/inference.rs <<'RS'
use anyhow::Result;
use crate::packet::CsiPacket;
#[derive(Clone, Debug)]
pub struct InferenceOutput { pub action: String, pub confidence: f32 }
#[derive(Clone)]
pub struct EdgeInference { model_path: Option<String> }
impl EdgeInference { pub fn new(model_path: Option<String>) -> Self { Self { model_path } } pub fn predict(&self, _window: &[CsiPacket]) -> Result<Option<InferenceOutput>> { if self.model_path.is_none(){return Ok(None)}; Ok(Some(InferenceOutput{action:"unknown".into(),confidence:0.0})) } }
RS
python3 - <<'PY'
from pathlib import Path
p=Path('gateway/rf-gateway/src/main.rs')
s=p.read_text()
s=s.replace('mod nats;\nmod packet;\nmod uploader;','mod nats;\nmod packet;\nmod uploader;\nmod metrics;\nmod inference;')
s=s.replace('use std::{collections::HashMap, sync::{Arc, Mutex}, time::Duration};','use std::{collections::HashMap, sync::{Arc, Mutex, atomic::Ordering}, time::Duration};')
s=s.replace('use uploader::{s3_client, BronzeUploader};','use uploader::{s3_client, BronzeUploader};\nuse metrics::GatewayMetrics;\nuse inference::EdgeInference;')
s=s.replace('let mut health: HashMap<u8, NodeHealth> = HashMap::new();','let mut health: HashMap<u8, NodeHealth> = HashMap::new();\n    let metrics = Arc::new(GatewayMetrics::default());\n    let _edge_infer = EdgeInference::new(std::env::var("RFPOSE_ONNX_MODEL").ok());')
s=s.replace('info!(uploaded, "uploaded Bronze CSI batch");','info!(uploaded, "uploaded Bronze CSI batch");')
s=s.replace('Ok(pkt) => {','Ok(pkt) => {\n                metrics.packets_ok.fetch_add(1, Ordering::Relaxed);')
s=s.replace('Err(e) => {\n                warn!', 'Err(e) => {\n                metrics.packets_bad.fetch_add(1, Ordering::Relaxed);\n                warn!')
p.write_text(s)
PY

# API Helios submit/eval promotion logic
cat > services/api/src/rfpose_api/routers/helios.py <<'PY'
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
PY
python3 - <<'PY'
from pathlib import Path
p=Path('services/api/src/rfpose_api/main.py')
s=p.read_text()
s=s.replace('from rfpose_api.routers import deployments, sessions, datasets, training, models','from rfpose_api.routers import deployments, sessions, datasets, training, models, helios')
s=s.replace('app.include_router(models.router)','app.include_router(models.router)\napp.include_router(helios.router)')
p.write_text(s)
PY

# Dashboard API fetch scaffold
cat > dashboard/src/app/lib.ts <<'TS'
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080'
export async function getJson(path: string) { const r = await fetch(`${API_BASE}${path}`, { cache: 'no-store' }); if (!r.ok) throw new Error(`API ${r.status}`); return r.json() }
TS
cat > dashboard/src/app/nodes/page.tsx <<'TS'
import { getJson } from '../lib'
export default async function Nodes(){ let data:any={nodes:[]}; try{data=await getJson('/api/v1/deployments/room01/status')}catch{} return <main style={{padding:32,fontFamily:'Inter, sans-serif'}}><h1>Nodes</h1><pre>{JSON.stringify(data.nodes||[],null,2)}</pre></main>}
TS
cat > dashboard/src/app/training/page.tsx <<'TS'
export default function Training(){return <main style={{padding:32,fontFamily:'Inter, sans-serif'}}><h1>Training Jobs</h1><p>Helios Slurm jobs, status, logs, artifacts and eval gates.</p></main>}
TS
cat > dashboard/src/app/models/page.tsx <<'TS'
export default function Models(){return <main style={{padding:32,fontFamily:'Inter, sans-serif'}}><h1>Model Registry</h1><p>Candidate → staging → production → rollback.</p></main>}
TS

# ETL label join utility
cat > pipelines/dagster/rfpose_pipelines/etl/join_labels.py <<'PY'
from __future__ import annotations
import json
from pathlib import Path

def load_events(path: str|Path):
    p=Path(path)
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def label_for_timestamp(events, ts_us:int, default='unlabeled'):
    ts=ts_us/1_000_000
    active=default
    for e in events:
        if e.get('t',0) <= ts and e.get('event')=='start': active=e.get('label',active)
        if e.get('t',0) <= ts and e.get('event')=='end' and e.get('label')==active: active=default
    return active
PY

# CI workflow
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'YAML'
name: ci
on: [push, pull_request]
jobs:
  rust-gateway:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --manifest-path gateway/rf-gateway/Cargo.toml
  python-compile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: python -m compileall services/api/src pipelines/dagster/rfpose_pipelines ml/rfpose
  firmware-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make -C firmware/esp32-csi-node/test check
YAML

python3 -m compileall services/api/src pipelines/dagster/rfpose_pipelines ml/rfpose >/tmp/rfpose_final_compile.log
cargo test --manifest-path gateway/rf-gateway/Cargo.toml --quiet
make -C firmware/esp32-csi-node/test check
PYTHONPATH=helios_runner python3 helios_runner/test_dry_run.py

git add .
git commit -m "feat: add final integration hooks for metrics inference helios dashboard labels and ci"
git log --oneline -12
