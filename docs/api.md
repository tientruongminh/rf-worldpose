# API Guide

## Health
`GET /health`

## Deployments
```http
POST /api/v1/deployments
GET /api/v1/deployments/{deployment_id}/status
PUT /api/v1/deployments/{deployment_id}/nodes/{node_id}
```

## Recording sessions
```http
POST /api/v1/recording-sessions
POST /api/v1/recording-sessions/{session_id}/finish
```

## Datasets
```http
POST /api/v1/datasets
GET /api/v1/datasets/{dataset_version}
```

## Training jobs
```http
POST /api/v1/training-jobs
GET /api/v1/training-jobs/{job_id}
POST /api/v1/training-jobs/{job_id}/mark-submitted
POST /api/v1/helios/training-jobs/{job_id}/submit?dry_run=true
```

## Models
```http
POST /api/v1/models
POST /api/v1/models/{model_id}/promote
POST /api/v1/models/{model_id}/rollback
```
