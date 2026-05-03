# Gateway Guide

## Responsibilities
UDP receive, packet decode, CRC validation, sequence/drop tracking, SQLite local buffer, NATS publish, Bronze upload, metrics, and edge inference hook.

## Environment
```text
RFPOSE_DEPLOYMENT_ID=room01
RFPOSE_GATEWAY_BIND=0.0.0.0:5006
RFPOSE_GATEWAY_SQLITE=/var/lib/rfpose/gateway.sqlite
NATS_URL=nats://localhost:4222
S3_BUCKET=rfpose
S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
RFPOSE_ONNX_MODEL=/models/model.onnx
```

## Failure behavior
If NATS fails, local SQLite still buffers. If S3 upload fails, packets remain unuploaded. If CRC fails, packet is dropped and bad counter increments.

## Bronze upload layout
`bronze/deployment={deployment_id}/date=YYYY-MM-DD/csi_raw/batch-{timestamp}-{uuid}.json`
