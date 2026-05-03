# Security Guide

## Principles
```text
No plaintext secrets in git
Device identity is mandatory
Gateway-cloud traffic uses mTLS
Firmware OTA is signed
Data at rest is encrypted
Video retention is short
All promotions/rollbacks are auditable
```

## Device security
Each ESP32-S3 node should have node_id, device keypair, firmware version, signed OTA validation, rollback partition, and audit heartbeat.

## mTLS
Use a private CA with API server certs and per-gateway client certs. Gateway verifies server identity; API verifies gateway identity.

## Secrets
Use SOPS + age for encrypted repo-managed env and Vault for managed production secrets.

## Video/privacy
Camera is train-time teacher only. Raw video retention should be short; derived pose labels can have longer retention by policy.

## Model promotion security
A model can be promoted only if eval gates passed, artifact hash recorded, model card generated, approver identity logged, and rollback target exists.
