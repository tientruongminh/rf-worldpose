# Testing Guide

```bash
make -C firmware/esp32-csi-node/test check
cargo test --manifest-path gateway/rf-gateway/Cargo.toml
python -m compileall services/api/src pipelines/dagster/rfpose_pipelines ml/rfpose
bash scripts/validate_etl.sh
python tools/mock_sender/send_mock_csi.py --node-id 1 --count 100
PYTHONPATH=helios_runner python helios_runner/test_dry_run.py
```
