# Signed OTA

Production firmware must enable ESP-IDF secure boot/flash encryption where appropriate and sign OTA images in CI. Rollout order: canary node → observe heartbeat → remaining nodes. Rollback partition must stay enabled.
