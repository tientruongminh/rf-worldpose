#!/bin/bash
# Install deploy webhook as systemd service on VPS
# Usage: bash infra/webhook/install.sh

set -e

REPO_DIR="/opt/rfpose/rf-worldpose"
WEBHOOK_SECRET="${1:-rfpose-deploy-2024}"

cat > /etc/systemd/system/rfpose-webhook.service << EOF
[Unit]
Description=RF-WorldPose Deploy Webhook
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=/usr/bin/python3 $REPO_DIR/infra/webhook/deploy-hook.py
Environment=WEBHOOK_SECRET=$WEBHOOK_SECRET
Environment=REPO_DIR=$REPO_DIR
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rfpose-webhook
systemctl restart rfpose-webhook
systemctl status rfpose-webhook --no-pager

echo ""
echo "=== Webhook installed ==="
echo "Listening on port 9999"
echo "Secret: $WEBHOOK_SECRET"
echo "GitHub Webhook URL: http://$(hostname -I | awk '{print $1}'):9999"
