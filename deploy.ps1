# RF-WorldPose — Deploy to VPS
# Usage: .\deploy.ps1

Write-Host "=== Push to GitHub ===" -ForegroundColor Cyan
git push origin main

Write-Host "`n=== Deploying to VPS ===" -ForegroundColor Cyan
ssh vps "cd /opt/rfpose/rf-worldpose && git pull origin main && cd infra/docker-compose && docker compose up -d --force-recreate --remove-orphans 2>&1 && sleep 10 && docker compose ps"

Write-Host "`n=== Deploy complete ===" -ForegroundColor Green
