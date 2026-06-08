"""GitHub Webhook listener — auto deploy on push to main.

Listens on port 9999. When GitHub sends a push event for the main branch,
pulls latest code and restarts docker-compose services.
"""
import hashlib
import hmac
import json
import logging
import os
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deploy-hook")

SECRET = os.environ.get("WEBHOOK_SECRET", "rfpose-deploy-2024")
REPO_DIR = os.environ.get("REPO_DIR", "/opt/rfpose/rf-worldpose")
COMPOSE_DIR = os.path.join(REPO_DIR, "infra/docker-compose")
PORT = int(os.environ.get("WEBHOOK_PORT", "9999"))


def verify_signature(payload: bytes, signature: str) -> bool:
    if not signature:
        return False
    mac = hmac.new(SECRET.encode(), payload, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature)


def deploy():
    log.info("Starting deploy...")
    try:
        subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=True, timeout=30)
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=True, timeout=10)
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--force-recreate", "--remove-orphans"],
            cwd=COMPOSE_DIR, capture_output=True, text=True, timeout=300,
        )
        log.info("Deploy output:\n%s", result.stdout[-500:] if result.stdout else "(empty)")
        if result.returncode != 0:
            log.error("Deploy failed:\n%s", result.stderr[-500:])
        else:
            log.info("Deploy successful!")
    except Exception as exc:
        log.error("Deploy error: %s", exc)


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        signature = self.headers.get("X-Hub-Signature-256", "")

        if not verify_signature(payload, signature):
            log.warning("Invalid signature from %s", self.client_address[0])
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"invalid signature")
            return

        event = self.headers.get("X-GitHub-Event", "")
        if event != "push":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"ignored event: {event}".encode())
            return

        data = json.loads(payload)
        ref = data.get("ref", "")
        if ref != "refs/heads/main":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"ignored branch: {ref}".encode())
            return

        pusher = data.get("pusher", {}).get("name", "unknown")
        commit_msg = data.get("head_commit", {}).get("message", "")[:80]
        log.info("Push to main by %s: %s", pusher, commit_msg)

        deploy()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"deploy triggered")

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), WebhookHandler)
    log.info("Webhook listener running on port %d", PORT)
    server.serve_forever()
