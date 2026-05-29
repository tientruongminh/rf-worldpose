#!/usr/bin/env python3
import sys, io, paramiko
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("207.180.243.242", username="root", password="teamKDL123456", timeout=15)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    return out, err, code

# Check if process running
out, _, _ = run("pgrep -fa uvicorn")
print(f"Process: {out or 'NOT RUNNING'}\n")

# Check logs
out, _, _ = run("tail -20 /var/log/rfpose-api.log 2>/dev/null")
print(f"Logs:\n{out}\n")

# Health check
out, _, code = run("curl -s http://localhost:8080/health")
print(f"Health: {out} (exit {code})")

# Portal check
out, _, code = run("curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/portal")
print(f"Portal HTTP status: {out}")

ssh.close()
