#!/usr/bin/env python3
import sys, io, paramiko
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("207.180.243.242", username="root", password="teamKDL123456", timeout=15)

# Hit portal and check error
stdin, stdout, stderr = ssh.exec_command(
    "curl -s http://localhost:8080/portal 2>&1; echo '---'; tail -30 /var/log/rfpose-api.log",
    timeout=15
)
print(stdout.read().decode("utf-8", errors="replace"))
ssh.close()
