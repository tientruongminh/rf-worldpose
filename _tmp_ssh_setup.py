import paramiko, os

pubkey = open(os.path.expanduser(r"~\.ssh\id_rsa_vps.pub")).read().strip()
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("207.180.243.242", username="root", password="teamKDL123456", timeout=15)

cmds = [
    "mkdir -p ~/.ssh",
    f'grep -qF "{pubkey[:40]}" ~/.ssh/authorized_keys 2>/dev/null || echo "{pubkey}" >> ~/.ssh/authorized_keys',
    "chmod 700 ~/.ssh",
    "chmod 600 ~/.ssh/authorized_keys",
    "echo KEY_ADDED_OK",
]
for c in cmds:
    stdin, stdout, stderr = ssh.exec_command(c)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print("ERR:", err)
ssh.close()
print("Done - SSH key added to VPS")
