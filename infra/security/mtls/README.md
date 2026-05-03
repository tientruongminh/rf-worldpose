# mTLS

Generate a private CA, issue gateway client certs and API server certs. Gateway-cloud traffic must verify both server and client identities. Store cert material with SOPS/Vault, never plaintext in git.
