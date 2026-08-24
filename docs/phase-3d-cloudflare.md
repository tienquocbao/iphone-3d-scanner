# Phase 3D HTTPS transfer through Cloudflare Tunnel

Phase 3D keeps the Phase 3 file-by-file protocol and places it behind a
Cloudflare Tunnel:

```text
iPhone HTTPS
  -> lidar3d.barospace.id.vn
  -> Cloudflare Tunnel
  -> http://127.0.0.1:8765
  -> Windows scanner receiver
```

The router does not expose port 8765. The receiver binds to loopback for the
remote deployment and requires `IPHONE3D_RECEIVER_TOKEN` as a bearer token.
The token is generated/stored locally on Windows and entered into the iPhone
SecureField, backed by iOS Keychain. It is never stored in UserDefaults or the
repository.

Use the receiver:

```powershell
$env:IPHONE3D_RECEIVER_TOKEN = "<local secret>"
python windows/scanner_server/server.py `
  --host 127.0.0.1 `
  --port 8765 `
  --storage-root samples/received
```

Health requests require the same bearer token:

```text
GET /api/v1/health
Authorization: Bearer <local secret>
```

Only a protocol-version-1 `status: verified` finalize response can cause the
iPhone to delete its local raw session. Authentication failures, timeouts,
checksum failures, and malformed responses preserve the local session.

The tunnel configuration is intentionally kept outside the repository because
it contains a machine-specific credentials-file path. Preserve unrelated
ingress rules when deploying the tunnel.

