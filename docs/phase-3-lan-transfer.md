# Phase 3 trusted-LAN transfer

Phase 3 transfers one completed iPhone scan session to a Windows receiver over
HTTP. The protocol is version `1` and transfers files individually; it never
builds the whole session into one in-memory upload.

## Start the Windows receiver

```powershell
python windows/scanner_server/server.py `
  --host 0.0.0.0 `
  --port 8765 `
  --storage-root samples/received
```

The host, port, and storage root are command-line configuration. The receiver
uses a staging directory named `.session_<id>.staging` until final verification
passes. Completed sessions are moved atomically to `session_<id>`.

## Protocol

1. iPhone sends `POST /v1/sessions/begin` with the completed session ID and a
   sorted manifest containing every relative path, byte count, and SHA-256.
2. Receiver returns the missing paths.
3. iPhone sends each missing file with `PUT /v1/sessions/<id>/files/<path>`.
   The receiver writes a temporary file, checks `Content-Length` and SHA-256,
   then atomically replaces the staged destination.
4. iPhone sends `POST /v1/sessions/<id>/finalize` with the manifest.
5. Receiver verifies every file, completed session metadata, frame count, and
   required four-file frame layout, then returns a VERIFIED ACK containing
   only `protocol_version`, `status`, and `session_id`.
6. The iPhone checks those three fields and only then deletes its local raw
   session. The receiver may retain detailed verification metadata internally.

Any timeout, HTTP error, checksum mismatch, incomplete manifest, or app restart
leaves the iPhone session untouched. Repeating `begin` resumes missing files;
an already verified matching session returns an idempotent verified response.

## iPhone configuration

Enter the receiver URL in the completed-scan screen, for example:

```text
http://192.168.1.50:8765
```

The value is stored in device-local `UserDefaults`; no machine-specific URL is
committed. The app requests local-network permission and enables only the
`NSAllowsLocalNetworking` ATS exception needed for trusted-LAN HTTP.
