# Phase 3 V2: local capture, batch upload, and web jobs

The iPhone capture pipeline is local-first:

```text
ARKit frame -> atomic local frame directory -> STOP -> local session validation
                                               -> COMPLETED -> optional batch upload
```

STOP never contacts the receiver. The local scan remains the only required
copy until the receiver returns a valid V2 VERIFIED acknowledgement.

## V2 transfer

The API is served under `/api/v2/` and requires protocol header `2` plus the
configured bearer token. The upload uses deterministic batches of complete
frame directories, targeting 16 MiB of source data (configurable in code).

V2 uses the streaming `IPHONE3D-BATCH-V2` container rather than multipart.
This lets the iPhone build a temporary batch file and lets Windows verify the
SHA-256 of the exact uploaded body while streaming it to disk. Each record
contains a canonical relative path, byte length, SHA-256, and original bytes.
The receiver extracts only `session.json` and the four allowed files in each
six-digit frame directory. It rejects traversal, conflicting retries, and
body/file hash mismatches.

```text
POST /api/v2/sessions/<id>/start
GET  /api/v2/sessions/<id>/upload-status
PUT  /api/v2/sessions/<id>/batches/<index>
POST /api/v2/sessions/<id>/finalize
```

Resume state is persisted under `incoming/`; a reconnect sends only missing
batch indexes. FINALIZE requires every declared batch and validates completed
`session.json`, sequential frames, and all four files per frame. It then
promotes `incoming/session_<id>` to `sessions/session_<id>` and returns only:

```json
{"protocol_version":2,"status":"verified","session_id":"..."}
```

The iPhone deletes only when all three fields match its completed session.

## Windows service and dashboard

Start the receiver from the repository root:

```powershell
$env:IPHONE3D_RECEIVER_TOKEN = "<receiver token>"
C:\Users\tienq\.conda\envs\iphone3d\python.exe -m windows.scanner_server.app --host 0.0.0.0 --port 8765 --storage-root data
```

The same FastAPI process serves the browser dashboard at `/`. The dashboard
lists verified sessions, starts explicit point-cloud or CPU TSDF mesh jobs,
shows job progress, and loads a bounded browser point-cloud artifact.

Jobs run in a separate spawned process. A reconstruction, CUDA, or Open3D
failure updates `artifacts/session_<id>/job.json`; it cannot terminate the HTTP
receiver or damage the raw verified session.

`/api/v2/diagnostics` reports optional PyTorch CUDA and Open3D CUDA capability.
The current mesh path stays on the previously validated CPU Open3D reference
implementation. CUDA is selected only when actually detected; it is never
assumed from the presence of an NVIDIA GPU.
