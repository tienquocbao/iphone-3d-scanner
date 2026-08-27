# Phase 3E live upload and processing

Phase 3E adds an optional live path without changing final transfer authority:

```text
atomic frame commit
  -> bounded four-worker HTTPS upload
  -> Windows staging
  -> complete-frame READY detection
  -> one ordered processing worker per session
```

Live routes use the same protocol header, bearer authentication, canonical
frame paths, SHA-256 validation, temporary files, and atomic rename as the
final receiver. The status route reports uploaded files/bytes, ready and
processed frames, backlog, and lightweight point counts.

At STOP, the app waits for queued live uploads, writes the authoritative final
`session.json`, and the existing BEGIN/missing-files/FINALIZE/VERIFIED ACK flow
remains the source of truth. Live upload never authorizes local deletion; the
completed session remains local until strict final ACK validation succeeds.

## Manifest identity

Protocol version 1 uses explicit canonical UTF-8 bytes for `manifest_sha256`,
not serialized JSON. The bytes contain a trailing newline and one value per
line in this order:

```text
protocol_version
session_id
path
size
sha256
path
size
sha256
...
```

File triples are sorted lexically by canonical relative path. Paths and
SHA-256 values are restricted to the validated ASCII forms, and sizes are
decimal integers. Swift and Python must hash these exact bytes. The server
returns this hash in both `ready` and `verified` responses.

The receiver endpoints are:

```text
POST /api/v1/live/sessions/<id>/start
PUT  /api/v1/live/sessions/<id>/files/<frame-path>
GET  /api/v1/live/sessions/<id>/status
```
