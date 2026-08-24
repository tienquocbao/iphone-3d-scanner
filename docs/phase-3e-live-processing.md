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

The receiver endpoints are:

```text
POST /api/v1/live/sessions/<id>/start
PUT  /api/v1/live/sessions/<id>/files/<frame-path>
GET  /api/v1/live/sessions/<id>/status
```
