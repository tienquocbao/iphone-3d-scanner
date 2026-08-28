# Phase 3E live upload and processing

Phase 3E adds an optional live path without changing final transfer authority:

```text
atomic frame commit
  -> bounded four-worker HTTPS upload
  -> Windows staging
  -> complete-frame READY detection
  -> one ordered Open3D worker process per session
```

Live routes use the same protocol header, bearer authentication, canonical
frame paths, SHA-256 validation, temporary files, and atomic rename as the
final receiver. The status route reports uploaded files/bytes, ready and
processed frames, backlog, and lightweight point counts.

At STOP, the app waits for queued live uploads, writes the authoritative final
`session.json`, and the existing BEGIN/missing-files/FINALIZE/VERIFIED ACK flow
remains the source of truth. Live upload never authorizes local deletion; the
completed session remains local until strict final ACK validation succeeds.

Realtime Open3D processing is disposable and isolated from the HTTP receiver.
The receiver sends immutable frame indexes to a bounded worker process. A
processor crash or shutdown timeout is recorded in live status and cannot
prevent raw-file verification or FINALIZE. FINALIZE stops the worker with a
bounded grace period, then verifies and promotes the staged files independently.

The iPhone retries FINALIZE reconciliation for transient 502, 503, 504, and
selected connection failures. Each retry repeats BEGIN, allowing an already
verified session to return a simple VERIFIED response. Permanent protocol or
authentication failures are not retried, and local data is preserved unless
the final ACK is valid.

## Final deletion authority

The receiver remains the authoritative verifier. It validates every file,
size, SHA-256, path, completed `session.json`, frame count, and required frame
files before returning a network ACK with only `protocol_version`, `status`,
and `session_id`. The iPhone deletes its exact local session only when those
three values are valid and the session ID matches. Missing, malformed, or
non-verified responses always preserve the local copy.

The receiver may retain manifest hashes, counts, byte totals, and verification
timestamps in `.verified.json` as internal diagnostics. They are not part of
the iPhone deletion contract.

The receiver endpoints are:

```text
POST /api/v1/live/sessions/<id>/start
PUT  /api/v1/live/sessions/<id>/files/<frame-path>
GET  /api/v1/live/sessions/<id>/status
```
