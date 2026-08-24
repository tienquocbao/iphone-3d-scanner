"""Trusted-LAN receiver for verified iPhone scan-session transfers."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROTOCOL_VERSION = 1
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_JSON_BYTES = 4 * 1024 * 1024


class TransferError(ValueError):
    pass


def json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_session_id(value: object) -> str:
    if not isinstance(value, str) or not SESSION_ID_RE.fullmatch(value):
        raise TransferError("invalid session_id")
    return value


def safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TransferError("invalid relative file path")
    path = Path(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise TransferError("file path must remain inside the session")
    normalized = "/".join(path.parts)
    if normalized != value or normalized.startswith("/"):
        raise TransferError("file path must use normalized forward-slash components")
    return normalized


def validate_manifest(payload: object) -> tuple[str, dict[str, dict[str, int | str]], str]:
    if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise TransferError("unsupported or missing protocol_version")
    session_id = safe_session_id(payload.get("session_id"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise TransferError("manifest files must be a non-empty list")
    entries: dict[str, dict[str, int | str]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise TransferError("manifest file entry must be an object")
        relative = safe_relative_path(item.get("path"))
        size = item.get("size")
        digest = item.get("sha256")
        if relative in entries or not isinstance(size, int) or size < 0:
            raise TransferError("manifest contains duplicate or invalid file size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise TransferError(f"invalid sha256 for {relative}")
        entries[relative] = {"size": size, "sha256": digest}
    manifest_hash = hashlib.sha256(json_bytes({"protocol_version": PROTOCOL_VERSION, "session_id": session_id, "files": [{"path": p, **entries[p]} for p in sorted(entries)]})).hexdigest()
    return session_id, entries, manifest_hash


class Receiver:
    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def staging(self, session_id: str) -> Path:
        return self.storage_root / f".session_{session_id}.staging"

    def completed(self, session_id: str) -> Path:
        return self.storage_root / f"session_{session_id}"

    def begin(self, payload: object) -> dict[str, object]:
        session_id, entries, manifest_hash = validate_manifest(payload)
        print(f"BEGIN files={len(entries)} bytes={sum(int(entry['size']) for entry in entries.values())}", flush=True)
        final = self.completed(session_id)
        if final.is_dir() and (final / ".verified.json").is_file():
            verified = json.loads((final / ".verified.json").read_text(encoding="utf-8"))
            if verified.get("manifest_sha256") == manifest_hash:
                return {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id, "missing": [], "manifest_sha256": verified.get("manifest_sha256"), "file_count": verified.get("file_count"), "total_bytes": verified.get("total_bytes")}
            raise TransferError("session already verified with a different manifest")
        staging = self.staging(session_id)
        staging.mkdir(parents=True, exist_ok=True)
        manifest = {"protocol_version": PROTOCOL_VERSION, "session_id": session_id, "files": [{"path": path, **entries[path]} for path in sorted(entries)], "manifest_sha256": manifest_hash}
        (staging / ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        missing = [path for path, entry in entries.items() if not self._matches(staging / path, entry)]
        return {"protocol_version": PROTOCOL_VERSION, "status": "ready", "session_id": session_id, "missing": sorted(missing)}

    def put_file(self, session_id: str, relative: str, content_length: int, body) -> dict[str, object]:
        session_id = safe_session_id(session_id)
        relative = safe_relative_path(relative)
        staging = self.staging(session_id)
        manifest_path = staging / ".manifest.json"
        if not manifest_path.is_file():
            raise TransferError("transfer has not been begun")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {item["path"]: item for item in manifest["files"]}
        if relative not in entries:
            raise TransferError("file is not present in manifest")
        expected = entries[relative]
        if content_length != expected["size"]:
            raise TransferError("Content-Length does not match manifest")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".upload-", delete=False) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            remaining = content_length
            while remaining:
                chunk = body.read(min(1024 * 1024, remaining))
                if not chunk:
                    temporary.unlink(missing_ok=True)
                    raise TransferError("request body ended before Content-Length")
                handle.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != expected["sha256"]:
            temporary.unlink(missing_ok=True)
            raise TransferError("sha256 mismatch")
        os.replace(temporary, destination)
        return {"protocol_version": PROTOCOL_VERSION, "status": "stored", "session_id": session_id, "path": relative, "sha256": expected["sha256"]}

    def finalize(self, payload: object) -> dict[str, object]:
        session_id, entries, manifest_hash = validate_manifest(payload)
        staging = self.staging(session_id)
        if not staging.is_dir():
            raise TransferError("transfer has not been begun")
        for relative, entry in entries.items():
            if not self._matches(staging / relative, entry):
                raise TransferError(f"missing or invalid file: {relative}")
        session_path = staging / "session.json"
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TransferError(f"invalid session.json: {exc}") from exc
        if session.get("status") != "completed" or session.get("session_id") != session_id:
            raise TransferError("session.json is not a matching completed session")
        frame_count = session.get("frame_count")
        frame_paths = sorted(path for path in entries if path.startswith("frames/") and path.endswith("/frame.json"))
        if not isinstance(frame_count, int) or frame_count != len(frame_paths):
            raise TransferError("session frame_count does not match manifest frame metadata")
        for index in range(frame_count):
            prefix = f"frames/{index:06d}/"
            required = {prefix + name for name in ("rgb.jpg", "depth.f32", "confidence.u8", "frame.json")}
            if not required.issubset(entries):
                raise TransferError(f"manifest is missing required files for frame {index:06d}")
        verified = {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id, "manifest_sha256": manifest_hash, "file_count": len(entries), "total_bytes": sum(int(e["size"]) for e in entries.values())}
        (staging / ".verified.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
        final = self.completed(session_id)
        if final.exists():
            if not final.is_dir():
                raise TransferError("completed session path is not a directory")
            shutil.rmtree(final)
        os.replace(staging, final)
        print("FINALIZE validation=PASS", flush=True)
        return verified

    @staticmethod
    def _matches(path: Path, entry: dict[str, int | str]) -> bool:
        return path.is_file() and path.stat().st_size == entry["size"] and file_sha256(path) == entry["sha256"]


class Handler(BaseHTTPRequestHandler):
    server_version = "ScannerReceiver/1"

    @property
    def receiver(self) -> Receiver:
        return self.server.receiver  # type: ignore[attr-defined]

    def _send(self, status: int, payload: object) -> None:
        data = json_bytes(payload)
        summary = ""
        if isinstance(payload, dict):
            if payload.get("status") == "verified":
                summary = " VERIFIED"
            elif "missing" in payload and isinstance(payload["missing"], list):
                summary = f" missing={len(payload['missing'])}"
            elif "error" in payload:
                summary = f" error={str(payload['error'])[:160].replace(chr(10), ' ')}"
        print(f"RESPONSE status={status}{summary}", flush=True)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError as exc:
            raise TransferError("invalid Content-Length") from exc
        if length < 0 or length > MAX_JSON_BYTES:
            raise TransferError("invalid JSON body length")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransferError(f"invalid JSON body: {exc}") from exc

    def _check_protocol(self) -> None:
        if self.headers.get("X-Protocol-Version") != str(PROTOCOL_VERSION):
            raise TransferError("missing or unsupported X-Protocol-Version")

    def _check_auth(self) -> None:
        expected = getattr(self.server, "auth_token", None)
        if not expected:
            return
        value = self.headers.get("Authorization", "")
        supplied = value[7:] if value.startswith("Bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied, expected):
            self._send(HTTPStatus.UNAUTHORIZED, {"protocol_version": PROTOCOL_VERSION, "error": "authentication required"})
            raise PermissionError("authentication failed")

    def do_POST(self) -> None:
        try:
            request_path = urlparse(self.path).path
            print(f"REQUEST method=POST route={request_path}", flush=True)
            self._check_auth()
            self._check_protocol()
            path = urlparse(self.path).path.rstrip("/")
            payload = self._json_body()
            if path == "/v1/sessions/begin":
                if isinstance(payload, dict):
                    print(f"REQUEST stage=BEGIN session={payload.get('session_id', '<invalid>')}", flush=True)
                self._send(HTTPStatus.OK, self.receiver.begin(payload))
            elif path.startswith("/v1/sessions/") and path.endswith("/finalize"):
                path_session_id = safe_session_id(path.split("/")[3])
                if not isinstance(payload, dict) or payload.get("session_id") != path_session_id:
                    raise TransferError("finalize path and manifest session_id differ")
                print(f"REQUEST stage=FINALIZE session={path_session_id}", flush=True)
                self._send(HTTPStatus.OK, self.receiver.finalize(payload))
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except PermissionError:
            return
        except TransferError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"protocol_version": PROTOCOL_VERSION, "error": str(exc)})
        except Exception as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"protocol_version": PROTOCOL_VERSION, "error": str(exc)})

    def do_PUT(self) -> None:
        try:
            print(f"REQUEST method=PUT route={urlparse(self.path).path}", flush=True)
            self._check_auth()
            self._check_protocol()
            parts = [unquote(part) for part in urlparse(self.path).path.split("/")]
            if len(parts) < 6 or parts[1:4] != ["v1", "sessions", parts[3]] or parts[4] != "files":
                raise TransferError("invalid file upload path")
            session_id = safe_session_id(parts[3])
            relative = "/".join(parts[5:])
            print(f"REQUEST stage=UPLOAD session={session_id} path={relative}", flush=True)
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0:
                raise TransferError("Content-Length is required")
            response = self.receiver.put_file(session_id, relative, length, self.rfile)
            print(f"UPLOAD received={length} sha256=PASS", flush=True)
            self._send(HTTPStatus.OK, response)
        except PermissionError:
            return
        except (TransferError, ValueError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"protocol_version": PROTOCOL_VERSION, "error": str(exc)})
        except Exception as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"protocol_version": PROTOCOL_VERSION, "error": str(exc)})

    def do_GET(self) -> None:
        try:
            print(f"REQUEST method=GET route={urlparse(self.path).path}", flush=True)
            self._check_auth()
            if urlparse(self.path).path != "/api/v1/health":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, {"protocol_version": PROTOCOL_VERSION, "status": "ok", "auth_required": bool(getattr(self.server, "auth_token", None))})
        except PermissionError:
            return

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive iPhone scan sessions over a trusted LAN")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-root", type=Path, default=Path("samples/received"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.receiver = Receiver(args.storage_root)  # type: ignore[attr-defined]
    server.auth_token = os.environ.get("IPHONE3D_RECEIVER_TOKEN", "").strip()  # type: ignore[attr-defined]
    print(f"Scanner receiver listening on http://{args.host}:{args.port}")
    print(f"Storage root: {server.receiver.storage_root}")  # type: ignore[attr-defined]
    print(f"Authentication: {'enabled' if server.auth_token else 'DISABLED (trusted LAN mode)'}")  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
