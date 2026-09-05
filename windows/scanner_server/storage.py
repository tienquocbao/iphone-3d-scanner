"""Persistent V2 batch storage; no processing runs in this module."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable

PROTOCOL_VERSION = 2
MAGIC = b"IPHONE3D-BATCH-V2\n"
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")
FRAME_FILES = {"rgb.jpg", "depth.f32", "confidence.u8", "frame.json"}
SINGLE_PASS_TRANSFORM_ID = "single-pass-identity-v1"


class StorageError(ValueError):
    pass


def _object_reconstruction_state(artifact_dir: Path, pass_count: int, backend: str) -> str:
    """Return missing/current/stale without importing reconstruction dependencies."""

    report_path = artifact_dir / "object" / "reconstruction" / backend / "reconstruction.json"
    mesh_path = artifact_dir / "object" / "reconstruction" / backend / f"object_{backend}_clean.ply"
    if not report_path.is_file() or not mesh_path.is_file():
        return "missing"
    transform_path = artifact_dir / "object" / "registration" / "pass_transforms.json"
    if pass_count > 1:
        if not transform_path.is_file():
            return "stale"
        current = sha256_file(transform_path)
    else:
        current = SINGLE_PASS_TRANSFORM_ID
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        recorded = report.get("pass_transforms_sha256") or report["registration"]["pass_transforms_sha256"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "stale"
    return "current" if recorded == current else "stale"


def _object_reconstruction_ready(artifact_dir: Path, pass_count: int) -> bool:
    if pass_count <= 1:
        return (artifact_dir / "object" / "object_clean.ply").is_file()
    transform_path = artifact_dir / "object" / "registration" / "pass_transforms.json"
    bounds_path = artifact_dir / "object" / "object_registered_clean.ply"
    if not transform_path.is_file() or not bounds_path.is_file():
        return False
    try:
        entries = json.loads(transform_path.read_text(encoding="utf-8"))["passes"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return len(entries) == pass_count and all(
        isinstance(entry, dict)
        and entry.get("registration_status") == ("reference" if entry.get("id") == 0 else "accepted")
        for entry in entries
    )


def safe_session_id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_RE.fullmatch(value):
        raise StorageError("invalid session_id")
    return value


def safe_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StorageError("invalid batch path")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise StorageError("batch path escapes session")
    if len(parts) == 1 and parts[0] == "session.json":
        return value
    if len(parts) == 3 and parts[0] == "frames" and re.fullmatch(r"\d{6}", parts[1]) and parts[2] in FRAME_FILES:
        return value
    raise StorageError("batch path is not a canonical session file")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class UploadStatus:
    session_id: str
    frame_count: int
    batch_count: int
    total_bytes: int
    received_batches: list[int]
    received_bytes: int

    def api(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "batch_count": self.batch_count,
            "total_bytes": self.total_bytes,
            "received_batches": self.received_batches,
            "received_bytes": self.received_bytes,
        }


class TransferStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.incoming = self.root / "incoming"
        self.sessions = self.root / "sessions"
        self.artifacts = self.root / "artifacts"
        for directory in (self.incoming, self.sessions, self.artifacts):
            directory.mkdir(parents=True, exist_ok=True)

    def incoming_session(self, session_id: str) -> Path:
        return self.incoming / f"session_{safe_session_id(session_id)}"

    def completed_session(self, session_id: str) -> Path:
        return self.sessions / f"session_{safe_session_id(session_id)}"

    def start(self, session_id: str, frame_count: int, batch_count: int, total_bytes: int) -> UploadStatus | dict[str, object]:
        session_id = safe_session_id(session_id)
        if min(frame_count, batch_count, total_bytes) < 0 or batch_count < 1:
            raise StorageError("invalid session upload declaration")
        final = self.completed_session(session_id)
        if final.is_dir():
            self._validate_completed(final, session_id)
            return {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id}
        directory = self.incoming_session(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        config = directory / ".upload.json"
        requested = {"protocol_version": PROTOCOL_VERSION, "session_id": session_id, "frame_count": frame_count, "batch_count": batch_count, "total_bytes": total_bytes}
        if config.exists():
            current = json.loads(config.read_text(encoding="utf-8"))
            if current != requested:
                raise StorageError("existing upload declaration does not match")
        else:
            config.write_text(json.dumps(requested, indent=2), encoding="utf-8")
        return self.status(session_id)

    def status(self, session_id: str) -> UploadStatus | dict[str, object]:
        session_id = safe_session_id(session_id)
        final = self.completed_session(session_id)
        if final.is_dir():
            self._validate_completed(final, session_id)
            return {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id}
        directory = self.incoming_session(session_id)
        config_path = directory / ".upload.json"
        if not config_path.is_file():
            raise StorageError("upload has not been started")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        receipts = self._receipts(directory)
        batches = sorted(int(index) for index in receipts)
        return UploadStatus(session_id, int(config["frame_count"]), int(config["batch_count"]), int(config["total_bytes"]), batches, sum(int(value["bytes"]) for value in receipts.values()))

    async def receive_batch(self, session_id: str, batch_index: int, content_length: int, expected_hash: str, body: AsyncIterable[bytes]) -> dict[str, object]:
        session_id = safe_session_id(session_id)
        if batch_index < 0 or content_length < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise StorageError("invalid batch request")
        status = self.status(session_id)
        if isinstance(status, dict):
            raise StorageError("session is already verified")
        if batch_index >= status.batch_count:
            raise StorageError("batch index exceeds declared batch_count")
        directory = self.incoming_session(session_id)
        receipts = self._receipts(directory)
        existing = receipts.get(str(batch_index))
        if existing:
            if existing["sha256"] == expected_hash and int(existing["bytes"]) == content_length:
                return {"protocol_version": PROTOCOL_VERSION, "status": "stored", "session_id": session_id, "batch_index": batch_index, "idempotent": True}
            raise StorageError("batch index already has different content")
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".batch-body-", delete=False) as handle:
            raw_path = Path(handle.name)
            digest = hashlib.sha256()
            count = 0
            async for chunk in body:
                if not chunk:
                    continue
                count += len(chunk)
                if count > content_length:
                    raise StorageError("request body exceeds Content-Length")
                handle.write(chunk)
                digest.update(chunk)
        try:
            if count != content_length or digest.hexdigest() != expected_hash:
                raise StorageError("batch SHA-256 or byte length mismatch")
            self._extract_batch(raw_path, directory, batch_index)
            receipts[str(batch_index)] = {"sha256": expected_hash, "bytes": content_length}
            self._write_receipts(directory, receipts)
        finally:
            raw_path.unlink(missing_ok=True)
        return {"protocol_version": PROTOCOL_VERSION, "status": "stored", "session_id": session_id, "batch_index": batch_index, "idempotent": False}

    def finalize(self, session_id: str) -> dict[str, object]:
        session_id = safe_session_id(session_id)
        final = self.completed_session(session_id)
        if final.is_dir():
            self._validate_completed(final, session_id)
            return {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id}
        status = self.status(session_id)
        if isinstance(status, dict):
            return status
        if status.received_batches != list(range(status.batch_count)):
            raise StorageError("not every declared batch has been received")
        directory = self.incoming_session(session_id)
        self._validate_completed(directory, session_id, expected_frames=status.frame_count)
        (directory / ".upload.json").unlink(missing_ok=True)
        (directory / ".batches.json").unlink(missing_ok=True)
        os.replace(directory, final)
        return {"protocol_version": PROTOCOL_VERSION, "status": "verified", "session_id": session_id}

    def list_sessions(self) -> list[dict[str, object]]:
        result = []
        for path in sorted(self.sessions.glob("session_*"), key=lambda item: item.name, reverse=True):
            try:
                metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
                size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
                scan_mode = metadata.get("scan_mode", "scene")
                if scan_mode not in {"scene", "object"}:
                    scan_mode = "scene"
                pass_count = len(metadata.get("passes") or [])
                artifact_dir = self.artifacts / path.name
                result.append({"session_id": metadata["session_id"], "frame_count": metadata["frame_count"], "total_bytes": size, "created_at": metadata.get("ended_at_utc"), "state": "ready", "scan_mode": scan_mode, "pass_count": pass_count, "object_reconstruction_ready": _object_reconstruction_ready(artifact_dir, pass_count) if scan_mode == "object" else False, "object_tsdf_state": _object_reconstruction_state(artifact_dir, pass_count, "tsdf") if scan_mode == "object" else None, "object_nksr_state": _object_reconstruction_state(artifact_dir, pass_count, "nksr") if scan_mode == "object" else None, "object_poisson_state": _object_reconstruction_state(artifact_dir, pass_count, "poisson") if scan_mode == "object" else None, "object_bpa_state": _object_reconstruction_state(artifact_dir, pass_count, "bpa") if scan_mode == "object" else None})
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return result

    def _extract_batch(self, raw_path: Path, session_directory: Path, batch_index: int) -> None:
        batch_directory = session_directory / f".batch-{batch_index:06d}.tmp"
        batch_directory.mkdir(parents=True, exist_ok=False)
        try:
            with raw_path.open("rb") as source:
                if source.read(len(MAGIC)) != MAGIC:
                    raise StorageError("unsupported batch container")
                while header_line := source.readline():
                    try:
                        header = json.loads(header_line)
                        path = safe_path(header["path"])
                        size = header["size"]
                        expected = header["sha256"]
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise StorageError("invalid batch file header") from exc
                    if type(size) is not int or size < 0 or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise StorageError("invalid batch file metadata")
                    target = batch_directory / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with target.open("wb") as output:
                        remaining = size
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise StorageError("batch ended inside a file")
                            output.write(chunk); digest.update(chunk); remaining -= len(chunk)
                    if digest.hexdigest() != expected or source.read(1) != b"\n":
                        raise StorageError("batch file hash or delimiter mismatch")
            for candidate in batch_directory.rglob("*"):
                if candidate.is_file():
                    relative = candidate.relative_to(batch_directory).as_posix()
                    destination = session_directory / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists() and sha256_file(destination) != sha256_file(candidate):
                        raise StorageError(f"batch conflicts with existing file: {relative}")
                    if not destination.exists():
                        os.replace(candidate, destination)
        finally:
            shutil.rmtree(batch_directory, ignore_errors=True)

    def _validate_completed(self, directory: Path, session_id: str, expected_frames: int | None = None) -> None:
        try:
            metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError("invalid session.json") from exc
        frame_count = metadata.get("frame_count")
        if metadata.get("status") != "completed" or metadata.get("session_id") != session_id or type(frame_count) is not int:
            raise StorageError("session.json is not a matching completed session")
        if expected_frames is not None and frame_count != expected_frames:
            raise StorageError("session frame_count differs from upload declaration")
        frames = directory / "frames"
        names = sorted(path.name for path in frames.iterdir() if path.is_dir()) if frames.is_dir() else []
        if names != [f"{index:06d}" for index in range(frame_count)]:
            raise StorageError("frame directories are not sequential")
        for name in names:
            files = {path.name for path in (frames / name).iterdir() if path.is_file()}
            if files != FRAME_FILES:
                raise StorageError(f"incomplete frame directory: {name}")

    def _receipts(self, directory: Path) -> dict[str, dict[str, object]]:
        path = directory / ".batches.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError("invalid batch receipt") from exc
        if not isinstance(value, dict):
            raise StorageError("invalid batch receipt")
        return value

    def _write_receipts(self, directory: Path, receipts: dict[str, dict[str, object]]) -> None:
        temporary = directory / ".batches.tmp"
        temporary.write_text(json.dumps(receipts, sort_keys=True), encoding="utf-8")
        os.replace(temporary, directory / ".batches.json")
