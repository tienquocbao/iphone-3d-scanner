"""FastAPI V2 receiver, job API, and dashboard host."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from .jobs import JobManager, device_diagnostics
from .storage import PROTOCOL_VERSION, StorageError, TransferStore


class StartPayload(BaseModel):
    protocol_version: int
    session_id: str
    frame_count: int = Field(ge=0)
    batch_count: int = Field(ge=1)
    total_bytes: int = Field(ge=0)


class FinalizePayload(BaseModel):
    protocol_version: int
    session_id: str


class JobPayload(BaseModel):
    kind: Literal["pointcloud", "mesh", "object_pointcloud", "registered_object_pointcloud", "object_tsdf", "object_nksr"]
    device: Literal["auto", "cpu", "cuda"] = "auto"


def create_app(storage_root: Path, bearer_token: str = "") -> FastAPI:
    store = TransferStore(storage_root)
    jobs = JobManager(store.artifacts, store.sessions)
    app = FastAPI(title="iPhone 3D Scanner V2", version="2")
    app.state.store = store
    app.state.jobs = jobs
    app.state.bearer_token = bearer_token

    def authorize(authorization: str | None = Header(default=None), x_protocol_version: str | None = Header(default=None)) -> None:
        if x_protocol_version != str(PROTOCOL_VERSION):
            raise HTTPException(400, "missing or unsupported X-Protocol-Version")
        if bearer_token:
            supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
            if not supplied or not hmac.compare_digest(supplied, bearer_token):
                raise HTTPException(401, "authentication required")

    def translate(exc: Exception) -> HTTPException:
        return HTTPException(400, str(exc))

    @app.get("/api/v2/health")
    def health(_: None = Depends(authorize)) -> dict[str, object]:
        return {"protocol_version": PROTOCOL_VERSION, "status": "ok", "auth_required": bool(bearer_token)}

    @app.post("/api/v2/sessions/{session_id}/start")
    def start(session_id: str, payload: StartPayload, _: None = Depends(authorize)) -> dict[str, object]:
        if payload.protocol_version != PROTOCOL_VERSION or payload.session_id != session_id:
            raise HTTPException(400, "protocol or session ID mismatch")
        try:
            value = store.start(session_id, payload.frame_count, payload.batch_count, payload.total_bytes)
            return value.api() if hasattr(value, "api") else value
        except StorageError as exc:
            raise translate(exc) from exc

    @app.get("/api/v2/sessions/{session_id}/upload-status")
    def upload_status(session_id: str, _: None = Depends(authorize)) -> dict[str, object]:
        try:
            value = store.status(session_id)
            return value.api() if hasattr(value, "api") else value
        except StorageError as exc:
            raise translate(exc) from exc

    @app.put("/api/v2/sessions/{session_id}/batches/{batch_index}")
    async def batch(session_id: str, batch_index: int, request: Request, _: None = Depends(authorize), x_batch_sha256: str | None = Header(default=None), content_length: int | None = Header(default=None)) -> dict[str, object]:
        if request.headers.get("content-type") != "application/vnd.iphone3d.batch-v2":
            raise HTTPException(415, "expected application/vnd.iphone3d.batch-v2")
        if content_length is None or x_batch_sha256 is None:
            raise HTTPException(400, "Content-Length and X-Batch-SHA256 are required")
        try:
            return await store.receive_batch(session_id, batch_index, content_length, x_batch_sha256, request.stream())
        except StorageError as exc:
            raise translate(exc) from exc

    @app.post("/api/v2/sessions/{session_id}/finalize")
    def finalize(session_id: str, payload: FinalizePayload, _: None = Depends(authorize)) -> dict[str, object]:
        if payload.protocol_version != PROTOCOL_VERSION or payload.session_id != session_id:
            raise HTTPException(400, "protocol or session ID mismatch")
        try:
            return store.finalize(session_id)
        except StorageError as exc:
            raise translate(exc) from exc

    @app.get("/api/v2/sessions")
    def sessions(_: None = Depends(authorize)) -> dict[str, object]:
        return {"sessions": store.list_sessions()}

    @app.post("/api/v2/sessions/{session_id}/jobs")
    def start_job(session_id: str, payload: JobPayload, _: None = Depends(authorize)) -> dict[str, object]:
        try:
            return jobs.start(session_id, payload.kind, payload.device)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/v2/sessions/{session_id}/job")
    def job(session_id: str, _: None = Depends(authorize)) -> dict[str, object]:
        return jobs.status(session_id)

    @app.get("/api/v2/sessions/{session_id}/job/events")
    async def job_events(session_id: str, _: None = Depends(authorize)) -> StreamingResponse:
        async def stream():
            last = None
            for _ in range(300):
                current = jobs.status(session_id)
                rendered = str(current)
                if rendered != last:
                    last = rendered
                    yield f"data: {current}\n\n"
                if current.get("state") in {"done", "failed"}:
                    break
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/v2/sessions/{session_id}/artifacts/{name:path}")
    def artifact(session_id: str, name: str, _: None = Depends(authorize)) -> FileResponse:
        allowed = {"pointcloud.ply", "mesh_mesh_clean.ply", "mesh_mesh_raw.ply", "mesh_tsdf_pointcloud.ply", "job.json", "object/object_raw.ply", "object/object_clean.ply", "object/object_registered_raw.ply", "object/object_registered_clean.ply", "object/processing.json", "object/registration/pass_transforms.json", "object/registration/registration.json", "object/reconstruction/tsdf/object_tsdf_raw.ply", "object/reconstruction/tsdf/object_tsdf_clean.ply", "object/reconstruction/tsdf/reconstruction.json", "object/reconstruction/nksr/object_nksr_raw.ply", "object/reconstruction/nksr/object_nksr_clean.ply", "object/reconstruction/nksr/reconstruction.json", "object/reconstruction/nksr/input_summary.json"}
        valid_mask = bool(re.fullmatch(r"object/masks/(?:pass_[0-9]{3}/)?[0-9]{6}\.png", name))
        valid_pass = bool(re.fullmatch(r"object/passes/pass_[0-9]{3}_(?:raw|clean)\.ply", name))
        if name not in allowed and not valid_mask and not valid_pass:
            raise HTTPException(404, "artifact not found")
        path = store.artifacts / f"session_{session_id}" / name
        if not path.is_file():
            raise HTTPException(404, "artifact not found")
        return FileResponse(path)

    @app.get("/api/v2/diagnostics")
    def diagnostics(_: None = Depends(authorize)) -> dict[str, object]:
        return device_diagnostics()

    web_root = Path(__file__).resolve().parents[1] / "web"

    @app.middleware("http")
    async def refresh_dashboard_assets(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in {"/", "/app.js", "/viewer.js", "/styles.css"}:
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="iPhone 3D Scanner V2 receiver and dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    token = os.environ.get("IPHONE3D_RECEIVER_TOKEN", "").strip()
    uvicorn.run(create_app(args.storage_root, token), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
