from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import open3d as o3d

from windows.scanner_server.app import create_app
from windows.scanner_server.storage import MAGIC
from windows.scanner_server.jobs import nksr_diagnostics


class V2ReceiverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = TestClient(create_app(self.root, "test-token"))
        self.headers = {"Authorization": "Bearer test-token", "X-Protocol-Version": "2"}
        self.files = {
            "session.json": b'{"schema_version":1,"session_id":"abc","status":"completed","frame_count":1}',
            "frames/000000/rgb.jpg": b"rgb",
            "frames/000000/depth.f32": b"depth",
            "frames/000000/confidence.u8": b"confidence",
            "frames/000000/frame.json": b"{}",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def start(self) -> None:
        response = self.client.post("/api/v2/sessions/abc/start", headers=self.headers, json={"protocol_version": 2, "session_id": "abc", "frame_count": 1, "batch_count": 1, "total_bytes": sum(map(len, self.files.values()))})
        self.assertEqual(response.status_code, 200)

    @staticmethod
    def batch(files: dict[str, bytes]) -> bytes:
        body = bytearray(MAGIC)
        for path, data in files.items():
            body.extend(json.dumps({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}, separators=(",", ":")).encode())
            body.extend(b"\n"); body.extend(data); body.extend(b"\n")
        return bytes(body)

    def upload(self, body: bytes | None = None, index: int = 0) -> int:
        body = body or self.batch(self.files)
        headers = {**self.headers, "Content-Type": "application/vnd.iphone3d.batch-v2", "X-Batch-SHA256": hashlib.sha256(body).hexdigest(), "Content-Length": str(len(body))}
        return self.client.put(f"/api/v2/sessions/abc/batches/{index}", headers=headers, content=body).status_code

    def test_batch_resume_idempotence_and_finalize(self) -> None:
        self.start(); self.assertEqual(self.upload(), 200); self.assertEqual(self.upload(), 200)
        status = self.client.get("/api/v2/sessions/abc/upload-status", headers=self.headers).json()
        self.assertEqual(status["received_batches"], [0])
        response = self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"})
        self.assertEqual(response.json(), {"protocol_version": 2, "status": "verified", "session_id": "abc"})
        self.assertEqual(self.client.post("/api/v2/sessions/abc/start", headers=self.headers, json={"protocol_version": 2, "session_id": "abc", "frame_count": 1, "batch_count": 1, "total_bytes": sum(map(len, self.files.values()))}).json()["status"], "verified")

    def test_hash_and_traversal_rejected(self) -> None:
        self.start(); body = self.batch(self.files)
        headers = {**self.headers, "Content-Type": "application/vnd.iphone3d.batch-v2", "X-Batch-SHA256": "0" * 64, "Content-Length": str(len(body))}
        self.assertEqual(self.client.put("/api/v2/sessions/abc/batches/0", headers=headers, content=body).status_code, 400)
        bad = self.batch({"frames/000000/../escape": b"x"})
        headers["X-Batch-SHA256"] = hashlib.sha256(bad).hexdigest(); headers["Content-Length"] = str(len(bad))
        self.assertEqual(self.client.put("/api/v2/sessions/abc/batches/0", headers=headers, content=bad).status_code, 400)

    def test_incomplete_finalize_and_auth_rejected(self) -> None:
        self.start()
        self.assertEqual(self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"}).status_code, 400)
        self.assertEqual(self.client.get("/api/v2/sessions", headers={"X-Protocol-Version": "2"}).status_code, 401)

    def test_list_sessions_requires_no_processing_lifecycle(self) -> None:
        self.start(); self.assertEqual(self.upload(), 200)
        self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"})
        sessions = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"]
        self.assertEqual(sessions[0]["state"], "ready")
        self.assertEqual(sessions[0]["scan_mode"], "scene")
        self.assertFalse((self.root / "incoming" / "session_abc").exists())

    def test_object_session_is_listed_and_has_dedicated_job_and_artifacts(self) -> None:
        session = self.root / "sessions" / "session_object"
        object_dir = self.root / "artifacts" / "session_object" / "object"
        frame = session / "frames" / "000000"
        frame.mkdir(parents=True); object_dir.mkdir(parents=True)
        rgb = np.full((24, 24, 3), [20, 180, 30], dtype=np.uint8); rgb[6:18, 6:18] = [220, 25, 30]
        o3d.io.write_image(str(frame / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
        np.ones((24, 24), dtype="<f4").tofile(frame / "depth.f32")
        np.full((24, 24), 2, dtype=np.uint8).tofile(frame / "confidence.u8")
        (frame / "frame.json").write_text(json.dumps({"schema_version": 1, "rgb": {"file": "rgb.jpg", "width": 24, "height": 24}, "depth": {"file": "depth.f32", "width": 24, "height": 24, "dtype": "float32", "endianness": "little", "unit": "meters"}, "confidence": {"file": "confidence.u8", "width": 24, "height": 24, "dtype": "uint8"}, "camera": {"image_width": 24, "image_height": 24, "intrinsics_rows": [[24, 0, 11.5], [0, 24, 11.5], [0, 0, 1]], "transform_semantics": "world_from_camera", "transform_rows": np.eye(4).tolist(), "coordinate_system": "ARKit", "units": "meters", "forward_axis": "-Z"}}), encoding="utf-8")
        (session / "session.json").write_text(json.dumps({"schema_version": 1, "session_id": "object", "status": "completed", "frame_count": 1, "scan_mode": "object", "passes": [{"id": 0, "start_frame": 0, "end_frame": 0}]}), encoding="utf-8")
        sessions = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"]
        self.assertEqual(sessions[0]["scan_mode"], "object")
        self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind": "object_pointcloud", "device": "cpu"}).status_code, 200)
        self.client.app.state.jobs.processes["object"].join(timeout=5)
        job_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
        self.assertEqual(job_state["state"], "done", job_state)
        self.assertEqual(self.client.get("/api/v2/health", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/object_clean.ply", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind": "object_tsdf", "device": "cpu"}).status_code, 200)
        self.client.app.state.jobs.processes["object"].join(timeout=10)
        tsdf_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
        self.assertEqual(tsdf_state["state"], "done", tsdf_state)
        self.assertEqual(self.client.get("/api/v2/health", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/tsdf/object_tsdf_clean.ply", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/tsdf/reconstruction.json", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind": "object_poisson", "device": "cpu"}).status_code, 200)
        self.client.app.state.jobs.processes["object"].join(timeout=15)
        poisson_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
        self.assertEqual(poisson_state["state"], "done", poisson_state)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/poisson/object_poisson_clean.ply", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind": "object_bpa", "device": "cpu"}).status_code, 200)
        self.client.app.state.jobs.processes["object"].join(timeout=15)
        bpa_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
        self.assertEqual(bpa_state["state"], "done", bpa_state)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/bpa/object_bpa_clean.ply", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/comparison.json", headers=self.headers).status_code, 200)
        listed = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"][0]
        self.assertEqual(listed["object_tsdf_state"], "current")
        self.assertEqual(listed["object_poisson_state"], "current")
        self.assertEqual(listed["object_bpa_state"], "current")
        diagnostics = self.client.get("/api/v2/diagnostics", headers=self.headers).json()
        self.assertIn("nksr", diagnostics)
        self.assertFalse(diagnostics["nksr"]["available"])
        unavailable = self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind": "object_nksr", "device": "auto"})
        self.assertEqual(unavailable.status_code, 400)
        self.assertIn("NKSR unavailable", unavailable.text)
        fake_runner = Path(__file__).parents[2] / "reconstruction" / "tests" / "fake_nksr_runner.py"
        with patch.dict(os.environ, {"IPHONE3D_NKSR_PYTHON":sys.executable, "IPHONE3D_NKSR_RUNNER":str(fake_runner)}):
            nksr_diagnostics.cache_clear()
            self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind":"object_nksr","device":"auto"}).status_code, 200)
            self.client.app.state.jobs.processes["object"].join(timeout=15)
            nksr_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
            self.assertEqual(nksr_state["state"], "done", nksr_state)
            self.assertEqual(self.client.get("/api/v2/health", headers=self.headers).status_code, 200)
            self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/nksr/object_nksr_clean.ply", headers=self.headers).status_code, 200)
            os.environ["IPHONE3D_FAKE_NKSR_MODE"] = "failure"
            self.assertEqual(self.client.post("/api/v2/sessions/object/jobs", headers=self.headers, json={"kind":"object_nksr","device":"auto"}).status_code, 200)
            self.client.app.state.jobs.processes["object"].join(timeout=15)
            failed_state = self.client.get("/api/v2/sessions/object/job", headers=self.headers).json()
            self.assertEqual(failed_state["state"], "failed", failed_state)
            self.assertIn("deliberate adapter failure", failed_state["message"])
            self.assertEqual(self.client.get("/api/v2/health", headers=self.headers).status_code, 200)
            self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/tsdf/object_tsdf_clean.ply", headers=self.headers).status_code, 200)
            self.assertEqual(self.client.get("/api/v2/sessions/object/artifacts/object/reconstruction/nksr/object_nksr_clean.ply", headers=self.headers).status_code, 200)
        nksr_diagnostics.cache_clear()

    def test_changed_registration_marks_object_tsdf_stale(self) -> None:
        session = self.root / "sessions" / "session_stale"
        artifact = self.root / "artifacts" / "session_stale" / "object"
        transform = artifact / "registration" / "pass_transforms.json"
        report = artifact / "reconstruction" / "tsdf" / "reconstruction.json"
        mesh = report.parent / "object_tsdf_clean.ply"
        session.mkdir(parents=True); transform.parent.mkdir(parents=True); report.parent.mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({"session_id":"stale","frame_count":2,"scan_mode":"object","passes":[{"id":0},{"id":1}]}), encoding="utf-8")
        transform.write_text('{"version":1}', encoding="utf-8")
        digest = hashlib.sha256(transform.read_bytes()).hexdigest()
        report.write_text(json.dumps({"registration":{"pass_transforms_sha256":digest}}), encoding="utf-8")
        mesh.write_bytes(b"ply")
        for backend in ("poisson", "bpa"):
            surface_report = artifact / "reconstruction" / backend / "reconstruction.json"
            surface_mesh = surface_report.parent / f"object_{backend}_clean.ply"
            surface_report.parent.mkdir(parents=True, exist_ok=True)
            surface_report.write_text(json.dumps({"pass_transforms_sha256": digest}), encoding="utf-8")
            surface_mesh.write_bytes(b"ply")
        listed = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"][0]
        self.assertEqual(listed["object_tsdf_state"], "current")
        self.assertEqual(listed["object_poisson_state"], "current")
        self.assertEqual(listed["object_bpa_state"], "current")
        transform.write_text('{"version":2}', encoding="utf-8")
        listed = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"][0]
        self.assertEqual(listed["object_tsdf_state"], "stale")
        self.assertEqual(listed["object_poisson_state"], "stale")
        self.assertEqual(listed["object_bpa_state"], "stale")

    def test_nksr_artifact_provenance_becomes_stale_with_registration(self) -> None:
        session = self.root / "sessions" / "session_nksr"
        artifact = self.root / "artifacts" / "session_nksr" / "object"
        transform = artifact / "registration" / "pass_transforms.json"
        report = artifact / "reconstruction" / "nksr" / "reconstruction.json"
        mesh = report.parent / "object_nksr_clean.ply"
        session.mkdir(parents=True); transform.parent.mkdir(parents=True); report.parent.mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({"session_id":"nksr","frame_count":2,"scan_mode":"object","passes":[{"id":0},{"id":1}]}), encoding="utf-8")
        transform.write_text('{"version":1}', encoding="utf-8")
        digest = hashlib.sha256(transform.read_bytes()).hexdigest()
        report.write_text(json.dumps({"pass_transforms_sha256":digest}), encoding="utf-8"); mesh.write_bytes(b"ply")
        listed = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"][0]
        self.assertEqual(listed["object_nksr_state"], "current")
        transform.write_text('{"version":2}', encoding="utf-8")
        listed = self.client.get("/api/v2/sessions", headers=self.headers).json()["sessions"][0]
        self.assertEqual(listed["object_nksr_state"], "stale")

    def test_object_job_rejects_scene_session(self) -> None:
        self.start(); self.assertEqual(self.upload(), 200)
        self.assertEqual(self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"}).status_code, 200)
        response = self.client.post("/api/v2/sessions/abc/jobs", headers=self.headers, json={"kind": "object_pointcloud", "device": "cpu"})
        self.assertEqual(response.status_code, 400)

    def test_processing_failure_does_not_break_http_receiver(self) -> None:
        self.start(); self.upload()
        self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"})
        self.assertEqual(self.client.post("/api/v2/sessions/abc/jobs", headers=self.headers, json={"kind": "pointcloud", "device": "cpu"}).status_code, 200)
        deadline = time.monotonic() + 10
        state = "running"
        while time.monotonic() < deadline:
            state = self.client.get("/api/v2/sessions/abc/job", headers=self.headers).json()["state"]
            if state == "failed":
                break
            time.sleep(0.1)
        self.assertEqual(state, "failed")
        self.assertEqual(self.client.get("/api/v2/health", headers=self.headers).status_code, 200)

    def test_dashboard_assets_are_local_and_no_session_state_is_visible(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn('type="importmap"', index.text)
        self.assertIn('"three": "/vendor/three/three.module.js"', index.text)
        self.assertEqual(index.headers["cache-control"], "no-cache")
        app_js = self.client.get("/app.js")
        self.assertEqual(app_js.status_code, 200)
        self.assertNotIn("cdn.jsdelivr.net", app_js.text)
        self.assertNotIn("OrbitControls", app_js.text)
        self.assertIn("loadSessions", app_js.text)
        self.assertIn("No verified sessions", app_js.text)
        self.assertIn("Authentication required", app_js.text)
        self.assertIn("Cannot reach receiver", app_js.text)
        self.assertIn("Build Object Mesh (TSDF)", app_js.text)
        self.assertIn("STALE", app_js.text)
        self.assertIn("NKSR unavailable", app_js.text)
        self.assertIn("Build Object Mesh (NKSR)", app_js.text)
        self.assertIn("Build Object Mesh (Poisson)", app_js.text)
        self.assertIn("Build Object Mesh (BPA)", app_js.text)
        self.assertIn("Compare object backends", app_js.text)
        self.assertEqual(self.client.get("/viewer.js").status_code, 200)
        self.assertIn("geometry.index !== null", self.client.get("/viewer.js").text)
        self.assertEqual(self.client.get("/vendor/three/three.module.js").status_code, 200)
        self.assertEqual(self.client.get("/vendor/three/addons/controls/OrbitControls.js").status_code, 200)
        self.assertEqual(self.client.get("/vendor/three/addons/loaders/PLYLoader.js").status_code, 200)

    def test_dashboard_session_api_and_auth_failure_are_explicit(self) -> None:
        self.start(); self.assertEqual(self.upload(), 200)
        self.assertEqual(self.client.post("/api/v2/sessions/abc/finalize", headers=self.headers, json={"protocol_version": 2, "session_id": "abc"}).status_code, 200)
        response = self.client.get("/api/v2/sessions", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sessions"][0]["session_id"], "abc")
        self.assertEqual(self.client.get("/api/v2/sessions", headers={"X-Protocol-Version": "2"}).status_code, 401)
