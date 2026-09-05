from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import open3d as o3d

from windows.scanner_server.app import create_app
from windows.scanner_server.storage import MAGIC


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
        self.assertEqual(self.client.get("/viewer.js").status_code, 200)
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
