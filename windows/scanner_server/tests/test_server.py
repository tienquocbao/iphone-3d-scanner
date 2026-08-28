from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from server import PROTOCOL_VERSION, Handler, Receiver, ThreadingHTTPServer, TransferError, canonical_manifest_bytes, json_bytes, manifest_sha256


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp_dir.name) / "received"
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.receiver = Receiver(self.storage)
        self.httpd.auth_token = "test-token"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def request(self, method, path, body, content_type="application/json", token="test-token"):
        connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=30)
        data = body if isinstance(body, bytes) else json_bytes(body)
        headers = {"Content-Length": str(len(data)), "X-Protocol-Version": str(PROTOCOL_VERSION), "Content-Type": content_type}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, payload

    def manifest(self):
        files = {
            "session.json": b'{"schema_version":1,"session_id":"abc","status":"completed","frame_count":1}',
            "frames/000000/frame.json": b'{"schema_version":1}',
            "frames/000000/rgb.jpg": b"rgb",
            "frames/000000/depth.f32": b"depth",
            "frames/000000/confidence.u8": b"confidence",
        }
        return {"protocol_version": 1, "session_id": "abc", "files": [{"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for path, data in sorted(files.items())]}, files

    def test_begin_put_finalize_and_idempotent_begin(self):
        manifest, files = self.manifest()
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual(status, 200)
        self.assertEqual(response["status"], "ready")
        for path, data in files.items():
            status, response = self.request("PUT", "/v1/sessions/abc/files/" + path, data, "application/octet-stream")
            self.assertEqual(status, 200)
            self.assertEqual(response["status"], "stored")
        status, response = self.request("POST", "/v1/sessions/abc/finalize", manifest)
        self.assertEqual(status, 200)
        self.assertEqual(response, {"protocol_version": 1, "status": "verified", "session_id": "abc"})

    def test_manifest_hash_fixture_is_fixed_and_not_legacy_json_hash(self):
        fixture = json.loads((Path(__file__).parents[3] / "shared" / "schema" / "manifest_hash_fixture.json").read_text(encoding="utf-8"))
        entries = {item["path"]: {"size": item["size"], "sha256": item["sha256"]} for item in fixture["files"]}
        self.assertEqual(manifest_sha256(fixture["protocol_version"], fixture["session_id"], entries), fixture["expected_canonical_sha256"])
        legacy = json.dumps({"protocol_version": 1, "session_id": "abc", "files": [{"path": path, **entries[path]} for path in sorted(entries)]}, separators=(",", ":")).encode()
        self.assertNotEqual(hashlib.sha256(legacy).hexdigest(), fixture["expected_canonical_sha256"])

    def test_legacy_verified_receipt_is_migrated_without_reupload(self):
        manifest, files = self.manifest()
        self.request("POST", "/v1/sessions/begin", manifest)
        for path, data in files.items():
            self.request("PUT", "/v1/sessions/abc/files/" + path, data, "application/octet-stream")
        status, verified = self.request("POST", "/v1/sessions/abc/finalize", manifest)
        self.assertEqual(status, 200)
        receipt_path = self.storage / "session_abc" / ".verified.json"
        legacy = json.loads((self.storage / "session_abc" / ".verified.json").read_text(encoding="utf-8"))
        legacy["manifest_sha256"] = "f" * 64
        legacy["file_count"] = legacy.pop("verified_file_count")
        legacy["total_bytes"] = legacy.pop("verified_total_bytes")
        receipt_path.write_text(json.dumps(legacy), encoding="utf-8")

        status, retry = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual((status, retry["status"]), (200, "verified"))
        self.assertEqual(retry, {"protocol_version": 1, "status": "verified", "session_id": "abc", "missing": []})
        self.assertTrue((self.storage / "session_abc" / ".verified.json").is_file())
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual((status, response["status"]), (200, "verified"))
        self.assertEqual(response, {"protocol_version": 1, "status": "verified", "session_id": "abc", "missing": []})

    def test_health_requires_correct_bearer_token(self):
        status, _ = self.request("GET", "/api/v1/health", b"", token=None)
        self.assertEqual(status, 401)
        status, _ = self.request("GET", "/api/v1/health", b"", token="wrong")
        self.assertEqual(status, 401)
        status, response = self.request("GET", "/api/v1/health", b"")
        self.assertEqual(status, 200)
        self.assertEqual((response["protocol_version"], response["status"]), (1, "ok"))

    def test_request_logs_never_contain_bearer_token(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status, _ = self.request("GET", "/api/v1/health", b"")
        self.assertEqual(status, 200)
        self.assertNotIn("test-token", output.getvalue())

    def test_bad_checksum_does_not_commit_file(self):
        manifest, files = self.manifest()
        self.request("POST", "/v1/sessions/begin", manifest)
        status, response = self.request("PUT", "/v1/sessions/abc/files/session.json", b"x" * len(files["session.json"]), "application/octet-stream")
        self.assertEqual(status, 400)
        self.assertIn("sha256", response["error"])
        self.assertFalse((self.storage / ".session_abc.staging" / "session.json").exists())

    def test_finalize_refuses_missing_files_and_traversal(self):
        manifest, _ = self.manifest()
        self.request("POST", "/v1/sessions/begin", manifest)
        status, response = self.request("POST", "/v1/sessions/abc/finalize", manifest)
        self.assertEqual(status, 400)
        self.assertIn("missing", response["error"])
        bad = dict(manifest)
        bad["files"] = [{"path": "../escape", "size": 0, "sha256": "0" * 64}]
        status, response = self.request("POST", "/v1/sessions/begin", bad)
        self.assertEqual(status, 400)

    def test_processor_crash_does_not_affect_receiver_finalize_or_health(self):
        manifest, files = self.manifest()
        receiver = Receiver(self.storage, processor_mode="crash")
        receiver.begin(manifest)
        staging = receiver.staging("abc")
        for path, data in files.items():
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        receiver.start_live("abc")
        self.httpd.receiver = receiver
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and receiver.live_status("abc")["processing_error"] is None:
            time.sleep(0.05)
        status, health = self.request("GET", "/api/v1/health", {})
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertIn("processor exited", receiver.live_status("abc")["processing_error"])
        self.assertEqual(receiver.finalize(manifest), {"protocol_version": 1, "status": "verified", "session_id": "abc"})

    def test_hung_processor_is_terminated_and_finalize_verifies_raw_files(self):
        manifest, files = self.manifest()
        receiver = Receiver(self.storage, processor_mode="hang")
        receiver.begin(manifest)
        staging = receiver.staging("abc")
        for path, data in files.items():
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        receiver.start_live("abc")
        started = time.monotonic()
        result = receiver.finalize(manifest)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(result, {"protocol_version": 1, "status": "verified", "session_id": "abc"})

    def test_finalize_waits_for_active_live_put_then_retry_succeeds(self):
        manifest, files = self.manifest()
        receiver = Receiver(self.storage)
        receiver.begin(manifest)
        receiver.start_live("abc")
        live = receiver.live_session("abc")
        live.begin_upload()
        result = {}
        finalizer = threading.Thread(target=lambda: result.setdefault("error", self._finalize_direct(receiver, manifest)))
        finalizer.start()
        time.sleep(0.1)
        self.assertTrue(finalizer.is_alive())
        self.assertTrue(receiver.staging("abc").exists())
        live.end_upload()
        finalizer.join(timeout=3)
        self.assertFalse(finalizer.is_alive())
        self.assertIsInstance(result["error"], TransferError)
        self.assertTrue(receiver.staging("abc").exists())

        for path, data in files.items():
            destination = receiver.staging("abc") / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        self.assertEqual(receiver.finalize(manifest), {"protocol_version": 1, "status": "verified", "session_id": "abc"})

    def test_concurrent_finalization_is_idempotent(self):
        manifest, files = self.manifest()
        receiver = Receiver(self.storage)
        receiver.begin(manifest)
        for path, data in files.items():
            destination = receiver.staging("abc") / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(receiver.finalize, manifest) for _ in range(2)]
            results = [future.result() for future in futures]
        self.assertEqual(results, [{"protocol_version": 1, "status": "verified", "session_id": "abc"}] * 2)

    @staticmethod
    def _finalize_direct(receiver, manifest):
        try:
            receiver.finalize(manifest)
        except TransferError as exc:
            return exc
        return None

    def test_begin_reports_rejected_manifest_path_without_normalizing_it(self):
        manifest, _ = self.manifest()
        manifest["files"][0] = {**manifest["files"][0], "path": "/frames/000000/rgb.jpg"}
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual(status, 400)
        self.assertEqual(response["path"], "/frames/000000/rgb.jpg")
        self.assertIn("normalized forward-slash", response["error"])

    def test_resume_after_receiver_restart_returns_only_missing_files(self):
        manifest, files = self.manifest()
        self.request("POST", "/v1/sessions/begin", manifest)
        first_path, first_data = next(iter(files.items()))
        self.request("PUT", "/v1/sessions/abc/files/" + first_path, first_data, "application/octet-stream")
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.receiver = Receiver(self.storage)
        self.httpd.auth_token = "test-token"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual(status, 200)
        self.assertEqual(response["status"], "ready")
        self.assertNotIn(first_path, response["missing"])
        self.assertEqual(len(response["missing"]), len(files) - 1)

    def test_legacy_verified_receipt_is_normalized_on_retry(self):
        manifest, files = self.manifest()
        self.request("POST", "/v1/sessions/begin", manifest)
        for path, data in files.items():
            self.request("PUT", "/v1/sessions/abc/files/" + path, data, "application/octet-stream")
        status, first = self.request("POST", "/v1/sessions/abc/finalize", manifest)
        self.assertEqual(status, 200)
        receipt_path = self.storage / "session_abc" / ".verified.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["file_count"] = receipt.pop("verified_file_count")
        receipt["total_bytes"] = receipt.pop("verified_total_bytes")
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual((status, response["status"]), (200, "verified"))
        self.assertEqual(response, {"protocol_version": 1, "status": "verified", "session_id": "abc", "missing": []})
        normalized = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertIn("verified_file_count", normalized)
        self.assertIn("verified_total_bytes", normalized)

    def test_realistic_129_frame_manifest_and_transfer(self):
        session_id = "large-test"
        files = {
            "session.json": json.dumps({"schema_version": 1, "session_id": session_id, "status": "completed", "frame_count": 129}, separators=(",", ":")).encode()
        }
        for index in range(129):
            prefix = f"frames/{index:06d}/"
            files[prefix + "frame.json"] = json.dumps({"schema_version": 1, "frame_index": index}).encode()
            files[prefix + "rgb.jpg"] = b"rgb" + bytes([index % 256])
            files[prefix + "depth.f32"] = b"depth" + bytes([index % 256])
            files[prefix + "confidence.u8"] = b"confidence" + bytes([index % 256])
        manifest = {"protocol_version": 1, "session_id": session_id, "files": [{"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for path, data in sorted(files.items())]}
        payload_bytes = len(json_bytes(manifest))
        self.assertEqual(len(manifest["files"]), 517)
        self.assertLess(payload_bytes, 100_000)
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual((status, len(response["missing"])), (200, 517))
        for path, data in files.items():
            status, _ = self.request("PUT", "/v1/sessions/large-test/files/" + path, data, "application/octet-stream")
            self.assertEqual(status, 200)
        status, response = self.request("POST", "/v1/sessions/large-test/finalize", manifest)
        self.assertEqual(response, {"protocol_version": 1, "status": "verified", "session_id": "large-test"})

    def test_live_routes_require_auth_and_stage_concurrent_frame_files(self):
        status, _ = self.request("POST", "/api/v1/live/sessions/live-test/start", {}, token="wrong")
        self.assertEqual(status, 401)
        status, response = self.request("POST", "/api/v1/live/sessions/live-test/start", {})
        self.assertEqual(status, 200)
        self.assertEqual(response["state"], "recording")

        files = {
            "frames/000000/rgb.jpg": b"rgb-live",
            "frames/000000/depth.f32": b"depth-live",
            "frames/000000/confidence.u8": b"confidence-live",
            "frames/000000/frame.json": b"{\"frame_index\":0}",
        }

        def upload(item):
            path, data = item
            connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=30)
            headers = {
                "Content-Length": str(len(data)),
                "Content-Type": "application/octet-stream",
                "X-Protocol-Version": str(PROTOCOL_VERSION),
                "X-File-SHA256": hashlib.sha256(data).hexdigest(),
                "Authorization": "Bearer test-token",
            }
            connection.request("PUT", "/api/v1/live/sessions/live-test/files/" + path, data, headers)
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            return response.status, payload

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(upload, files.items()))
        self.assertTrue(all(status == 200 for status, _ in results))
        status, response = self.request("GET", "/api/v1/live/sessions/live-test/status", b"")
        self.assertEqual(status, 200)
        self.assertEqual(response["uploaded_files"], 4)
        self.assertEqual(response["ready_frames"], 1)

        status, response = self.request(
            "PUT",
            "/api/v1/live/sessions/live-test/files/../escape",
            b"x",
            "application/octet-stream",
        )
        self.assertEqual(status, 400)

    def test_final_begin_reconciles_live_staged_frame_files(self):
        manifest, files = self.manifest()
        self.request("POST", "/api/v1/live/sessions/abc/start", {})
        for path, data in files.items():
            if path == "session.json":
                continue
            headers = {
                "Content-Length": str(len(data)),
                "Content-Type": "application/octet-stream",
                "X-File-SHA256": hashlib.sha256(data).hexdigest(),
            }
            connection = HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=30)
            headers["X-Protocol-Version"] = str(PROTOCOL_VERSION)
            headers["Authorization"] = "Bearer test-token"
            connection.request("PUT", "/api/v1/live/sessions/abc/files/" + path, data, headers)
            response = connection.getresponse()
            response.read()
            connection.close()
        status, response = self.request("POST", "/v1/sessions/begin", manifest)
        self.assertEqual(status, 200)
        self.assertEqual(response["missing"], ["session.json"])
