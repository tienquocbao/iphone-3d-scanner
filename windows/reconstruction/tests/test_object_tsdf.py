from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parents[1]))

from foreground import ForegroundConfig, calibrate_green_background, foreground_mask
from frame_io import load_frame
from object_tsdf import (
    ObjectTSDFConfig,
    build_object_tsdf,
    load_object_from_pass_transforms,
    object_depth_mask,
)
from registration import compose_object_from_camera


def _write_cloud(path: Path) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.array(
            [
                [-0.35, -0.35, -1.12],
                [0.35, -0.35, -1.12],
                [-0.35, 0.35, -0.88],
                [0.35, 0.35, -0.88],
            ]
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), cloud)


def _write_frame(frame_dir: Path, index: int, pose: np.ndarray) -> None:
    frame_dir.mkdir(parents=True)
    width, height = 32, 24
    rgb = np.full((height, width, 3), [20, 180, 30], dtype=np.uint8)
    rgb[6:18, 8:24] = [220, 25, 30]
    assert o3d.io.write_image(str(frame_dir / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
    depth = np.full((height, width), 1.4, dtype="<f4")
    depth[6:18, 8:24] = 1.0
    depth.tofile(frame_dir / "depth.f32")
    np.full((height, width), 2, dtype=np.uint8).tofile(frame_dir / "confidence.u8")
    metadata = {
        "schema_version": 1,
        "frame_index": index,
        "timestamp_seconds": float(index),
        "rgb": {"file": "rgb.jpg", "width": width, "height": height},
        "depth": {"file": "depth.f32", "width": width, "height": height, "dtype": "float32", "endianness": "little", "unit": "meters"},
        "confidence": {"file": "confidence.u8", "width": width, "height": height, "dtype": "uint8"},
        "camera": {
            "image_width": width,
            "image_height": height,
            "intrinsics_rows": [[32, 0, 15.5], [0, 32, 11.5], [0, 0, 1]],
            "transform_semantics": "world_from_camera",
            "transform_rows": pose.tolist(),
            "coordinate_system": "ARKit",
            "units": "meters",
            "forward_axis": "-Z",
        },
    }
    (frame_dir / "frame.json").write_text(json.dumps(metadata), encoding="utf-8")


class ObjectTSDFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.session = self.base / "session_object"
        self.artifacts = self.base / "artifacts"

    def tearDown(self):
        self.temp.cleanup()

    def _session(self, poses: list[np.ndarray], passes: list[dict[str, int]]) -> None:
        for index, pose in enumerate(poses):
            _write_frame(self.session / "frames" / f"{index:06d}", index, pose)
        self.session.mkdir(exist_ok=True)
        (self.session / "session.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "object",
                    "status": "completed",
                    "frame_count": len(poses),
                    "scan_mode": "object",
                    "passes": passes,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _config() -> ObjectTSDFConfig:
        return ObjectTSDFConfig(
            voxel_length=0.02,
            sdf_trunc=0.08,
            min_depth_m=0.1,
            max_depth_m=2.0,
            minimum_foreground_samples=4,
            mesh_min_component_triangles=1,
            foreground=ForegroundConfig(morphology_kernel=1, minimum_component_pixels=1),
        )

    def test_single_pass_masked_tsdf_excludes_green_background_plane(self):
        self._session([np.eye(4)], [{"id": 0, "start_frame": 0, "end_frame": 0}])
        _write_cloud(self.artifacts / "object" / "object_clean.ply")
        report = build_object_tsdf(self.session, self.artifacts, self._config())
        self.assertEqual(report["integrated_frames"], 1)
        self.assertEqual(report["registration"]["pass_transforms_sha256"], "single-pass-identity-v1")
        self.assertGreater(report["mesh"]["clean"]["triangles"], 0)
        mesh = o3d.io.read_triangle_mesh(
            str(self.artifacts / "object" / "reconstruction" / "tsdf" / "object_tsdf_clean.ply")
        )
        z = np.asarray(mesh.vertices)[:, 2]
        self.assertLess(float(np.max(np.abs(z + 1.0))), 0.12)

    def test_moved_pass_uses_composed_object_relative_pose(self):
        object_from_pass = np.eye(4)
        object_from_pass[0, 3] = 0.30
        pass_world_from_camera = np.linalg.inv(object_from_pass)
        correct = compose_object_from_camera(object_from_pass, pass_world_from_camera)
        np.testing.assert_allclose(correct, np.eye(4), atol=1e-12)
        self.assertGreater(abs(pass_world_from_camera[0, 3]), 0.29)
        self._session(
            [np.eye(4), pass_world_from_camera],
            [{"id": 0, "start_frame": 0, "end_frame": 0}, {"id": 1, "start_frame": 1, "end_frame": 1}],
        )
        transform_path = self.artifacts / "object" / "registration" / "pass_transforms.json"
        transform_path.parent.mkdir(parents=True)
        transform_path.write_text(
            json.dumps(
                {
                    "canonical_pass": 0,
                    "passes": [
                        {"id": 0, "registration_status": "reference", "object_from_pass": np.eye(4).tolist()},
                        {"id": 1, "registration_status": "accepted", "object_from_pass": object_from_pass.tolist()},
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_cloud(self.artifacts / "object" / "object_registered_clean.ply")
        report = build_object_tsdf(self.session, self.artifacts, self._config())
        self.assertEqual(report["integrated_frames"], 2)
        expected_hash = hashlib.sha256(transform_path.read_bytes()).hexdigest()
        self.assertEqual(report["registration"]["pass_transforms_sha256"], expected_hash)
        extent_x = report["mesh"]["clean"]["bbox_max"][0] - report["mesh"]["clean"]["bbox_min"][0]
        self.assertLess(extent_x, 0.8, "wrong pass pose would create separated/ghost geometry")

    def test_foreground_depth_mask_rejects_background_before_integration(self):
        self._session([np.eye(4)], [{"id": 0, "start_frame": 0, "end_frame": 0}])
        frame = load_frame(self.session / "frames" / "000000")
        config = self._config()
        model = calibrate_green_background([frame.rgb], config.foreground)
        rgb_mask = foreground_mask(frame.rgb, model, config.foreground)
        mask, valid, kept = object_depth_mask(
            frame, rgb_mask, np.eye(4), np.array([-1, -1, -1.2]), np.array([1, 1, -0.8]), config
        )
        self.assertEqual(valid, frame.depth.size)
        self.assertGreater(kept, 100)
        self.assertLess(kept, frame.depth.size // 2)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[12, 16])

    def test_rejected_registration_blocks_tsdf(self):
        self._session(
            [np.eye(4), np.eye(4)],
            [{"id": 0, "start_frame": 0, "end_frame": 0}, {"id": 1, "start_frame": 1, "end_frame": 1}],
        )
        path = self.artifacts / "object" / "registration" / "pass_transforms.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "canonical_pass": 0,
                    "passes": [
                        {"id": 0, "registration_status": "reference", "object_from_pass": np.eye(4).tolist()},
                        {"id": 1, "registration_status": "rejected", "object_from_pass": np.eye(4).tolist()},
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "OBJECT_REGISTRATION_REQUIRED"):
            load_object_from_pass_transforms(
                [{"id": 0}, {"id": 1}], path
            )
        self.assertTrue((self.session / "frames" / "000001" / "depth.f32").is_file())

    def test_empty_foreground_fails_without_fake_mesh(self):
        self._session([np.eye(4)], [{"id": 0, "start_frame": 0, "end_frame": 0}])
        frame_path = self.session / "frames" / "000000"
        green = np.full((24, 32, 3), [20, 180, 30], dtype=np.uint8)
        o3d.io.write_image(str(frame_path / "rgb.jpg"), o3d.geometry.Image(green), quality=100)
        _write_cloud(self.artifacts / "object" / "object_clean.ply")
        with self.assertRaisesRegex(ValueError, "integrated zero frames"):
            build_object_tsdf(self.session, self.artifacts, self._config())
        self.assertFalse((self.artifacts / "object" / "reconstruction" / "tsdf" / "object_tsdf_clean.ply").exists())


if __name__ == "__main__":
    unittest.main()
