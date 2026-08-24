from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parents[1]))

from frame_io import FrameValidationError, load_frame
from fuse_session import SessionValidationError, fuse_loaded_frames, session_frame_dirs


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_session(self, count=2, transforms=None, invalid_index=None, empty_index=None):
        root = Path(self.temp_dir.name) / "session_test"
        frames = root / "frames"
        frames.mkdir(parents=True)
        transforms = transforms or [np.eye(4) for _ in range(count)]
        for index in range(count):
            frame = frames / f"{index:06d}"
            frame.mkdir()
            depth = np.ones((2, 2), dtype=np.float32)
            confidence = np.full((2, 2), 2, dtype=np.uint8)
            rgb = np.full((2, 2, 3), [20 + index, 40, 80], dtype=np.uint8)
            o3d.io.write_image(str(frame / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
            depth.astype("<f4").tofile(frame / "depth.f32")
            confidence.tofile(frame / "confidence.u8")
            metadata = {
                "schema_version": 1, "frame_index": index, "timestamp_seconds": float(index),
                "rgb": {"file": "rgb.jpg", "width": 2, "height": 2},
                "depth": {"file": "depth.f32", "width": 2, "height": 2, "dtype": "float32", "endianness": "little", "unit": "meters"},
                "confidence": {"file": "confidence.u8", "width": 2, "height": 2, "dtype": "uint8"},
                "camera": {"image_width": 2, "image_height": 2, "intrinsics_rows": [[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]], "transform_semantics": "world_from_camera", "transform_rows": transforms[index].tolist(), "coordinate_system": "ARKit", "units": "meters", "forward_axis": "-Z"},
            }
            (frame / "frame.json").write_text(json.dumps(metadata), encoding="utf-8")
            if invalid_index == index:
                (frame / "depth.f32").write_bytes(b"bad")
            if empty_index == index:
                (frame / "confidence.u8").write_bytes(bytes([0, 0, 0, 0]))
        (root / "session.json").write_text(json.dumps({"schema_version": 1, "status": "completed", "frame_count": count}), encoding="utf-8")
        return root

    def test_overlapping_world_observations_and_no_double_transform(self):
        transform = np.eye(4)
        transform[0, 3] = 1.0
        root = self.make_session(1, [transform])
        result = fuse_loaded_frames([load_frame(root / "frames/000000")], min_confidence=2, voxel_size=0.01)
        points = np.asarray(result.raw_cloud.points)
        self.assertAlmostEqual(points[0, 0], 0.75)
        self.assertAlmostEqual(points[0, 2], -1.0)

    def test_voxel_reduces_duplicate_points_and_preserves_colors(self):
        root = self.make_session(2)
        frames = [load_frame(root / "frames" / f"{index:06d}") for index in range(2)]
        result = fuse_loaded_frames(frames, min_confidence=2, voxel_size=0.5)
        self.assertLess(len(result.voxel_cloud.points), len(result.raw_cloud.points))
        self.assertEqual(len(result.raw_cloud.colors), len(result.raw_cloud.points))

    def test_selection_is_sorted_and_supports_stride_and_limit(self):
        root = self.make_session(4)
        selected = session_frame_dirs(root, every_n=2, max_frames=1)
        self.assertEqual([path.name for path in selected], ["000000"])

    def test_invalid_frame_fails_session_validation(self):
        root = self.make_session(2, invalid_index=1)
        with self.assertRaises(FrameValidationError):
            [load_frame(path) for path in session_frame_dirs(root)]

    def test_empty_filtered_result_fails(self):
        root = self.make_session(1, empty_index=0)
        with self.assertRaises(SessionValidationError):
            fuse_loaded_frames([load_frame(root / "frames/000000")], min_confidence=2)

    def test_realistic_translation_and_rotation_are_finite(self):
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 0.1
        angle = np.radians(10.0)
        second[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        root = self.make_session(2, [first, second])
        result = fuse_loaded_frames([load_frame(root / "frames" / f"{i:06d}") for i in range(2)], min_confidence=2, voxel_size=0.01)
        self.assertAlmostEqual(result.trajectory.path_length_meters, 0.1)
        self.assertAlmostEqual(result.trajectory.max_rotation_degrees, 10.0, places=5)

