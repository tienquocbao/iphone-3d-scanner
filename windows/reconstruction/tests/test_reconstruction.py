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
from geometry import depth_to_world_points, sample_rgb, scale_intrinsics


class ReconstructionTests(unittest.TestCase):
    def make_frame(self, depth=None, confidence=None, transform=None, rgb=None):
        root = Path(self.temp_dir.name) / "000000"
        root.mkdir()
        depth = np.asarray(depth if depth is not None else [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        confidence = np.asarray(confidence if confidence is not None else [[0, 1], [2, 1]], dtype=np.uint8)
        rgb = np.asarray(rgb if rgb is not None else np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8))
        rgb_image = o3d.geometry.Image(rgb)
        o3d.io.write_image(str(root / "rgb.jpg"), rgb_image, quality=100)
        depth.astype("<f4").tofile(root / "depth.f32")
        confidence.tofile(root / "confidence.u8")
        transform = np.eye(4, dtype=np.float64) if transform is None else np.asarray(transform, dtype=np.float64)
        rgb_height, rgb_width = rgb.shape[:2]
        principal_x = 0.5 if rgb_width == 2 else rgb_width / 2
        principal_y = 0.5 if rgb_height == 2 else rgb_height / 2
        metadata = {
            "schema_version": 1,
            "frame_index": 0,
            "timestamp_seconds": 1.0,
            "timestamp_origin": "ARSession",
            "rgb": {"file": "rgb.jpg", "width": int(rgb_width), "height": int(rgb_height)},
            "depth": {"file": "depth.f32", "width": 2, "height": 2, "dtype": "float32", "endianness": "little", "unit": "meters"},
            "confidence": {"file": "confidence.u8", "width": 2, "height": 2, "dtype": "uint8", "encoding": {"0": "low", "1": "medium", "2": "high"}},
            "camera": {"image_width": int(rgb_width), "image_height": int(rgb_height), "intrinsics_rows": [[2.0, 0.0, principal_x], [0.0, 2.0, principal_y], [0.0, 0.0, 1.0]], "transform_semantics": "world_from_camera", "transform_rows": transform.tolist(), "coordinate_system": "ARKit", "units": "meters", "forward_axis": "-Z"},
        }
        (root / "frame.json").write_text(json.dumps(metadata), encoding="utf-8")
        return root

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_identity_pose_converts_forward_depth_to_negative_z(self):
        frame_dir = self.make_frame(depth=[[1.0, 1.0], [1.0, 1.0]], confidence=[[2, 2], [2, 2]])
        frame = load_frame(frame_dir)
        points, _, _ = depth_to_world_points(frame, min_confidence=2)
        np.testing.assert_allclose(points[0], [-0.25, 0.25, -1.0], atol=1e-6)

    def test_translation_is_applied(self):
        transform = np.eye(4)
        transform[0, 3] = 2.5
        frame = load_frame(self.make_frame(depth=[[1.0, 1.0], [1.0, 1.0]], confidence=[[2, 2], [2, 2]], transform=transform))
        points, _, _ = depth_to_world_points(frame, min_confidence=2)
        self.assertAlmostEqual(points[0, 0], 2.25)

    def test_intrinsics_are_scaled_independently(self):
        frame = load_frame(self.make_frame())
        scaled = scale_intrinsics(frame)
        self.assertEqual((scaled.fx, scaled.fy, scaled.cx, scaled.cy), (2.0, 2.0, 0.5, 0.5))

    def test_little_endian_depth_is_loaded_exactly(self):
        values = [[1.25, 2.5], [3.75, 5.0]]
        frame = load_frame(self.make_frame(depth=values))
        np.testing.assert_array_equal(frame.depth, np.asarray(values, dtype=np.float32))

    def test_confidence_filtering(self):
        frame = load_frame(self.make_frame(depth=np.ones((2, 2)), confidence=[[0, 1], [2, 2]]))
        points, _, _ = depth_to_world_points(frame, min_confidence=2)
        self.assertEqual(len(points), 2)

    def test_rgb_mapping_uses_depth_pixel_centers(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:32, :32] = [255, 0, 0]
        rgb[32:, 32:] = [255, 255, 255]
        frame = load_frame(self.make_frame(depth=np.ones((2, 2)), confidence=np.full((2, 2), 2), rgb=rgb))
        colors = sample_rgb(frame, np.asarray([[0, 0], [1, 1]], dtype=np.int32))
        np.testing.assert_allclose(colors[0], [1, 0, 0], atol=0.02)
        np.testing.assert_allclose(colors[1], [1, 1, 1], atol=0.02)

    def test_malformed_depth_size_fails(self):
        frame_dir = self.make_frame()
        (frame_dir / "depth.f32").write_bytes(b"bad")
        with self.assertRaises(FrameValidationError):
            load_frame(frame_dir)

    def test_non_affine_transform_fails(self):
        transform = np.eye(4)
        transform[3, 3] = 2.0
        with self.assertRaises(FrameValidationError):
            load_frame(self.make_frame(transform=transform))


if __name__ == "__main__":
    unittest.main()
