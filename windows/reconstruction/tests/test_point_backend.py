from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from frame_io import FrameData
from geometry import depth_to_world_points
from point_backend import depth_to_world_points_backend


class PointBackendTests(unittest.TestCase):
    def test_cpu_backend_matches_phase_2_reference(self) -> None:
        metadata = {"camera": {"image_width": 2, "image_height": 2}}
        frame = FrameData(Path("000000"), metadata, np.zeros((2, 2, 3), dtype=np.uint8), np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32), np.array([[2, 1], [2, 0]], dtype=np.uint8), np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]]), np.eye(4))
        expected, pixels, _ = depth_to_world_points(frame, min_confidence=1)
        actual, actual_pixels = depth_to_world_points_backend(frame, "cpu", min_confidence=1)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(actual_pixels, pixels)
