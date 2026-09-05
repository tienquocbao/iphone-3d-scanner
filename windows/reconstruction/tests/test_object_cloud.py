from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parents[1]))

from foreground import ForegroundConfig
from object_cloud import ObjectCloudConfig, build_object_cloud


class ObjectCloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "session_object"
        frame = self.root / "frames" / "000000"
        frame.mkdir(parents=True)
        rgb = np.full((8, 8, 3), [20, 180, 30], dtype=np.uint8)
        rgb[2:6, 2:6] = [220, 25, 30]
        o3d.io.write_image(str(frame / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
        np.ones((8, 8), dtype="<f4").tofile(frame / "depth.f32")
        np.full((8, 8), 2, dtype=np.uint8).tofile(frame / "confidence.u8")
        metadata = {
            "schema_version": 1, "frame_index": 0, "timestamp_seconds": 0,
            "rgb": {"file": "rgb.jpg", "width": 8, "height": 8},
            "depth": {"file": "depth.f32", "width": 8, "height": 8, "dtype": "float32", "endianness": "little", "unit": "meters"},
            "confidence": {"file": "confidence.u8", "width": 8, "height": 8, "dtype": "uint8"},
            "camera": {"image_width": 8, "image_height": 8, "intrinsics_rows": [[8, 0, 3.5], [0, 8, 3.5], [0, 0, 1]], "transform_semantics": "world_from_camera", "transform_rows": np.eye(4).tolist(), "coordinate_system": "ARKit", "units": "meters", "forward_axis": "-Z"},
        }
        (frame / "frame.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.root / "session.json").write_text(json.dumps({"schema_version": 1, "session_id": "object", "status": "completed", "frame_count": 1, "scan_mode": "object", "passes": [{"id": 0, "start_frame": 0, "end_frame": 0}]}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_object_cloud_excludes_green_background_before_backprojection(self):
        artifacts = Path(self.temp.name) / "artifacts"
        config = ObjectCloudConfig(voxel_size=0.001, outlier_neighbors=2, foreground=ForegroundConfig(morphology_kernel=1, minimum_component_pixels=1))
        report = build_object_cloud(self.root, artifacts, config)
        self.assertEqual(report["points_before_mask"], 64)
        self.assertGreaterEqual(report["foreground_points"], 12)
        self.assertLess(report["foreground_points"], 32)
        self.assertTrue((artifacts / "object" / "object_raw.ply").is_file())
        self.assertTrue((artifacts / "object" / "object_clean.ply").is_file())
        self.assertTrue((artifacts / "object" / "masks" / "000000.png").is_file())
        raw = o3d.io.read_point_cloud(str(artifacts / "object" / "object_raw.ply"))
        self.assertEqual(len(raw.points), report["foreground_points"])

    def test_scene_session_is_rejected_without_mutating_raw_data(self):
        metadata_path = self.root / "session.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")); metadata["scan_mode"] = "scene"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scan_mode=object"):
            build_object_cloud(self.root, Path(self.temp.name) / "artifacts")
        self.assertTrue((self.root / "frames" / "000000" / "depth.f32").is_file())


if __name__ == "__main__":
    unittest.main()
