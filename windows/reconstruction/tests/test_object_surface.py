from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

from foreground import ForegroundConfig
from object_surface import (
    BPAConfig,
    PoissonConfig,
    SurfaceNormalConfig,
    SurfaceReconstructionError,
    _crop_to_bounds,
    adaptive_bpa_radii,
    build_object_bpa,
    build_object_poisson,
    orient_normals_to_sensors,
    prepare_surface_cloud,
)
from surface_input import SurfaceInput, SurfaceInputConfig, joint_voxel_aggregate


def _asymmetric_observations(count: int = 1200) -> SurfaceInput:
    """A box plus offset cone avoids the ambiguity of a sphere or centered cube."""

    box = o3d.geometry.TriangleMesh.create_box(0.18, 0.12, 0.08)
    box.translate([-0.09, -0.06, -0.04])
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.035, height=0.09, resolution=12)
    cone.translate([0.04, 0.0, 0.04])
    mesh = box + cone
    mesh.compute_vertex_normals()
    o3d.utility.random.seed(7)
    cloud = mesh.sample_points_poisson_disk(count)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    sensors = points + normals * 0.35
    colors = np.clip((points - points.min(axis=0)) / np.ptp(points, axis=0), 0, 1)
    return SurfaceInput(points.astype(np.float32), sensors.astype(np.float32), (colors * 255).astype(np.uint8), {}, "synthetic")


def _write_object_session(root: Path) -> tuple[Path, Path]:
    session, artifacts = root / "session_object", root / "artifacts"
    frame = session / "frames" / "000000"
    frame.mkdir(parents=True)
    rgb = np.full((48, 48, 3), [20, 180, 30], dtype=np.uint8)
    rgb[12:36, 12:36] = [220, 35, 28]
    o3d.io.write_image(str(frame / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
    depth = np.full((48, 48), 1.4, dtype="<f4")
    v, u = np.indices((24, 24))
    depth[12:36, 12:36] = 0.9 + 0.12 * ((u - 11.5) ** 2 + (v - 11.5) ** 2) / 11.5**2
    depth.tofile(frame / "depth.f32")
    np.full((48, 48), 2, dtype=np.uint8).tofile(frame / "confidence.u8")
    (frame / "frame.json").write_text(json.dumps({
        "schema_version": 1, "frame_index": 0, "timestamp_seconds": 0.0,
        "rgb": {"file": "rgb.jpg", "width": 48, "height": 48},
        "depth": {"file": "depth.f32", "width": 48, "height": 48, "dtype": "float32", "endianness": "little", "unit": "meters"},
        "confidence": {"file": "confidence.u8", "width": 48, "height": 48, "dtype": "uint8"},
        "camera": {"image_width": 48, "image_height": 48, "intrinsics_rows": [[48, 0, 23.5], [0, 48, 23.5], [0, 0, 1]], "transform_semantics": "world_from_camera", "transform_rows": np.eye(4).tolist(), "coordinate_system": "ARKit", "units": "meters", "forward_axis": "-Z"},
    }), encoding="utf-8")
    (session / "session.json").write_text(json.dumps({"schema_version": 1, "session_id": "object", "status": "completed", "frame_count": 1, "scan_mode": "object", "passes": [{"id": 0, "start_frame": 0, "end_frame": 0}]}), encoding="utf-8")
    bounds = o3d.geometry.PointCloud()
    bounds.points = o3d.utility.Vector3dVector([[-0.4, -0.4, -1.5], [0.4, 0.4, -0.5]])
    target = artifacts / "object" / "object_clean.ply"
    target.parent.mkdir(parents=True)
    o3d.io.write_point_cloud(str(target), bounds)
    return session, artifacts


class ObjectSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _normal_config(self) -> SurfaceNormalConfig:
        return SurfaceNormalConfig(radius_m=0.035, max_nn=48)

    def _input_config(self) -> SurfaceInputConfig:
        return SurfaceInputConfig(input_voxel_size_m=0.005, outlier_neighbors=8, max_input_points=10000, foreground=ForegroundConfig(morphology_kernel=1, minimum_component_pixels=1))

    def test_joint_observation_aggregation_keeps_point_sensor_color_aligned(self) -> None:
        xyz = np.array([[.001, 0, 0], [.002, 0, 0], [.02, 0, 0]])
        sensors = np.array([[0, 0, 1], [.2, 0, 1], [1, 0, 1]])
        colors = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]])
        points, origins, result_colors = joint_voxel_aggregate(xyz, sensors, colors, .01)
        self.assertEqual(len(points), 2)
        np.testing.assert_allclose(origins[0], [.1, 0, 1])
        np.testing.assert_allclose(result_colors[0], [50, 0, 0])

    def test_sensor_orientation_corrects_random_normal_signs(self) -> None:
        points = np.tile(np.array([[0.0, 0.0, 0.0]]), (24, 1))
        points[:, 0] = np.linspace(-.02, .02, len(points))
        sensors = points + np.array([0.0, 0.0, 1.0])
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))
        normals[::2] *= -1
        oriented, flips = orient_normals_to_sensors(points, normals, sensors)
        self.assertEqual(flips, 12)
        self.assertTrue(np.all(np.einsum("ij,ij->i", oriented, sensors - points) > 0))

    def test_poisson_and_bpa_reconstruct_asymmetric_metric_surface(self) -> None:
        prepared = _asymmetric_observations()
        cloud, diagnostics = prepare_surface_cloud(prepared, self._normal_config())
        self.assertEqual(diagnostics["normal_count"], len(prepared.xyz))
        self.assertGreater(diagnostics["flipped_by_sensor_count"], 0)
        poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=6)
        bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, o3d.utility.DoubleVector(adaptive_bpa_radii(cloud, (1.5, 3.0, 6.0))))
        self.assertGreater(len(poisson.triangles), 100)
        self.assertGreater(len(bpa.triangles), 100)
        extent = np.asarray(poisson.get_axis_aligned_bounding_box().get_extent())
        self.assertGreater(float(np.max(extent)), 0.12)
        self.assertLess(float(np.max(extent)), 0.5)

    def test_adaptive_bpa_radii_scale_with_input_coordinates(self) -> None:
        prepared = _asymmetric_observations(600)
        cloud, _ = prepare_surface_cloud(prepared, self._normal_config())
        twice = o3d.geometry.PointCloud(cloud)
        twice.points = o3d.utility.Vector3dVector(np.asarray(cloud.points) * 2.0)
        first, second = adaptive_bpa_radii(cloud, (1.5, 3.0)), adaptive_bpa_radii(twice, (1.5, 3.0))
        np.testing.assert_allclose(np.asarray(second) / np.asarray(first), [2.0, 2.0], rtol=.05)

    def test_bounds_crop_removes_external_geometry_without_damaging_inside_mesh(self) -> None:
        inside = o3d.geometry.TriangleMesh.create_box(.1, .1, .1)
        outside = o3d.geometry.TriangleMesh.create_box(.1, .1, .1)
        outside.translate([2, 0, 0])
        cropped, removed = _crop_to_bounds(inside + outside, np.array([-.1, -.1, -.1]), np.array([.2, .2, .2]))
        self.assertGreater(removed, 0)
        self.assertEqual(len(cropped.triangles), len(inside.triangles))

    def test_native_builds_publish_meshes_and_failed_rebuild_preserves_previous_poisson(self) -> None:
        session, artifacts = _write_object_session(self.root)
        poisson_config = PoissonConfig(depth=6, density_quantile=.01, minimum_component_triangles=1, input=self._input_config(), normals=SurfaceNormalConfig(radius_m=.06, max_nn=32))
        bpa_config = BPAConfig(radius_multipliers=(1.5, 3.0, 6.0), minimum_component_triangles=1, input=self._input_config(), normals=SurfaceNormalConfig(radius_m=.06, max_nn=32))
        poisson = build_object_poisson(session, artifacts, poisson_config)
        bpa = build_object_bpa(session, artifacts, bpa_config)
        self.assertGreater(poisson["mesh"]["clean"]["triangles"], 0)
        self.assertGreater(poisson["density_filter"]["removed_vertices"], 0)
        self.assertGreater(bpa["mesh"]["clean"]["triangles"], 0)
        clean = artifacts / "object" / "reconstruction" / "poisson" / "object_poisson_clean.ply"
        previous = clean.read_bytes()
        with patch("object_surface.create_poisson_mesh", side_effect=SurfaceReconstructionError("deliberate rebuild failure")):
            with self.assertRaisesRegex(SurfaceReconstructionError, "deliberate"):
                build_object_poisson(session, artifacts, poisson_config)
        self.assertEqual(clean.read_bytes(), previous)
        bpa_clean = artifacts / "object" / "reconstruction" / "bpa" / "object_bpa_clean.ply"
        bpa_previous = bpa_clean.read_bytes()
        with patch("object_surface.create_bpa_mesh", side_effect=SurfaceReconstructionError("deliberate BPA rebuild failure")):
            with self.assertRaisesRegex(SurfaceReconstructionError, "deliberate BPA"):
                build_object_bpa(session, artifacts, bpa_config)
        self.assertEqual(bpa_clean.read_bytes(), bpa_previous)

    def test_bad_surface_input_is_rejected(self) -> None:
        empty = SurfaceInput(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), {}, "test")
        with self.assertRaisesRegex(SurfaceReconstructionError, "at least 20"):
            prepare_surface_cloud(empty, self._normal_config())
        nonfinite = SurfaceInput(np.full((24, 3), np.nan, dtype=np.float32), np.zeros((24, 3), dtype=np.float32), np.zeros((24, 3), dtype=np.uint8), {}, "test")
        with self.assertRaisesRegex(SurfaceReconstructionError, "finite"):
            prepare_surface_cloud(nonfinite, self._normal_config())


if __name__ == "__main__":
    unittest.main()
