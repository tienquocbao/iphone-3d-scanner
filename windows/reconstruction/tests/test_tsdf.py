from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parents[1]))

from tsdf import (
    CV_TO_ARKIT,
    TSDFPolicy,
    TSDFValidationError,
    conservative_clean_mesh,
    mesh_metrics,
    open3d_extrinsic_from_arkit_pose,
    write_mesh,
)
from frame_io import FrameData
from tsdf import prepare_tsdf_frame


class TSDFTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_rgbd(self, depth=1.0, width=48, height=36):
        depth_image = np.full((height, width), depth, dtype=np.float32)
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        color_image[:, :, 0] = 200
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 50, 50, width / 2 - 0.5, height / 2 - 0.5)
        return rgbd, intrinsic

    def make_frame(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(4, dtype=np.uint8)[None, :] * 50
        rgb[:, :, 1] = np.arange(4, dtype=np.uint8)[:, None] * 50
        depth = np.ones((2, 2), dtype=np.float32)
        confidence = np.array([[2, 0], [1, 2]], dtype=np.uint8)
        metadata = {"camera": {"image_width": 4, "image_height": 4}}
        return FrameData(Path(self.temp_dir.name), metadata, rgb, depth, confidence, np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]], dtype=np.float64), np.eye(4))

    def integrate_plane(self, poses):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02,
            sdf_trunc=0.08,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        rgbd, intrinsic = self.make_rgbd()
        for pose in poses:
            volume.integrate(rgbd, intrinsic, open3d_extrinsic_from_arkit_pose(pose))
        return volume

    def test_identity_extrinsic_matches_arkit_axis_conversion(self):
        pose = np.eye(4)
        np.testing.assert_allclose(open3d_extrinsic_from_arkit_pose(pose), CV_TO_ARKIT, atol=1e-12)
        world_from_cv = np.linalg.inv(open3d_extrinsic_from_arkit_pose(pose))
        point_world = world_from_cv @ [0.0, 0.0, 1.0, 1.0]
        np.testing.assert_allclose(point_world[:3], [0.0, 0.0, -1.0], atol=1e-12)

    def test_translated_camera_has_phase_2a_world_position(self):
        pose = np.eye(4)
        pose[0, 3] = 0.25
        world_from_cv = np.linalg.inv(open3d_extrinsic_from_arkit_pose(pose))
        np.testing.assert_allclose(world_from_cv @ [0.0, 0.0, 1.0, 1.0], [0.25, 0.0, -1.0, 1.0], atol=1e-12)

    def test_policy_rejects_invalid_parameters(self):
        for policy in (TSDFPolicy(0, 0.025), TSDFPolicy(0.005, 0.005), TSDFPolicy(0.005, 0.025, 3), TSDFPolicy(0.005, 0.025, 1, 2, 1)):
            with self.assertRaises(TSDFValidationError):
                policy.validate()

    def test_preparation_scales_intrinsics_and_masks_confidence(self):
        prepared = prepare_tsdf_frame(self.make_frame(), TSDFPolicy(min_confidence=2))
        self.assertEqual(np.asarray(prepared.rgbd.color).shape[:2], (2, 2))
        self.assertEqual((prepared.intrinsic.width, prepared.intrinsic.height), (2, 2))
        self.assertAlmostEqual(prepared.intrinsic.get_focal_length()[0], 2.0)
        self.assertEqual(prepared.valid_samples, 2)
        depth = np.asarray(prepared.rgbd.depth)
        self.assertEqual(float(depth[0, 1]), 0.0)
        self.assertEqual(float(depth[1, 0]), 0.0)

    def test_preparation_reuses_depth_to_rgb_color_mapping(self):
        prepared = prepare_tsdf_frame(self.make_frame(), TSDFPolicy(min_confidence=0))
        colors = np.asarray(prepared.rgbd.color)
        self.assertEqual(colors[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(colors[0, 1].tolist(), [100, 0, 0])
        self.assertEqual(colors[1, 0].tolist(), [0, 100, 0])

    def test_synthetic_plane_is_at_negative_world_z(self):
        mesh = self.integrate_plane([np.eye(4)]).extract_triangle_mesh()
        metrics = mesh_metrics(mesh)
        self.assertGreater(metrics.triangles, 0)
        mean_z = float(np.mean(np.asarray(mesh.vertices)[:, 2]))
        self.assertAlmostEqual(mean_z, -1.0, delta=0.04)

    def test_two_views_keep_one_coherent_plane(self):
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 0.15
        mesh = self.integrate_plane([first, second]).extract_triangle_mesh()
        vertices = np.asarray(mesh.vertices)
        self.assertGreater(len(mesh.triangles), 0)
        self.assertLess(float(np.std(vertices[:, 2])), 0.03)
        self.assertAlmostEqual(float(np.median(vertices[:, 2])), -1.0, delta=0.04)

    def test_mesh_cleanup_and_export_readback(self):
        raw = self.integrate_plane([np.eye(4)]).extract_triangle_mesh()
        raw.compute_vertex_normals()
        clean = conservative_clean_mesh(raw)
        path = Path(self.temp_dir.name) / "mesh.ply"
        write_mesh(path, clean)
        loaded = o3d.io.read_triangle_mesh(str(path))
        self.assertGreater(len(loaded.vertices), 0)
        self.assertGreater(len(loaded.triangles), 0)
        self.assertTrue(np.all(np.isfinite(np.asarray(loaded.vertices))))
