from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parents[1]))

from foreground import ForegroundConfig
from nksr_backend import (
    NKSRConfig,
    NKSRExecutionError,
    NKSRUnavailableError,
    build_object_nksr,
    probe_nksr_backend,
)
from nksr_input import NKSRInputConfig, joint_voxel_aggregate, prepare_nksr_input
from nksr_runner import _select_attempts

FAKE_RUNNER = Path(__file__).with_name("fake_nksr_runner.py")


def _write_frame(path: Path, index: int, pose: np.ndarray) -> None:
    path.mkdir(parents=True)
    rgb = np.full((16, 16, 3), [20, 180, 30], dtype=np.uint8)
    rgb[4:12, 4:12] = [210, 30, 20]
    o3d.io.write_image(str(path / "rgb.jpg"), o3d.geometry.Image(rgb), quality=100)
    depth = np.full((16, 16), 1.4, dtype="<f4"); depth[4:12, 4:12] = 1.0
    depth.tofile(path / "depth.f32")
    np.full((16, 16), 2, dtype=np.uint8).tofile(path / "confidence.u8")
    (path / "frame.json").write_text(json.dumps({
        "schema_version":1,"frame_index":index,"timestamp_seconds":index,
        "rgb":{"file":"rgb.jpg","width":16,"height":16},
        "depth":{"file":"depth.f32","width":16,"height":16,"dtype":"float32","endianness":"little","unit":"meters"},
        "confidence":{"file":"confidence.u8","width":16,"height":16,"dtype":"uint8"},
        "camera":{"image_width":16,"image_height":16,"intrinsics_rows":[[16,0,7.5],[0,16,7.5],[0,0,1]],"transform_semantics":"world_from_camera","transform_rows":pose.tolist(),"coordinate_system":"ARKit","units":"meters","forward_axis":"-Z"}
    }), encoding="utf-8")


def _write_bounds(path: Path) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[-.5,-.5,-1.2],[.5,.5,-.8]])
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud)


class NKSRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = self.root / "session_object"
        self.artifact = self.root / "artifacts"

    def tearDown(self):
        self.temp.cleanup()

    def _make_session(self, multipass: bool = False) -> None:
        poses = [np.eye(4)]
        passes = [{"id":0,"start_frame":0,"end_frame":0}]
        if multipass:
            second = np.eye(4); second[0,3] = -0.2
            poses.append(second); passes.append({"id":1,"start_frame":1,"end_frame":1})
        for index, pose in enumerate(poses): _write_frame(self.session/"frames"/f"{index:06d}",index,pose)
        (self.session/"session.json").write_text(json.dumps({"schema_version":1,"session_id":"object","status":"completed","frame_count":len(poses),"scan_mode":"object","passes":passes}),encoding="utf-8")
        if multipass:
            transform=np.eye(4); transform[0,3]=0.3
            path=self.artifact/"object"/"registration"/"pass_transforms.json";path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"canonical_pass":0,"passes":[{"id":0,"registration_status":"reference","object_from_pass":np.eye(4).tolist()},{"id":1,"registration_status":"accepted","object_from_pass":transform.tolist()}]}),encoding="utf-8")
            _write_bounds(self.artifact/"object"/"object_registered_clean.ply")
        else: _write_bounds(self.artifact/"object"/"object_clean.ply")

    def _input_config(self):
        return NKSRInputConfig(input_voxel_size_m=.001,outlier_neighbors=2,max_input_points=10000,foreground=ForegroundConfig(morphology_kernel=1,minimum_component_pixels=1))

    def test_joint_voxel_aggregation_preserves_sensor_provenance(self):
        xyz=np.array([[.001,0,0],[.002,0,0],[.02,0,0]])
        sensor=np.array([[0,0,0],[.2,0,0],[1,0,0]])
        color=np.array([[0,0,0],[100,0,0],[200,0,0]])
        points,sensors,colors=joint_voxel_aggregate(xyz,sensor,color,.01)
        self.assertEqual(len(points),2); np.testing.assert_allclose(sensors[0],[.1,0,0]); np.testing.assert_allclose(colors[0],[50,0,0])

    def test_prepared_xyz_and_sensor_share_metric_object_frame(self):
        self._make_session(multipass=True)
        prepared=prepare_nksr_input(self.session,self.artifact,self._input_config())
        self.assertEqual(prepared.xyz.shape,prepared.sensor.shape)
        self.assertEqual(prepared.xyz.dtype,np.float32)
        self.assertGreater(prepared.summary["unique_sensor_origins"],1)
        origins=np.unique(np.round(prepared.sensor[:,0],4)); self.assertIn(0.0,origins); self.assertIn(0.1,origins)
        self.assertLess(float(np.max(prepared.summary["metric_extent_m"])),1.0)

    def test_auto_policy_prefers_chunk_on_four_gib_gpu_and_has_bounded_oom_fallback(self):
        config={"execution_mode":"auto","device":"auto","low_vram_threshold_bytes":6*1024**3,"full_cuda_max_points":100000,"cpu_fallback":False}
        self.assertEqual(_select_attempts(config,{"cuda_available":True,"gpu_vram_bytes":4*1024**3},25000),[("chunk","cuda")])
        probe={"cuda_available":True,"gpu_vram_bytes":12*1024**3}
        self.assertEqual(_select_attempts(config,probe,25000),[("full","cuda"),("chunk","cuda")])

    def test_missing_backend_is_explicit_and_does_not_touch_tsdf(self):
        result=probe_nksr_backend(self.root/"missing-python.exe")
        self.assertFalse(result["available"])
        with self.assertRaisesRegex(NKSRUnavailableError,"unavailable"):
            build_object_nksr(self.session,self.artifact,python_executable=self.root/"missing-python.exe")

    def test_fake_subprocess_adapter_places_artifacts_and_preserves_attempts(self):
        self._make_session()
        config=NKSRConfig(timeout_seconds=10,mesh_min_component_triangles=1,input=self._input_config())
        report=build_object_nksr(self.session,self.artifact,config,python_executable=Path(sys.executable),runner_path=FAKE_RUNNER)
        output=self.artifact/"object"/"reconstruction"/"nksr"
        self.assertTrue((output/"object_nksr_raw.ply").is_file());self.assertTrue((output/"object_nksr_clean.ply").is_file())
        self.assertEqual(report["attempts"][0]["status"],"oom");self.assertEqual(report["attempts"][1]["status"],"success")
        self.assertEqual(report["coordinate_frame"],"object");self.assertFalse(any(path.name.startswith('.job-') for path in output.iterdir()))

    def test_timeout_kills_adapter_and_removes_partial_outputs(self):
        self._make_session()
        config=NKSRConfig(timeout_seconds=1,mesh_min_component_triangles=1,input=self._input_config())
        with patch.dict(os.environ, {"IPHONE3D_FAKE_NKSR_MODE":"timeout"}):
            with self.assertRaisesRegex(NKSRExecutionError,"timed out"):
                build_object_nksr(self.session,self.artifact,config,python_executable=Path(sys.executable),runner_path=FAKE_RUNNER)
        output=self.artifact/"object"/"reconstruction"/"nksr"
        self.assertFalse((output/"object_nksr_raw.ply").exists());self.assertFalse((output/"object_nksr_clean.ply").exists())


if __name__ == "__main__": unittest.main()
