from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).parents[1]))
from registration import ObjectRegistrationConfig, compose_object_from_camera, register_pass, transform_points

def cloud(points):
    value=o3d.geometry.PointCloud(); value.points=o3d.utility.Vector3dVector(points); return value

class RegistrationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        base=rng.uniform([-0.12,-0.05,-0.08],[0.16,0.09,0.11],(500,3)); base=np.vstack((base, [[.22,.02,.01],[.18,.04,.09],[-.10,.08,-.05]]))
        self.target=cloud(base); angle=np.radians(24); self.transform=np.eye(4); self.transform[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]; self.transform[:3,3]=[.04,-.02,.03]
        self.source=cloud(transform_points(base,np.linalg.inv(self.transform)))
    def test_global_and_icp_recover_asymmetric_rigid_transform(self):
        result=register_pass(self.source,self.target,ObjectRegistrationConfig(registration_voxel_size=.01,max_icp_rmse=.02))
        self.assertTrue(result['accepted'],result); np.testing.assert_allclose(result['object_from_pass'],self.transform,atol=.025)
    def test_partial_overlap_and_noise_is_accepted(self):
        points=np.asarray(self.source.points); partial=cloud(points[:350]+np.random.default_rng(1).normal(0,.001,(350,3)))
        result=register_pass(partial,self.target,ObjectRegistrationConfig(registration_voxel_size=.01,max_icp_rmse=.025)); self.assertTrue(result['accepted'],result)
    def test_insufficient_overlap_is_rejected(self):
        far=cloud(np.random.default_rng(5).uniform(4,5,(200,3)))
        result=register_pass(far,self.target,ObjectRegistrationConfig(registration_voxel_size=.01)); self.assertFalse(result['accepted'])
    def test_object_from_pass_and_camera_composition(self):
        camera=np.eye(4); camera[:3,3]=[1,2,3]
        np.testing.assert_allclose(compose_object_from_camera(self.transform,camera),self.transform@camera)
        np.testing.assert_allclose(transform_points(np.array([[0,0,0]]),np.eye(4)),[[0,0,0]])
