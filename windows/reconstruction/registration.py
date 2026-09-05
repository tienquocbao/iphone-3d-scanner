"""Classical multi-pass object registration into pass 0's canonical coordinates."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import open3d as o3d

@dataclass(frozen=True)
class ObjectRegistrationConfig:
    registration_voxel_size: float = 0.01
    normal_radius_multiplier: float = 2.0
    feature_radius_multiplier: float = 5.0
    ransac_distance_multiplier: float = 1.5
    icp_distance_multiplier: float = 0.8
    min_global_fitness: float = 0.08
    min_icp_fitness: float = 0.20
    max_icp_rmse: float = 0.03
    max_iterations: int = 100000

    def validate(self):
        if self.registration_voxel_size <= 0 or self.max_iterations < 1:
            raise ValueError("invalid registration policy")

def compose_object_from_camera(object_from_pass: np.ndarray, pass_world_from_camera: np.ndarray) -> np.ndarray:
    """Compose row-stored homogeneous transforms: object_from_pass @ pass_world_from_camera."""
    return np.asarray(object_from_pass, dtype=np.float64) @ np.asarray(pass_world_from_camera, dtype=np.float64)

def transform_points(points: np.ndarray, object_from_pass: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64); transform = np.asarray(object_from_pass, dtype=np.float64)
    return (np.c_[points, np.ones(len(points))] @ transform.T)[:, :3]

def _prepare(cloud: o3d.geometry.PointCloud, config: ObjectRegistrationConfig):
    down = cloud.voxel_down_sample(config.registration_voxel_size)
    if len(down.points) < 20: raise ValueError("registration cloud has fewer than 20 points")
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.registration_voxel_size * config.normal_radius_multiplier, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=config.registration_voxel_size * config.feature_radius_multiplier, max_nn=100))
    return down, feature

def _angle(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3,:3]) - 1) / 2, -1, 1))))

def register_pass(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, config: ObjectRegistrationConfig = ObjectRegistrationConfig(), initial_transform: np.ndarray | None = None) -> dict[str, object]:
    """Return object_from_pass (source-to-target), diagnostics, and accepted status."""
    config.validate(); o3d.utility.random.seed(7)
    source_down, source_feature = _prepare(source, config); target_down, target_feature = _prepare(target, config)
    distance = config.registration_voxel_size * config.ransac_distance_multiplier
    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_feature, target_feature, True, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(config.max_iterations, 0.999),
    )
    seed = np.asarray(initial_transform, dtype=np.float64) if initial_transform is not None else global_result.transformation
    source_fine = o3d.geometry.PointCloud(source); target_fine = o3d.geometry.PointCloud(target)
    source_fine.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.registration_voxel_size * config.normal_radius_multiplier, max_nn=30))
    target_fine.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.registration_voxel_size * config.normal_radius_multiplier, max_nn=30))
    fine = o3d.pipelines.registration.registration_icp(source_fine, target_fine, config.registration_voxel_size * config.icp_distance_multiplier, seed, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    accepted = bool(global_result.fitness >= config.min_global_fitness and fine.fitness >= config.min_icp_fitness and fine.inlier_rmse <= config.max_icp_rmse)
    return {"object_from_pass": fine.transformation.tolist(), "global_fitness": float(global_result.fitness), "global_rmse": float(global_result.inlier_rmse), "icp_fitness": float(fine.fitness), "icp_rmse": float(fine.inlier_rmse), "translation_meters": float(np.linalg.norm(fine.transformation[:3,3])), "rotation_degrees": _angle(fine.transformation), "accepted": accepted, "policy": asdict(config)}
