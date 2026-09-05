"""TSDF preparation and mesh extraction for Phase 2C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from frame_io import FrameData
from geometry import sample_rgb, scale_intrinsics


CV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])


class TSDFValidationError(ValueError):
    """Raised when TSDF parameters or generated geometry are invalid."""


@dataclass(frozen=True)
class TSDFPolicy:
    voxel_length: float = 0.005
    sdf_trunc: float = 0.025
    min_confidence: int = 1
    min_depth: float | None = None
    max_depth: float | None = None

    def validate(self) -> None:
        if self.voxel_length <= 0:
            raise TSDFValidationError("voxel_length must be positive")
        if self.sdf_trunc <= self.voxel_length:
            raise TSDFValidationError("sdf_trunc must be greater than voxel_length")
        if self.min_confidence not in (0, 1, 2):
            raise TSDFValidationError("min_confidence must be 0, 1, or 2")
        if self.min_depth is not None and self.min_depth <= 0:
            raise TSDFValidationError("min_depth must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise TSDFValidationError("max_depth must be positive")
        if self.min_depth is not None and self.max_depth is not None and self.min_depth > self.max_depth:
            raise TSDFValidationError("min_depth must not exceed max_depth")


@dataclass(frozen=True)
class PreparedTSDFFrame:
    rgbd: o3d.geometry.RGBDImage
    intrinsic: o3d.camera.PinholeCameraIntrinsic
    extrinsic: np.ndarray
    valid_samples: int
    total_samples: int
    min_depth: float | None
    median_depth: float | None
    max_depth: float | None


def open3d_extrinsic_from_arkit_pose(world_from_arkit: np.ndarray) -> np.ndarray:
    """Return Open3D's world-to-CV-camera extrinsic for an ARKit pose.

    Open3D pinhole coordinates use +X right, +Y down, +Z forward. ARKit uses
    +X right, +Y up, -Z forward, so world_from_cv is the stored pose followed
    by CV_TO_ARKIT. Open3D integration expects the inverse, world_from_cv^-1.
    """

    pose = np.asarray(world_from_arkit, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise TSDFValidationError("world_from_arkit must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4):
        raise TSDFValidationError("world_from_arkit final row must be approximately [0,0,0,1]")
    world_from_cv = pose @ CV_TO_ARKIT
    extrinsic = np.linalg.inv(world_from_cv)
    if not np.all(np.isfinite(extrinsic)):
        raise TSDFValidationError("Open3D extrinsic contains non-finite values")
    return extrinsic


def _depth_mask(
    frame: FrameData,
    policy: TSDFPolicy,
    additional_mask: np.ndarray | None = None,
) -> np.ndarray:
    depth = frame.depth
    valid = np.isfinite(depth) & (depth > 0) & (frame.confidence >= policy.min_confidence)
    if policy.min_depth is not None:
        valid &= depth >= policy.min_depth
    if policy.max_depth is not None:
        valid &= depth <= policy.max_depth
    if additional_mask is not None:
        mask = np.asarray(additional_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise TSDFValidationError(
                f"additional depth mask shape {mask.shape} does not match depth {depth.shape}"
            )
        valid &= mask
    return valid


def prepare_tsdf_frame(
    frame: FrameData,
    policy: TSDFPolicy,
    additional_depth_mask: np.ndarray | None = None,
    world_from_arkit: np.ndarray | None = None,
) -> PreparedTSDFFrame:
    """Create depth-sized RGB-D data and the correctly directed extrinsic."""

    policy.validate()
    valid = _depth_mask(frame, policy, additional_depth_mask)
    depth = np.where(valid, frame.depth, 0.0).astype(np.float32, copy=False)
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.int32)
    pixels = np.column_stack((u.reshape(-1), v.reshape(-1)))
    colors = (sample_rgb(frame, pixels).reshape(height, width, 3) * 255.0).round().astype(np.uint8)
    color_image = o3d.geometry.Image(np.ascontiguousarray(colors))
    depth_image = o3d.geometry.Image(np.ascontiguousarray(depth))
    depth_trunc = policy.max_depth if policy.max_depth is not None else 1000.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    scaled = scale_intrinsics(frame)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, scaled.fx, scaled.fy, scaled.cx, scaled.cy)
    values = frame.depth[valid]
    return PreparedTSDFFrame(
        rgbd=rgbd,
        intrinsic=intrinsic,
        extrinsic=open3d_extrinsic_from_arkit_pose(
            frame.world_from_camera if world_from_arkit is None else world_from_arkit
        ),
        valid_samples=int(np.count_nonzero(valid)),
        total_samples=int(depth.size),
        min_depth=float(np.min(values)) if len(values) else None,
        median_depth=float(np.median(values)) if len(values) else None,
        max_depth=float(np.max(values)) if len(values) else None,
    )


@dataclass(frozen=True)
class MeshMetrics:
    vertices: int
    triangles: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    surface_area: float
    edge_manifold: bool
    vertex_manifold: bool
    self_intersecting: bool | None
    watertight: bool
    orientable: bool


def validate_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise TSDFValidationError("TSDF mesh is empty")
    if not np.all(np.isfinite(vertices)):
        raise TSDFValidationError("TSDF mesh vertices contain non-finite values")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise TSDFValidationError("TSDF mesh triangles are invalid")
    if len(mesh.vertex_colors) and not np.all(np.isfinite(np.asarray(mesh.vertex_colors))):
        raise TSDFValidationError("TSDF mesh colors contain non-finite values")


def mesh_metrics(mesh: o3d.geometry.TriangleMesh) -> MeshMetrics:
    validate_mesh(mesh)
    bbox = mesh.get_axis_aligned_bounding_box()
    return MeshMetrics(
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        bbox_min=np.asarray(bbox.min_bound),
        bbox_max=np.asarray(bbox.max_bound),
        surface_area=float(mesh.get_surface_area()),
        edge_manifold=mesh.is_edge_manifold(),
        vertex_manifold=mesh.is_vertex_manifold(),
        # Open3D's self-intersection test is quadratic and can stall on a
        # production-sized mesh; leave it unavailable in the default report.
        self_intersecting=None,
        watertight=mesh.is_watertight(),
        orientable=mesh.is_orientable(),
    )


def conservative_clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove only duplicate/degenerate/unreferenced topology."""

    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_duplicated_vertices()
    cleaned.remove_duplicated_triangles()
    cleaned.remove_degenerate_triangles()
    cleaned.remove_unreferenced_vertices()
    cleaned.compute_vertex_normals()
    validate_mesh(cleaned)
    return cleaned


def conservative_clean_mesh_components(
    mesh: o3d.geometry.TriangleMesh,
    minimum_component_triangles: int,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, object]]:
    """Apply topology cleanup and remove only components below a configured size."""

    if minimum_component_triangles < 1:
        raise TSDFValidationError("minimum_component_triangles must be positive")
    cleaned = conservative_clean_mesh(mesh)
    labels, counts, _ = cleaned.cluster_connected_triangles()
    component_sizes = np.asarray(counts, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    remove = component_sizes[labels_array] < minimum_component_triangles
    removed_triangles = int(np.count_nonzero(remove))
    removed_components = int(np.count_nonzero(component_sizes < minimum_component_triangles))
    if removed_triangles:
        cleaned.remove_triangles_by_mask(remove)
        cleaned.remove_unreferenced_vertices()
        cleaned.compute_vertex_normals()
    validate_mesh(cleaned)
    return cleaned, {
        "component_count": int(len(component_sizes)),
        "component_triangle_counts": component_sizes.tolist(),
        "removed_components": removed_components,
        "removed_triangles": removed_triangles,
    }


def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> int:
    validate_mesh(mesh)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write mesh: {path}")
    loaded = o3d.io.read_triangle_mesh(str(path))
    validate_mesh(loaded)
    if len(loaded.vertices) != len(mesh.vertices) or len(loaded.triangles) != len(mesh.triangles):
        raise OSError(f"Mesh readback mismatch for {path}")
    return len(loaded.triangles)


def write_point_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> int:
    points = np.asarray(cloud.points)
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise TSDFValidationError("TSDF point cloud is empty or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write point cloud: {path}")
    loaded = o3d.io.read_point_cloud(str(path))
    loaded_points = np.asarray(loaded.points)
    if len(loaded_points) != len(points) or not np.all(np.isfinite(loaded_points)):
        raise OSError(f"Point cloud readback mismatch for {path}")
    return len(loaded_points)
