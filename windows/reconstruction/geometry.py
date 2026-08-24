"""Vectorized RGB-D geometry for the documented ARKit coordinate contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frame_io import FrameData


@dataclass(frozen=True)
class DepthIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def scale_intrinsics(frame: FrameData) -> DepthIntrinsics:
    camera = frame.metadata["camera"]
    rgb_width = float(camera["image_width"])
    rgb_height = float(camera["image_height"])
    depth_height, depth_width = frame.depth.shape
    rows = frame.intrinsics
    sx = depth_width / rgb_width
    sy = depth_height / rgb_height
    return DepthIntrinsics(
        fx=float(rows[0, 0] * sx),
        fy=float(rows[1, 1] * sy),
        cx=float(rows[0, 2] * sx),
        cy=float(rows[1, 2] * sy),
    )


def depth_to_world_points(
    frame: FrameData,
    min_confidence: int = 1,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Back-project depth pixels and return world XYZ plus source pixel indices."""

    if min_confidence not in (0, 1, 2):
        raise ValueError("min_confidence must be 0, 1, or 2")
    intrinsics = scale_intrinsics(frame)
    depth = frame.depth
    confidence = frame.confidence
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0) & (confidence >= min_confidence)
    if min_depth is not None:
        valid &= depth >= min_depth
    if max_depth is not None:
        valid &= depth <= max_depth
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32), {
            "total_samples": int(depth.size),
            "valid_samples": 0,
        }

    z_cv = depth[valid].astype(np.float64)
    x_cv = (u[valid] - intrinsics.cx) * z_cv / intrinsics.fx
    y_cv = (v[valid] - intrinsics.cy) * z_cv / intrinsics.fy

    # CV optical: +X right, +Y down, +Z forward.
    # ARKit camera: +X right, +Y up, -Z forward.
    points_arkit = np.column_stack((x_cv, -y_cv, -z_cv))
    homogeneous = np.column_stack((points_arkit, np.ones(len(points_arkit))))
    points_world = (homogeneous @ frame.world_from_camera.T)[:, :3]
    source_pixels = np.column_stack((u[valid], v[valid])).astype(np.int32)
    stats = {
        "total_samples": int(depth.size),
        "valid_samples": int(len(points_world)),
        "min_depth": float(np.min(z_cv)),
        "median_depth": float(np.median(z_cv)),
        "max_depth": float(np.max(z_cv)),
    }
    return points_world, source_pixels, stats


def sample_rgb(frame: FrameData, source_pixels: np.ndarray) -> np.ndarray:
    """Map depth pixel centers to native-orientation RGB pixels with nearest neighbor."""

    rgb_height, rgb_width = frame.rgb.shape[:2]
    depth_height, depth_width = frame.depth.shape
    u_depth = source_pixels[:, 0].astype(np.float64)
    v_depth = source_pixels[:, 1].astype(np.float64)
    u_rgb = np.rint(((u_depth + 0.5) * rgb_width / depth_width) - 0.5).astype(np.int64)
    v_rgb = np.rint(((v_depth + 0.5) * rgb_height / depth_height) - 0.5).astype(np.int64)
    u_rgb = np.clip(u_rgb, 0, rgb_width - 1)
    v_rgb = np.clip(v_rgb, 0, rgb_height - 1)
    colors = frame.rgb[v_rgb, u_rgb, :3].astype(np.float64) / 255.0
    return colors
