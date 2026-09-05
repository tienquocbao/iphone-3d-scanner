"""Prepare provenance-preserving canonical object observations.

The historical public names remain for Gate C2 compatibility.  This module
does not import NKSR or PyTorch and is also the data source for native surface
backends through :mod:`surface_input`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import open3d as o3d

from foreground import ForegroundConfig, calibrate_green_background, foreground_mask
from frame_io import load_frame
from fuse_session import SessionValidationError, session_frame_dirs
from geometry import depth_to_world_points, sample_rgb
from object_tsdf import (
    load_object_from_pass_transforms,
    object_bounds,
    object_depth_mask,
    object_session_passes,
)
from registration import compose_object_from_camera


@dataclass(frozen=True)
class NKSRInputConfig:
    min_confidence: int = 1
    min_depth_m: float | None = 0.10
    max_depth_m: float | None = 3.0
    object_bounds_margin_m: float = 0.04
    input_voxel_size_m: float = 0.004
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    max_input_points: int = 250_000
    foreground: ForegroundConfig = ForegroundConfig()

    def validate(self) -> None:
        if self.min_confidence not in (0, 1, 2):
            raise ValueError("min_confidence must be 0, 1, or 2")
        if self.input_voxel_size_m <= 0 or self.object_bounds_margin_m < 0:
            raise ValueError("invalid NKSR input dimensions")
        if self.outlier_neighbors < 1 or self.outlier_std_ratio <= 0 or self.max_input_points < 1:
            raise ValueError("invalid NKSR input filtering configuration")
        self.foreground.validate()


@dataclass(frozen=True)
class NKSRInput:
    xyz: np.ndarray
    sensor: np.ndarray
    color: np.ndarray
    summary: dict[str, object]
    pass_transforms_sha256: str


def joint_voxel_aggregate(
    xyz: np.ndarray,
    sensor: np.ndarray,
    color: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average xyz, sensor origin, and color together in deterministic voxels."""

    xyz = np.asarray(xyz, dtype=np.float64)
    sensor = np.asarray(sensor, dtype=np.float64)
    color = np.asarray(color, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or sensor.shape != xyz.shape or color.shape != xyz.shape:
        raise ValueError("xyz, sensor, and color must be aligned Nx3 arrays")
    if not len(xyz) or voxel_size <= 0 or not all(np.all(np.isfinite(value)) for value in (xyz, sensor, color)):
        raise ValueError("joint voxel input must be finite, non-empty, and use a positive voxel size")
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(order)]).astype(np.float64)[:, None]
    aggregated_xyz = np.add.reduceat(xyz[order], starts, axis=0) / counts
    aggregated_sensor = np.add.reduceat(sensor[order], starts, axis=0) / counts
    aggregated_color = np.add.reduceat(color[order], starts, axis=0) / counts
    return aggregated_xyz, aggregated_sensor, aggregated_color


def _filter_outliers(
    xyz: np.ndarray,
    sensor: np.ndarray,
    color: np.ndarray,
    config: NKSRInputConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xyz) <= config.outlier_neighbors:
        return xyz, sensor, color
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    _, indexes = cloud.remove_statistical_outlier(
        nb_neighbors=min(config.outlier_neighbors, len(xyz) - 1),
        std_ratio=config.outlier_std_ratio,
    )
    selected = np.asarray(indexes, dtype=np.int64)
    if not len(selected):
        return xyz, sensor, color
    return xyz[selected], sensor[selected], color[selected]


def prepare_nksr_input(
    session_dir: Path,
    artifact_dir: Path,
    config: NKSRInputConfig = NKSRInputConfig(),
    progress=None,
) -> NKSRInput:
    """Build metric object-frame xyz/sensor/color arrays from original RGB-D frames."""

    config.validate()
    started = time.perf_counter()
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    try:
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionValidationError(f"Cannot read session.json: {exc}") from exc
    frame_dirs = session_frame_dirs(session_dir)
    passes = object_session_passes(metadata, len(frame_dirs))
    transforms, transform_sha = load_object_from_pass_transforms(
        passes, artifact_dir / "object" / "registration" / "pass_transforms.json"
    )
    bounds_min, bounds_max, bounds_source = object_bounds(
        artifact_dir, len(passes), config.object_bounds_margin_m
    )
    point_chunks: list[np.ndarray] = []
    sensor_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    rejected: list[dict[str, object]] = []
    pass_reports: list[dict[str, object]] = []

    for pass_number, pass_info in enumerate(passes):
        pass_id = pass_info["id"]
        selected = frame_dirs[pass_info["start_frame"] : pass_info["end_frame"] + 1]
        model = calibrate_green_background(
            (load_frame(directory).rgb for directory in selected), config.foreground
        )
        pass_points = 0
        for local_index, frame_dir in enumerate(selected):
            try:
                frame = load_frame(frame_dir)
                object_from_camera = compose_object_from_camera(
                    transforms[pass_id], frame.world_from_camera
                )
                rgb_mask = foreground_mask(frame.rgb, model, config.foreground)
                depth_mask, _, _ = object_depth_mask(
                    frame,
                    rgb_mask,
                    object_from_camera,
                    bounds_min,
                    bounds_max,
                    config,  # structural compatibility with ObjectTSDFConfig fields
                )
                object_frame = replace(frame, world_from_camera=object_from_camera)
                points, pixels, _ = depth_to_world_points(
                    object_frame,
                    config.min_confidence,
                    config.min_depth_m,
                    config.max_depth_m,
                )
                keep = depth_mask[pixels[:, 1], pixels[:, 0]] if len(pixels) else np.empty(0, dtype=bool)
                points, pixels = points[keep], pixels[keep]
                if not len(points):
                    raise ValueError("no usable foreground points")
                colors = sample_rgb(frame, pixels)
                sensor_origin = object_from_camera[:3, 3]
                point_chunks.append(points)
                sensor_chunks.append(np.broadcast_to(sensor_origin, points.shape).copy())
                color_chunks.append(colors)
                pass_points += len(points)
            except (OSError, ValueError) as exc:
                rejected.append({"frame": frame_dir.name, "pass_id": pass_id, "reason": str(exc)[:256]})
            if progress:
                completed = pass_info["start_frame"] + local_index + 1
                progress(10 + int(45 * completed / len(frame_dirs)), f"PREPARING PASS {pass_id + 1}")
        pass_reports.append(
            {
                "id": pass_id,
                "frames": len(selected),
                "points_before_voxel": pass_points,
                "background_model": model.to_dict(),
            }
        )
    if not point_chunks:
        raise SessionValidationError("NKSR input contains no usable object observations")
    xyz = np.concatenate(point_chunks)
    sensor = np.concatenate(sensor_chunks)
    color = np.concatenate(color_chunks)
    before_filter = len(xyz)
    xyz, sensor, color = joint_voxel_aggregate(
        xyz, sensor, color, config.input_voxel_size_m
    )
    after_voxel = len(xyz)
    xyz, sensor, color = _filter_outliers(xyz, sensor, color, config)
    after_outlier = len(xyz)
    limited = False
    if len(xyz) > config.max_input_points:
        indexes = np.linspace(0, len(xyz) - 1, config.max_input_points, dtype=np.int64)
        xyz, sensor, color = xyz[indexes], sensor[indexes], color[indexes]
        limited = True
    if not len(xyz) or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(sensor)):
        raise SessionValidationError("NKSR input is empty or non-finite after filtering")
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    if float(np.max(extent)) <= 1e-5:
        raise SessionValidationError("NKSR input spatial extent is too small")
    summary = {
        "coordinate_frame": "object",
        "point_count_before_filter": before_filter,
        "point_count_after_voxel": after_voxel,
        "point_count_after_outlier_filter": after_outlier,
        "point_count_after_filter": len(xyz),
        "point_limit_applied": limited,
        "voxel_size_m": config.input_voxel_size_m,
        "bounds_min_m": xyz.min(axis=0).tolist(),
        "bounds_max_m": xyz.max(axis=0).tolist(),
        "metric_extent_m": extent.tolist(),
        "pass_count": len(passes),
        "frame_count": len(frame_dirs),
        "rejected_frames": rejected,
        "sensor_origin_count": len(sensor),
        "unique_sensor_origins": int(len(np.unique(sensor, axis=0))),
        "color_available": True,
        "input_scale_to_nksr": 1.0,
        "object_bounds_source": bounds_source,
        "passes": pass_reports,
        "preparation_seconds": time.perf_counter() - started,
        "config": {**asdict(config), "foreground": asdict(config.foreground)},
    }
    return NKSRInput(
        xyz.astype(np.float32),
        sensor.astype(np.float32),
        np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8),
        summary,
        transform_sha,
    )


def write_nksr_input(path: Path, prepared: NKSRInput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz=prepared.xyz, sensor=prepared.sensor, color=prepared.color)
