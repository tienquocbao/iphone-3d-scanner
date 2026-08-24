"""Fuse a completed iPhone scan session into world-space point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from frame_io import FrameData, FrameValidationError, load_frame
from geometry import depth_to_world_points, sample_rgb


class SessionValidationError(ValueError):
    """Raised when a session cannot be safely fused."""


@dataclass(frozen=True)
class TrajectoryReport:
    frame_count: int
    start_position: np.ndarray
    end_position: np.ndarray
    path_length_meters: float
    max_step_meters: float
    median_step_meters: float
    max_rotation_degrees: float


@dataclass(frozen=True)
class FusionResult:
    frame_dirs: tuple[Path, ...]
    raw_cloud: o3d.geometry.PointCloud
    voxel_cloud: o3d.geometry.PointCloud
    clean_cloud: o3d.geometry.PointCloud | None
    trajectory: TrajectoryReport
    points_before_voxel: int
    frame_stats: tuple[dict[str, int | float], ...]


def session_frame_dirs(session_dir: Path, every_n: int = 1, max_frames: int | None = None) -> tuple[Path, ...]:
    """Validate session-level ordering and return the deterministic frame selection."""

    session_dir = Path(session_dir)
    try:
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionValidationError(f"Cannot read session.json: {exc}") from exc
    if metadata.get("schema_version") != 1:
        raise SessionValidationError("Unsupported session schema_version; expected 1")
    if metadata.get("status") != "completed":
        raise SessionValidationError(f"Session status must be completed, got {metadata.get('status')!r}")
    if every_n < 1:
        raise ValueError("every_n must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        raise SessionValidationError(f"Missing frames directory: {frames_dir}")
    directories = sorted((path for path in frames_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
    expected_names = [f"{index:06d}" for index in range(len(directories))]
    if [path.name for path in directories] != expected_names:
        raise SessionValidationError("Frame directories must be sequential six-digit indexes starting at 000000")
    expected_count = metadata.get("frame_count")
    if not isinstance(expected_count, int) or expected_count != len(directories):
        raise SessionValidationError(
            f"session.json frame_count {expected_count!r} does not match {len(directories)} frame directories"
        )
    selected = directories[::every_n]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise SessionValidationError("Session contains no frames after selection")
    return tuple(selected)


def trajectory_report(frames: list[FrameData]) -> TrajectoryReport:
    positions = np.asarray([frame.world_from_camera[:3, 3] for frame in frames], dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        raise SessionValidationError("Trajectory contains non-finite camera positions")
    if len(positions) == 1:
        steps = np.empty(0, dtype=np.float64)
        rotations = np.empty(0, dtype=np.float64)
    else:
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        rotations = []
        for previous, current in zip(frames, frames[1:]):
            previous_rotation = previous.world_from_camera[:3, :3]
            current_rotation = current.world_from_camera[:3, :3]
            relative = previous_rotation.T @ current_rotation
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            rotations.append(np.degrees(np.arccos(cosine)))
        rotations = np.asarray(rotations, dtype=np.float64)
    return TrajectoryReport(
        frame_count=len(frames),
        start_position=positions[0],
        end_position=positions[-1],
        path_length_meters=float(np.sum(steps)),
        max_step_meters=float(np.max(steps)) if len(steps) else 0.0,
        median_step_meters=float(np.median(steps)) if len(steps) else 0.0,
        max_rotation_degrees=float(np.max(rotations)) if len(rotations) else 0.0,
    )


def write_trajectory_csv(path: Path, frames: list[FrameData]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame_index", "timestamp_seconds", "x_m", "y_m", "z_m"))
        for frame in frames:
            metadata = frame.metadata
            writer.writerow(
                (
                    metadata.get("frame_index", frame.frame_dir.name),
                    metadata.get("timestamp_seconds", ""),
                    *[f"{value:.9f}" for value in frame.world_from_camera[:3, 3]],
                )
            )


def _cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    if len(points) == 0:
        raise SessionValidationError("Fusion produced zero valid points")
    if points.shape != (len(colors), 3) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(colors)):
        raise SessionValidationError("Fusion produced invalid point or color arrays")
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    result.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return result


def fuse_loaded_frames(
    frames: list[FrameData],
    min_confidence: int = 1,
    min_depth: float | None = None,
    max_depth: float | None = None,
    voxel_size: float = 0.005,
    remove_outliers: bool = False,
    outlier_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
) -> FusionResult:
    if not frames:
        raise SessionValidationError("No frames selected for fusion")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if remove_outliers and outlier_neighbors < 1:
        raise ValueError("outlier_neighbors must be positive")
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    stats: list[dict[str, int | float]] = []
    for frame in frames:
        points, pixels, frame_stat = depth_to_world_points(
            frame, min_confidence=min_confidence, min_depth=min_depth, max_depth=max_depth
        )
        if len(points) == 0:
            raise SessionValidationError(f"Frame {frame.frame_dir.name} has no valid points after filtering")
        colors = sample_rgb(frame, pixels)
        all_points.append(points)
        all_colors.append(colors)
        stats.append(frame_stat)
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    raw = _cloud(points, colors)
    voxel = raw.voxel_down_sample(voxel_size)
    if len(voxel.points) == 0:
        raise SessionValidationError("Voxel downsampling produced zero points")
    clean = None
    if remove_outliers:
        neighbor_count = min(outlier_neighbors, len(voxel.points) - 1)
        if neighbor_count < 1:
            raise SessionValidationError("Not enough voxel points for outlier removal")
        clean, _ = voxel.remove_statistical_outlier(nb_neighbors=neighbor_count, std_ratio=outlier_std_ratio)
        if len(clean.points) == 0:
            raise SessionValidationError("Outlier removal produced zero points")
    return FusionResult(
        frame_dirs=tuple(frame.frame_dir for frame in frames),
        raw_cloud=raw,
        voxel_cloud=voxel,
        clean_cloud=clean,
        trajectory=trajectory_report(frames),
        points_before_voxel=len(raw.points),
        frame_stats=tuple(stats),
    )


def write_cloud(path: Path, cloud: o3d.geometry.PointCloud) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"Failed to write point cloud: {path}")
    check = o3d.io.read_point_cloud(str(path))
    if len(check.points) != len(cloud.points) or len(check.colors) != len(cloud.colors):
        raise OSError(f"PLY readback mismatch for {path}")
    return len(check.points)


def _bounds(cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(cloud.points)
    return points.min(axis=0), points.max(axis=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse a completed iPhone scan session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--show", action="store_true", help="Show voxel output, or clean output when enabled")
    parser.add_argument("--show-raw", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    try:
        frame_dirs = session_frame_dirs(args.session, args.every_n, args.max_frames)
        frames = [load_frame(path) for path in frame_dirs]
        result = fuse_loaded_frames(
            frames,
            min_confidence=args.min_confidence,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            voxel_size=args.voxel_size,
            remove_outliers=args.remove_outliers,
            outlier_neighbors=args.outlier_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
        )
        prefix = args.output_prefix or (args.session / "session_pointcloud")
        raw_path = prefix.parent / f"{prefix.name}_raw.ply"
        voxel_path = prefix.parent / f"{prefix.name}_voxel.ply"
        clean_path = prefix.parent / f"{prefix.name}_clean.ply"
        write_cloud(raw_path, result.raw_cloud)
        write_cloud(voxel_path, result.voxel_cloud)
        if result.clean_cloud is not None:
            write_cloud(clean_path, result.clean_cloud)
        trajectory_path = args.session / "camera_trajectory.csv"
        write_trajectory_csv(trajectory_path, frames)
        trajectory = result.trajectory
        print(f"Session: {args.session}")
        print(f"Frames fused: {len(frames)}")
        print(f"Raw points: {len(result.raw_cloud.points)}")
        print(f"Voxel points: {len(result.voxel_cloud.points)}")
        if result.clean_cloud is not None:
            print(f"Clean points: {len(result.clean_cloud.points)}")
        print(f"Start camera: {trajectory.start_position.tolist()}")
        print(f"End camera: {trajectory.end_position.tolist()}")
        print(f"Path length: {trajectory.path_length_meters:.6f} m")
        print(f"Step translation max/median: {trajectory.max_step_meters:.6f} / {trajectory.median_step_meters:.6f} m")
        print(f"Frame rotation max: {trajectory.max_rotation_degrees:.3f} deg")
        for label, path, cloud in (("Raw", raw_path, result.raw_cloud), ("Voxel", voxel_path, result.voxel_cloud)):
            low, high = _bounds(cloud)
            print(f"{label} PLY: {path} ({len(cloud.points)} points)")
            print(f"{label} bounds min/max: {low.tolist()} / {high.tolist()}")
        print(f"Trajectory CSV: {trajectory_path}")
        print(f"Elapsed: {time.perf_counter() - started:.3f} s")
        if args.show:
            display_cloud = result.clean_cloud if result.clean_cloud is not None else result.voxel_cloud
            o3d.visualization.draw_geometries([display_cloud], window_name="Session voxel point cloud")
        if args.show_raw:
            o3d.visualization.draw_geometries([result.raw_cloud], window_name="Session raw point cloud")
        return 0
    except (FrameValidationError, SessionValidationError, OSError, ValueError) as exc:
        print(f"Fusion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
