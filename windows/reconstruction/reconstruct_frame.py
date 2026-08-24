from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from frame_io import FrameValidationError, load_frame
from geometry import depth_to_world_points, sample_rgb, scale_intrinsics


def build_point_cloud(
    frame_dir: Path,
    min_confidence: int = 1,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> tuple[o3d.geometry.PointCloud, dict[str, int | float]]:
    frame = load_frame(frame_dir)
    points, source_pixels, stats = depth_to_world_points(
        frame,
        min_confidence=min_confidence,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    colors = sample_rgb(frame, source_pixels)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct one iPhone RGB-D frame")
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    try:
        frame = load_frame(args.frame_dir)
        cloud, stats = build_point_cloud(
            args.frame_dir,
            min_confidence=args.min_confidence,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
    except (FrameValidationError, OSError, ValueError) as exc:
        print(f"Reconstruction: FAIL\nReason: {exc}")
        return 1

    output = args.output or args.frame_dir / "pointcloud_world.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False):
        print(f"Reconstruction: FAIL\nReason: could not write {output}")
        return 1
    readback = o3d.io.read_point_cloud(str(output))
    if len(readback.points) != len(cloud.points):
        print("Reconstruction: FAIL\nReason: PLY readback point count mismatch")
        return 1

    print(f"Frame: {args.frame_dir.name}")
    print(f"Depth intrinsics: {scale_intrinsics(frame)}")
    print(f"Points: {len(cloud.points)}")
    if len(cloud.points):
        points = np.asarray(cloud.points)
        print(f"World min: {points.min(axis=0).tolist()}")
        print(f"World max: {points.max(axis=0).tolist()}")
    print(f"Depth stats: {stats}")
    print(f"PLY: {output}")
    print("PLY readback: PASS")
    if args.show:
        o3d.visualization.draw_geometries([cloud], window_name="iPhone 3D Scanner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
