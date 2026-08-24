"""Integrate one completed scan session with Open3D's scalable TSDF volume."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from frame_io import FrameValidationError, load_frame
from fuse_session import SessionValidationError, session_frame_dirs
from tsdf import (
    TSDFPolicy,
    TSDFValidationError,
    conservative_clean_mesh,
    mesh_metrics,
    prepare_tsdf_frame,
    validate_mesh,
    write_mesh,
    write_point_cloud,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a scan session with a scalable TSDF")
    parser.add_argument("session", type=Path)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--voxel-length", type=float, default=0.005)
    parser.add_argument("--sdf-trunc", type=float, default=0.025)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--show", action="store_true", help="Show the cleaned mesh")
    parser.add_argument("--show-tsdf-cloud", action="store_true")
    parser.add_argument("--show-raw-mesh", action="store_true")
    return parser


def _print_metrics(label: str, mesh: o3d.geometry.TriangleMesh) -> None:
    metrics = mesh_metrics(mesh)
    print(f"{label}")
    print(f"  vertices: {metrics.vertices}")
    print(f"  triangles: {metrics.triangles}")
    print(f"  bbox min: {metrics.bbox_min.tolist()}")
    print(f"  bbox max: {metrics.bbox_max.tolist()}")
    print(f"  bbox extent: {(metrics.bbox_max - metrics.bbox_min).tolist()}")
    print(f"  surface area: {metrics.surface_area:.6f} m^2")
    print(f"  edge/vertex manifold: {metrics.edge_manifold}/{metrics.vertex_manifold}")
    print(f"  self-intersecting: {metrics.self_intersecting if metrics.self_intersecting is not None else 'not checked (expensive)'}")
    print(f"  watertight/orientable: {metrics.watertight}/{metrics.orientable}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    try:
        policy = TSDFPolicy(args.voxel_length, args.sdf_trunc, args.min_confidence, args.min_depth, args.max_depth)
        policy.validate()
        frame_dirs = session_frame_dirs(args.session, args.every_n, args.max_frames)
        prefix = args.output_prefix or (args.session / "session")
        tsdf_cloud_path = prefix.parent / f"{prefix.name}_tsdf_pointcloud.ply"
        raw_mesh_path = prefix.parent / f"{prefix.name}_mesh_raw.ply"
        clean_mesh_path = prefix.parent / f"{prefix.name}_mesh_clean.ply"
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=policy.voxel_length,
            sdf_trunc=policy.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        preparation_started = time.perf_counter()
        prepared = []
        for frame_dir in frame_dirs:
            frame = load_frame(frame_dir)
            item = prepare_tsdf_frame(frame, policy)
            if item.valid_samples == 0:
                raise TSDFValidationError(f"Frame {frame_dir.name} has no valid depth after filtering")
            prepared.append((frame_dir, item))
        preparation_seconds = time.perf_counter() - preparation_started
        integration_started = time.perf_counter()
        for frame_dir, item in prepared:
            volume.integrate(item.rgbd, item.intrinsic, item.extrinsic)
            print(f"  {frame_dir.name}: valid depth {item.valid_samples}/{item.total_samples}")
        integration_seconds = time.perf_counter() - integration_started
        extraction_started = time.perf_counter()
        tsdf_cloud = volume.extract_point_cloud()
        raw_mesh = volume.extract_triangle_mesh()
        raw_mesh.compute_vertex_normals()
        validate_mesh(raw_mesh)
        clean_mesh = conservative_clean_mesh(raw_mesh)
        extraction_seconds = time.perf_counter() - extraction_started
        write_point_cloud(tsdf_cloud_path, tsdf_cloud)
        write_mesh(raw_mesh_path, raw_mesh)
        write_mesh(clean_mesh_path, clean_mesh)
        print("SESSION")
        print(f"  path: {args.session}")
        print(f"  frames: {len(frame_dirs)}")
        print("TSDF")
        print(f"  voxel length: {policy.voxel_length} m")
        print(f"  sdf trunc: {policy.sdf_trunc} m")
        print(f"  min confidence: {policy.min_confidence}")
        print(f"  depth range: {policy.min_depth} .. {policy.max_depth} m")
        print(f"INTEGRATION: {len(prepared)} frames")
        _print_metrics("MESH RAW", raw_mesh)
        _print_metrics("MESH CLEAN", clean_mesh)
        print(f"OUTPUT\n  {tsdf_cloud_path}\n  {raw_mesh_path}\n  {clean_mesh_path}")
        print(f"PERFORMANCE\n  preparation: {preparation_seconds:.3f} s\n  integration: {integration_seconds:.3f} s\n  extraction: {extraction_seconds:.3f} s\n  total: {time.perf_counter() - started:.3f} s")
        if args.show_tsdf_cloud:
            o3d.visualization.draw_geometries([tsdf_cloud], window_name="TSDF point cloud")
        if args.show_raw_mesh:
            o3d.visualization.draw_geometries([raw_mesh], window_name="Raw TSDF mesh")
        if args.show:
            o3d.visualization.draw_geometries([clean_mesh], window_name="Clean TSDF mesh")
        print("Validation: PASS")
        return 0
    except (FrameValidationError, SessionValidationError, TSDFValidationError, OSError, ValueError) as exc:
        print(f"TSDF reconstruction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
