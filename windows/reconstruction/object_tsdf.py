"""Gate C1 object-relative TSDF reconstruction from masked RGB-D observations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import open3d as o3d

from foreground import (
    ForegroundConfig,
    calibrate_green_background,
    foreground_mask,
    project_rgb_mask_to_depth,
)
from frame_io import FrameData, load_frame
from fuse_session import SessionValidationError, session_frame_dirs
from geometry import depth_to_world_points
from registration import compose_object_from_camera
from tsdf import (
    TSDFPolicy,
    conservative_clean_mesh_components,
    mesh_metrics,
    prepare_tsdf_frame,
    validate_mesh,
    write_mesh,
)


OBJECT_TSDF_VERSION = 1
SINGLE_PASS_TRANSFORM_ID = "single-pass-identity-v1"


@dataclass(frozen=True)
class ObjectTSDFConfig:
    """CPU-safe defaults for tabletop-scale object reconstruction, in meters."""

    voxel_length: float = 0.003
    sdf_trunc: float = 0.015
    min_confidence: int = 1
    min_depth_m: float | None = 0.10
    max_depth_m: float | None = 3.0
    object_bounds_margin_m: float = 0.04
    minimum_foreground_samples: int = 16
    mesh_min_component_triangles: int = 50
    mesh_cleanup_enabled: bool = True
    foreground: ForegroundConfig = ForegroundConfig()

    def validate(self) -> None:
        TSDFPolicy(
            self.voxel_length,
            self.sdf_trunc,
            self.min_confidence,
            self.min_depth_m,
            self.max_depth_m,
        ).validate()
        if self.object_bounds_margin_m < 0:
            raise ValueError("object_bounds_margin_m must not be negative")
        if self.minimum_foreground_samples < 1:
            raise ValueError("minimum_foreground_samples must be positive")
        if self.mesh_min_component_triangles < 1:
            raise ValueError("mesh_min_component_triangles must be positive")
        self.foreground.validate()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_session_passes(metadata: dict[str, object], frame_count: int) -> list[dict[str, int]]:
    if metadata.get("scan_mode") != "object":
        raise SessionValidationError("Build Object Mesh (TSDF) requires scan_mode=object")
    raw_passes = metadata.get("passes")
    if not isinstance(raw_passes, list) or not raw_passes:
        # Gate A sessions predate explicit multi-pass UX but are one complete pass.
        return [{"id": 0, "start_frame": 0, "end_frame": frame_count - 1}]
    result: list[dict[str, int]] = []
    expected_start = 0
    for expected_id, item in enumerate(raw_passes):
        if not isinstance(item, dict):
            raise SessionValidationError("invalid object pass metadata")
        pass_id = item.get("id")
        start = item.get("start_frame")
        end = item.get("end_frame")
        if type(pass_id) is not int or type(start) is not int or type(end) is not int:
            raise SessionValidationError("object pass IDs and frame ranges must be integers")
        if pass_id != expected_id or start != expected_start or start > end or end >= frame_count:
            raise SessionValidationError("object passes must be sequential, contiguous, and cover valid frames")
        result.append({"id": pass_id, "start_frame": start, "end_frame": end})
        expected_start = end + 1
    if expected_start != frame_count:
        raise SessionValidationError("every object frame must belong to exactly one completed pass")
    return result


def _finite_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise SessionValidationError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4):
        raise SessionValidationError(f"{label} final row must be approximately [0,0,0,1]")
    return matrix


def load_object_from_pass_transforms(
    passes: list[dict[str, int]],
    transform_path: Path,
) -> tuple[dict[int, np.ndarray], str]:
    """Load only accepted Gate B transforms; synthesize identity for one Gate A pass."""

    if len(passes) == 1 and not transform_path.is_file():
        return {0: np.eye(4)}, SINGLE_PASS_TRANSFORM_ID
    if not transform_path.is_file():
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: pass_transforms.json is missing")
    try:
        payload = json.loads(transform_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: pass transforms are invalid") from exc
    if payload.get("canonical_pass") != 0:
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: canonical pass must be 0")
    entries = payload.get("passes")
    if not isinstance(entries, list):
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: transform entries are missing")
    transforms: dict[int, np.ndarray] = {}
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("id")) is not int:
            raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: invalid transform entry")
        pass_id = entry["id"]
        expected_status = "reference" if pass_id == 0 else "accepted"
        if entry.get("registration_status") != expected_status:
            raise SessionValidationError(
                f"OBJECT_REGISTRATION_REQUIRED: pass {pass_id} registration is not {expected_status}"
            )
        transforms[pass_id] = _finite_transform(entry.get("object_from_pass"), f"pass {pass_id} object_from_pass")
    expected_ids = {item["id"] for item in passes}
    if set(transforms) != expected_ids:
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: transforms do not match completed passes")
    if not np.allclose(transforms[0], np.eye(4), atol=1e-6):
        raise SessionValidationError("OBJECT_REGISTRATION_REQUIRED: canonical pass transform must be identity")
    return transforms, _sha256(transform_path)


def object_bounds(artifact_dir: Path, pass_count: int, margin: float) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = (
        [artifact_dir / "object" / "object_registered_clean.ply"]
        if pass_count > 1
        else [
            artifact_dir / "object" / "object_registered_clean.ply",
            artifact_dir / "object" / "object_clean.ply",
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        prerequisite = "Build Registered Object Cloud" if pass_count > 1 else "Build Object Point Cloud"
        raise SessionValidationError(f"{prerequisite} before Build Object Mesh (TSDF)")
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if not len(points) or not np.all(np.isfinite(points)):
        raise SessionValidationError("object bounds point cloud is empty or invalid")
    return points.min(axis=0) - margin, points.max(axis=0) + margin, path.name


def object_depth_mask(
    frame: FrameData,
    rgb_foreground_mask: np.ndarray,
    object_from_camera: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    config: ObjectTSDFConfig,
) -> tuple[np.ndarray, int, int]:
    """Return a depth-sized mask intersecting chroma key, confidence, depth, and object bounds."""

    depth_foreground = project_rgb_mask_to_depth(rgb_foreground_mask, frame.depth.shape)
    object_frame = replace(frame, world_from_camera=object_from_camera)
    points, pixels, stats = depth_to_world_points(
        object_frame,
        config.min_confidence,
        config.min_depth_m,
        config.max_depth_m,
    )
    mask = np.zeros(frame.depth.shape, dtype=bool)
    if not len(points):
        return mask, int(stats["valid_samples"]), 0
    foreground = depth_foreground[pixels[:, 1], pixels[:, 0]]
    in_bounds = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
    keep = foreground & in_bounds
    kept_pixels = pixels[keep]
    mask[kept_pixels[:, 1], kept_pixels[:, 0]] = True
    return mask, int(stats["valid_samples"]), int(np.count_nonzero(keep))


def _mesh_report(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    metrics = mesh_metrics(mesh)
    return {
        "vertices": metrics.vertices,
        "triangles": metrics.triangles,
        "bbox_min": metrics.bbox_min.tolist(),
        "bbox_max": metrics.bbox_max.tolist(),
        "surface_area_m2": metrics.surface_area,
    }


def build_object_tsdf(
    session_dir: Path,
    artifact_dir: Path,
    config: ObjectTSDFConfig = ObjectTSDFConfig(),
    progress=None,
) -> dict[str, object]:
    """Integrate original per-frame observations in the canonical object coordinate frame."""

    config.validate()
    started = time.perf_counter()
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    try:
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionValidationError(f"Cannot read session.json: {exc}") from exc
    frame_dirs = session_frame_dirs(session_dir)
    passes = object_session_passes(metadata, len(frame_dirs))
    transform_path = artifact_dir / "object" / "registration" / "pass_transforms.json"
    transforms, transform_sha256 = load_object_from_pass_transforms(passes, transform_path)
    bounds_min, bounds_max, bounds_source = object_bounds(
        artifact_dir, len(passes), config.object_bounds_margin_m
    )
    policy = TSDFPolicy(
        config.voxel_length,
        config.sdf_trunc,
        config.min_confidence,
        config.min_depth_m,
        config.max_depth_m,
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_length,
        sdf_trunc=config.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    rejected: list[dict[str, str]] = []
    pass_reports: list[dict[str, object]] = []
    input_frames = len(frame_dirs)
    integrated_frames = 0
    masking_seconds = preparation_seconds = integration_seconds = 0.0

    for pass_index, item in enumerate(passes):
        pass_id = item["id"]
        selected = frame_dirs[item["start_frame"] : item["end_frame"] + 1]
        if progress:
            progress(5 + int(15 * pass_index / len(passes)), f"CALIBRATING PASS {pass_id + 1}")
        calibration_started = time.perf_counter()
        model = calibrate_green_background(
            (load_frame(directory).rgb for directory in selected), config.foreground
        )
        pass_mask_seconds = time.perf_counter() - calibration_started
        pass_integrated = 0
        for local_index, frame_dir in enumerate(selected):
            try:
                frame = load_frame(frame_dir)
                mask_started = time.perf_counter()
                rgb_mask = foreground_mask(frame.rgb, model, config.foreground)
                object_from_camera = compose_object_from_camera(
                    transforms[pass_id], frame.world_from_camera
                )
                depth_mask, _, foreground_samples = object_depth_mask(
                    frame,
                    rgb_mask,
                    object_from_camera,
                    bounds_min,
                    bounds_max,
                    config,
                )
                pass_mask_seconds += time.perf_counter() - mask_started
                if foreground_samples < config.minimum_foreground_samples:
                    rejected.append(
                        {
                            "frame": frame_dir.name,
                            "pass_id": str(pass_id),
                            "reason": f"only {foreground_samples} usable foreground depth samples",
                        }
                    )
                    continue
                preparation_started = time.perf_counter()
                prepared = prepare_tsdf_frame(
                    frame,
                    policy,
                    additional_depth_mask=depth_mask,
                    world_from_arkit=object_from_camera,
                )
                preparation_seconds += time.perf_counter() - preparation_started
                integration_started = time.perf_counter()
                volume.integrate(prepared.rgbd, prepared.intrinsic, prepared.extrinsic)
                integration_seconds += time.perf_counter() - integration_started
                integrated_frames += 1
                pass_integrated += 1
            except (OSError, ValueError) as exc:
                rejected.append(
                    {"frame": frame_dir.name, "pass_id": str(pass_id), "reason": str(exc)[:256]}
                )
            if progress:
                completed = item["start_frame"] + local_index + 1
                progress(20 + int(55 * completed / input_frames), f"INTEGRATING PASS {pass_id + 1}")
        masking_seconds += pass_mask_seconds
        pass_reports.append(
            {
                "id": pass_id,
                "input_frames": len(selected),
                "integrated_frames": pass_integrated,
                "rejected_frames": len(selected) - pass_integrated,
                "background_model": model.to_dict(),
            }
        )

    if integrated_frames == 0:
        raise SessionValidationError("Object TSDF integrated zero frames after masking and validation")
    if progress:
        progress(80, "EXTRACTING MESH")
    extraction_started = time.perf_counter()
    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    validate_mesh(raw_mesh)
    extraction_seconds = time.perf_counter() - extraction_started

    output_dir = artifact_dir / "object" / "reconstruction" / "tsdf"
    raw_path = output_dir / "object_tsdf_raw.ply"
    clean_path = output_dir / "object_tsdf_clean.ply"
    write_mesh(raw_path, raw_mesh)
    if progress:
        progress(90, "CLEANING MESH")
    cleanup_started = time.perf_counter()
    if config.mesh_cleanup_enabled:
        clean_mesh, components = conservative_clean_mesh_components(
            raw_mesh, config.mesh_min_component_triangles
        )
    else:
        clean_mesh = o3d.geometry.TriangleMesh(raw_mesh)
        components = {
            "component_count": None,
            "component_triangle_counts": [],
            "removed_components": 0,
            "removed_triangles": 0,
        }
    cleanup_seconds = time.perf_counter() - cleanup_started
    write_mesh(clean_path, clean_mesh)
    total_seconds = time.perf_counter() - started
    warnings: list[str] = []
    if rejected:
        warnings.append(f"{len(rejected)} frame(s) were rejected; inspect rejected_frames")
    report = {
        "schema_version": OBJECT_TSDF_VERSION,
        "backend": "tsdf",
        "coordinate_frame": "object",
        "pass_count": len(passes),
        "passes": pass_reports,
        "input_frames": input_frames,
        "integrated_frames": integrated_frames,
        "rejected_frames": rejected,
        "config": {**asdict(config), "foreground": asdict(config.foreground)},
        "registration": {
            "pass_transforms_sha256": transform_sha256,
            "transform_semantics": "object_from_camera = object_from_pass @ pass_world_from_camera",
        },
        "object_bounds": {
            "source": bounds_source,
            "minimum": bounds_min.tolist(),
            "maximum": bounds_max.tolist(),
            "margin_m": config.object_bounds_margin_m,
        },
        "mesh": {
            "raw": _mesh_report(raw_mesh),
            "clean": _mesh_report(clean_mesh),
            "connected_components": components,
        },
        "timings": {
            "masking_seconds": masking_seconds,
            "rgbd_preparation_seconds": preparation_seconds,
            "integration_seconds": integration_seconds,
            "extraction_seconds": extraction_seconds,
            "cleanup_seconds": cleanup_seconds,
            "total_seconds": total_seconds,
        },
        "warnings": warnings,
        "outputs": {
            "raw_mesh": "object/reconstruction/tsdf/object_tsdf_raw.ply",
            "clean_mesh": "object/reconstruction/tsdf/object_tsdf_clean.ply",
        },
    }
    _write_json(output_dir / "reconstruction.json", report)
    if progress:
        progress(100, "DONE", report)
    return report
