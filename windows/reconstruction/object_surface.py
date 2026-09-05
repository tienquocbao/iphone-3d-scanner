"""Windows-native Poisson and BPA reconstruction in canonical object space."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from object_tsdf import object_bounds
from surface_input import SurfaceInput, SurfaceInputConfig, prepare_surface_input
from tsdf import conservative_clean_mesh_components, mesh_metrics, validate_mesh, write_mesh


class SurfaceReconstructionError(ValueError):
    """Raised for invalid canonical observations or invalid surface output."""


@dataclass(frozen=True)
class SurfaceNormalConfig:
    """Normal configuration in meters; normals face toward their observing sensor."""

    radius_m: float = 0.012
    max_nn: int = 48
    orient_to_sensor: bool = True
    consistency_k: int | None = None

    def validate(self) -> None:
        if self.radius_m <= 0 or self.max_nn < 3:
            raise SurfaceReconstructionError("normal radius must be positive and max_nn must be at least 3")
        if self.consistency_k is not None and self.consistency_k < 3:
            raise SurfaceReconstructionError("normal consistency_k must be at least 3")


@dataclass(frozen=True)
class PoissonConfig:
    depth: int = 8
    width: float = 0.0
    scale: float = 1.1
    linear_fit: bool = False
    n_threads: int = -1
    density_quantile: float = 0.03
    object_bounds_margin_m: float = 0.04
    minimum_component_triangles: int = 50
    input: SurfaceInputConfig = SurfaceInputConfig()
    normals: SurfaceNormalConfig = SurfaceNormalConfig()

    def validate(self) -> None:
        if not 5 <= self.depth <= 12:
            raise SurfaceReconstructionError("Poisson depth must be between 5 and 12")
        if self.width < 0 or self.scale <= 0 or self.n_threads == 0:
            raise SurfaceReconstructionError("invalid Poisson solver configuration")
        if not 0 <= self.density_quantile < 1:
            raise SurfaceReconstructionError("density_quantile must be in [0, 1)")
        if self.object_bounds_margin_m < 0 or self.minimum_component_triangles < 1:
            raise SurfaceReconstructionError("invalid Poisson cleanup configuration")
        self.input.validate()
        self.normals.validate()


@dataclass(frozen=True)
class BPAConfig:
    radius_multipliers: tuple[float, ...] = (1.5, 3.0, 6.0)
    object_bounds_margin_m: float = 0.04
    minimum_component_triangles: int = 50
    input: SurfaceInputConfig = SurfaceInputConfig()
    normals: SurfaceNormalConfig = SurfaceNormalConfig()

    def validate(self) -> None:
        if not self.radius_multipliers or any(value <= 0 for value in self.radius_multipliers):
            raise SurfaceReconstructionError("BPA radius multipliers must be positive")
        if tuple(sorted(self.radius_multipliers)) != self.radius_multipliers:
            raise SurfaceReconstructionError("BPA radius multipliers must be ascending")
        if self.object_bounds_margin_m < 0 or self.minimum_component_triangles < 1:
            raise SurfaceReconstructionError("invalid BPA cleanup configuration")
        self.input.validate()
        self.normals.validate()


def _validate_aligned_observations(xyz: np.ndarray, sensor: np.ndarray, color: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or sensor.shape != xyz.shape or color.shape != xyz.shape:
        raise SurfaceReconstructionError("canonical xyz, sensor, and color must be aligned Nx3 arrays")
    if len(xyz) < 20 or not all(np.all(np.isfinite(value)) for value in (xyz, sensor, color)):
        raise SurfaceReconstructionError("canonical surface input must contain at least 20 finite aligned observations")
    if float(np.max(np.ptp(xyz, axis=0))) <= 1e-5:
        raise SurfaceReconstructionError("canonical surface input has zero spatial extent")


def orient_normals_to_sensors(points: np.ndarray, normals: np.ndarray, sensors: np.ndarray) -> tuple[np.ndarray, int]:
    """Flip each normal toward its aligned sensor origin.

    For a point ``p`` and observing sensor ``s``, a retained normal satisfies
    ``dot(normal, s - p) >= 0``.  This gives Poisson and BPA a consistent
    outward/view-facing orientation without guessing an object-wide up axis.
    """

    points, normals, sensors = (np.asarray(value, dtype=np.float64) for value in (points, normals, sensors))
    _validate_aligned_observations(points, sensors, np.zeros_like(points))
    if normals.shape != points.shape or not np.all(np.isfinite(normals)):
        raise SurfaceReconstructionError("normal array must be finite and aligned with points")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise SurfaceReconstructionError("normal estimation produced zero-length normals")
    oriented = normals / lengths[:, None]
    flips = np.einsum("ij,ij->i", oriented, sensors - points) < 0
    oriented[flips] *= -1.0
    return oriented, int(np.count_nonzero(flips))


def prepare_surface_cloud(prepared: SurfaceInput, config: SurfaceNormalConfig) -> tuple[o3d.geometry.PointCloud, dict[str, object]]:
    """Estimate and sensor-orient normals without losing xyz/sensor/color alignment."""

    config.validate()
    xyz = np.asarray(prepared.xyz, dtype=np.float64)
    sensor = np.asarray(prepared.sensor, dtype=np.float64)
    color = np.asarray(prepared.color, dtype=np.float64)
    if color.size and float(np.max(color)) > 1.0:
        color = color / 255.0
    _validate_aligned_observations(xyz, sensor, color)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(color, 0.0, 1.0))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=config.radius_m, max_nn=config.max_nn))
    normals = np.asarray(cloud.normals)
    if config.consistency_k is not None:
        cloud.orient_normals_consistent_tangent_plane(config.consistency_k)
        normals = np.asarray(cloud.normals)
    if config.orient_to_sensor:
        normals, flipped = orient_normals_to_sensors(xyz, normals, sensor)
        cloud.normals = o3d.utility.Vector3dVector(normals)
    else:
        flipped = 0
    distances = np.asarray(cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    finite_distances = distances[np.isfinite(distances) & (distances > 0)]
    return cloud, {
        "point_count": len(xyz),
        "normal_count": len(normals),
        "nonfinite_normal_count": int(np.count_nonzero(~np.isfinite(normals))),
        "flipped_by_sensor_count": flipped,
        "normal_radius_m": config.radius_m,
        "normal_max_nn": config.max_nn,
        "orientation_method": "sensor-facing" if config.orient_to_sensor else "open3d-estimated",
        "consistency_k": config.consistency_k,
        "median_nn_distance_m": float(np.median(finite_distances)) if len(finite_distances) else None,
    }


def adaptive_bpa_radii(cloud: o3d.geometry.PointCloud, multipliers: tuple[float, ...]) -> tuple[float, ...]:
    distances = np.asarray(cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
    finite = distances[np.isfinite(distances) & (distances > 0)]
    if not len(finite):
        raise SurfaceReconstructionError("cannot derive BPA radii from zero point spacing")
    spacing = float(np.median(finite))
    return tuple(spacing * value for value in multipliers)


def create_poisson_mesh(cloud: o3d.geometry.PointCloud, config: PoissonConfig) -> tuple[o3d.geometry.TriangleMesh, object]:
    """Small seam retained for deterministic failure-isolation tests."""

    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=config.depth,
        width=config.width,
        scale=config.scale,
        linear_fit=config.linear_fit,
        n_threads=config.n_threads,
    )


def create_bpa_mesh(cloud: o3d.geometry.PointCloud, radii: tuple[float, ...]) -> o3d.geometry.TriangleMesh:
    """Small seam retained for deterministic failure-isolation tests."""

    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii)
    )


def _crop_to_bounds(mesh: o3d.geometry.TriangleMesh, minimum: np.ndarray, maximum: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, int]:
    before = len(mesh.vertices)
    box = o3d.geometry.AxisAlignedBoundingBox(np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64))
    cropped = mesh.crop(box)
    if not len(cropped.triangles):
        raise SurfaceReconstructionError("surface mesh was empty after conservative object bounds crop")
    return cropped, before - len(cropped.vertices)


def _density_stats(densities: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(densities)),
        "p05": float(np.percentile(densities, 5)),
        "median": float(np.median(densities)),
        "p95": float(np.percentile(densities, 95)),
        "maximum": float(np.max(densities)),
    }


def _clean_mesh(mesh: o3d.geometry.TriangleMesh, minimum: np.ndarray, maximum: np.ndarray, minimum_component_triangles: int) -> tuple[o3d.geometry.TriangleMesh, dict[str, object]]:
    cropped, cropped_vertices = _crop_to_bounds(mesh, minimum, maximum)
    clean, components = conservative_clean_mesh_components(cropped, minimum_component_triangles)
    return clean, {
        "cropped_vertices": cropped_vertices,
        "component_count": components.get("component_count"),
        "connected_components": components,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _promote_directory(temporary: Path, output: Path) -> None:
    """Swap a complete backend directory in only after all artifacts validate."""

    backup = output.with_name(f".{output.name}-previous")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if output.exists():
        output.replace(backup)
    try:
        temporary.replace(output)
    except Exception:
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def observed_point_consistency(points: np.ndarray, mesh_path: Path, maximum_samples: int = 50_000) -> dict[str, object] | None:
    """Captured-point consistency only; this is not a ground-truth accuracy metric."""

    if not mesh_path.is_file():
        return None
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        return None
    sample = np.asarray(points, dtype=np.float32)
    if len(sample) > maximum_samples:
        sample = sample[np.linspace(0, len(sample) - 1, maximum_samples, dtype=np.int64)]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    values = scene.compute_distance(o3d.core.Tensor(sample)).numpy()
    return {"sample_count": len(sample), "mean_m": float(np.mean(values)), "median_m": float(np.median(values)), "p95_m": float(np.percentile(values, 95))}


def _comparison(artifact_dir: Path, points: np.ndarray) -> None:
    reconstruction = artifact_dir / "object" / "reconstruction"
    entries: dict[str, object] = {}
    for backend in ("tsdf", "poisson", "bpa", "nksr"):
        report = reconstruction / backend / "reconstruction.json"
        mesh = reconstruction / backend / f"object_{backend}_clean.ply"
        if not report.is_file() or not mesh.is_file():
            continue
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            metrics = mesh_metrics(o3d.io.read_triangle_mesh(str(mesh)))
            entries[backend] = {
                "runtime_seconds": payload.get("processing_seconds") or payload.get("timings", {}).get("total_seconds"),
                "vertices": metrics.vertices,
                "triangles": metrics.triangles,
                "observed_point_consistency": observed_point_consistency(points, mesh),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    _write_json(reconstruction / "comparison.json", {"coordinate_frame": "object", "note": "Observed-point consistency measures captured-point distance, not ground-truth accuracy. TSDF uses RGB-D integration; Poisson/BPA use canonical point samples.", "backends": entries})


def _mesh_summary(mesh: o3d.geometry.TriangleMesh) -> dict[str, object]:
    metrics = mesh_metrics(mesh)
    return {"vertices": metrics.vertices, "triangles": metrics.triangles, "bbox_min": metrics.bbox_min.tolist(), "bbox_max": metrics.bbox_max.tolist(), "surface_area_m2": metrics.surface_area}


def _surface_context(session_dir: Path, artifact_dir: Path, config: SurfaceInputConfig, margin: float, progress) -> tuple[SurfaceInput, np.ndarray, np.ndarray, str]:
    if progress:
        progress(5, "PREPARING CANONICAL OBSERVATIONS")
    prepared = prepare_surface_input(session_dir, artifact_dir, config, progress)
    pass_count = int(prepared.summary["pass_count"])
    minimum, maximum, source = object_bounds(artifact_dir, pass_count, margin)
    return prepared, minimum, maximum, source


def build_object_poisson(session_dir: Path, artifact_dir: Path, config: PoissonConfig = PoissonConfig(), progress=None) -> dict[str, object]:
    config.validate()
    started = time.perf_counter()
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    prepared, minimum, maximum, bounds_source = _surface_context(session_dir, artifact_dir, config.input, config.object_bounds_margin_m, progress)
    if progress:
        progress(55, "ESTIMATING NORMALS")
    normals_started = time.perf_counter()
    cloud, normal_report = prepare_surface_cloud(prepared, config.normals)
    normals_seconds = time.perf_counter() - normals_started
    output = artifact_dir / "object" / "reconstruction" / "poisson"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".poisson-", dir=output.parent))
    success = False
    try:
        if progress:
            progress(70, "RUNNING POISSON")
        solver_started = time.perf_counter()
        raw_mesh, density_vector = create_poisson_mesh(cloud, config)
        solver_seconds = time.perf_counter() - solver_started
        validate_mesh(raw_mesh)
        densities = np.asarray(density_vector, dtype=np.float64)
        if len(densities) != len(raw_mesh.vertices) or not np.all(np.isfinite(densities)):
            raise SurfaceReconstructionError("Poisson returned invalid density values")
        write_mesh(temporary / "object_poisson_raw.ply", raw_mesh)
        if progress:
            progress(85, "TRIMMING POISSON DENSITY")
        threshold = float(np.quantile(densities, config.density_quantile))
        density_mesh = o3d.geometry.TriangleMesh(raw_mesh)
        density_mesh.remove_vertices_by_mask(densities < threshold)
        clean_mesh, cleanup = _clean_mesh(density_mesh, minimum, maximum, config.minimum_component_triangles)
        write_mesh(temporary / "object_poisson_clean.ply", clean_mesh)
        report = {
            "schema_version": 1, "backend": "poisson", "coordinate_frame": "object",
            "input_points": len(prepared.xyz), "pass_transforms_sha256": prepared.pass_transforms_sha256,
            "config": {**asdict(config), "input": asdict(config.input), "normals": asdict(config.normals)},
            "normal_diagnostics": normal_report,
            "poisson": {"depth": config.depth, "width": config.width, "scale": config.scale, "linear_fit": config.linear_fit, "n_threads": config.n_threads},
            "density_filter": {"quantile": config.density_quantile, "threshold": threshold, "statistics": _density_stats(densities), "removed_vertices": int(np.count_nonzero(densities < threshold))},
            "object_bounds": {"source": bounds_source, "minimum": minimum.tolist(), "maximum": maximum.tolist(), "margin_m": config.object_bounds_margin_m},
            "mesh": {"raw": _mesh_summary(raw_mesh), "clean": _mesh_summary(clean_mesh), **cleanup},
            "processing_seconds": time.perf_counter() - started,
            "timings": {"normal_seconds": normals_seconds, "reconstruction_seconds": solver_seconds, "total_seconds": time.perf_counter() - started},
            "warnings": [], "outputs": {"raw_mesh": "object/reconstruction/poisson/object_poisson_raw.ply", "clean_mesh": "object/reconstruction/poisson/object_poisson_clean.ply"},
        }
        _write_json(temporary / "reconstruction.json", report)
        _promote_directory(temporary, output)
        success = True
        try:
            _comparison(artifact_dir, prepared.xyz)
        except Exception:
            # Comparison is diagnostic-only; a valid native mesh stays valid if
            # another backend's stale artifact cannot be inspected.
            pass
        if progress:
            progress(100, "DONE", report)
        return report
    finally:
        if not success:
            shutil.rmtree(temporary, ignore_errors=True)


def build_object_bpa(session_dir: Path, artifact_dir: Path, config: BPAConfig = BPAConfig(), progress=None) -> dict[str, object]:
    config.validate()
    started = time.perf_counter()
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    prepared, minimum, maximum, bounds_source = _surface_context(session_dir, artifact_dir, config.input, config.object_bounds_margin_m, progress)
    if progress:
        progress(55, "ESTIMATING NORMALS")
    normals_started = time.perf_counter()
    cloud, normal_report = prepare_surface_cloud(prepared, config.normals)
    radii = adaptive_bpa_radii(cloud, config.radius_multipliers)
    normals_seconds = time.perf_counter() - normals_started
    output = artifact_dir / "object" / "reconstruction" / "bpa"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bpa-", dir=output.parent))
    success = False
    try:
        if progress:
            progress(70, "RUNNING BALL PIVOTING")
        solver_started = time.perf_counter()
        raw_mesh = create_bpa_mesh(cloud, radii)
        solver_seconds = time.perf_counter() - solver_started
        validate_mesh(raw_mesh)
        write_mesh(temporary / "object_bpa_raw.ply", raw_mesh)
        if progress:
            progress(85, "CLEANING BPA MESH")
        clean_mesh, cleanup = _clean_mesh(raw_mesh, minimum, maximum, config.minimum_component_triangles)
        write_mesh(temporary / "object_bpa_clean.ply", clean_mesh)
        report = {
            "schema_version": 1, "backend": "bpa", "coordinate_frame": "object",
            "input_points": len(prepared.xyz), "pass_transforms_sha256": prepared.pass_transforms_sha256,
            "config": {**asdict(config), "input": asdict(config.input), "normals": asdict(config.normals)},
            "normal_diagnostics": normal_report,
            "bpa": {"radius_multipliers": list(config.radius_multipliers), "ball_radii_m": list(radii), "median_nn_distance_m": normal_report["median_nn_distance_m"]},
            "object_bounds": {"source": bounds_source, "minimum": minimum.tolist(), "maximum": maximum.tolist(), "margin_m": config.object_bounds_margin_m},
            "mesh": {"raw": _mesh_summary(raw_mesh), "clean": _mesh_summary(clean_mesh), **cleanup},
            "processing_seconds": time.perf_counter() - started,
            "timings": {"normal_seconds": normals_seconds, "reconstruction_seconds": solver_seconds, "total_seconds": time.perf_counter() - started},
            "warnings": [], "outputs": {"raw_mesh": "object/reconstruction/bpa/object_bpa_raw.ply", "clean_mesh": "object/reconstruction/bpa/object_bpa_clean.ply"},
        }
        _write_json(temporary / "reconstruction.json", report)
        _promote_directory(temporary, output)
        success = True
        try:
            _comparison(artifact_dir, prepared.xyz)
        except Exception:
            # Comparison is diagnostic-only; a valid native mesh stays valid if
            # another backend's stale artifact cannot be inspected.
            pass
        if progress:
            progress(100, "DONE", report)
        return report
    finally:
        if not success:
            shutil.rmtree(temporary, ignore_errors=True)
