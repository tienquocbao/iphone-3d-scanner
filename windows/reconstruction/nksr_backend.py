"""Failure-isolated optional NKSR adapter and Object Scan orchestration."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from nksr_input import NKSRInputConfig, prepare_nksr_input, write_nksr_input
from tsdf import conservative_clean_mesh_components, mesh_metrics, validate_mesh, write_mesh


OFFICIAL_NKSR_COMMIT = "e40336845e67761343a756788e5a98b827d4a143"
OFFICIAL_NKSR_VERSION = "1.0.3"


class NKSRUnavailableError(RuntimeError):
    pass


class NKSRExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class NKSRConfig:
    device: str = "auto"
    execution_mode: str = "auto"
    nksr_voxel_size_m: float | None = 0.005
    detail_level: float | None = None
    chunk_size: float = 20.0
    mise_iter: int = 1
    enable_color: bool = True
    cpu_fallback: bool = False
    timeout_seconds: int = 1800
    low_vram_threshold_bytes: int = 6 * 1024**3
    full_cuda_max_points: int = 100_000
    normal_knn: int = 64
    normal_drop_threshold_degrees: float = 85.0
    mesh_min_component_triangles: int = 50
    retain_failed_input: bool = False
    input: NKSRInputConfig = NKSRInputConfig()

    def validate(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.execution_mode not in {"auto", "full", "chunk", "cpu"}:
            raise ValueError("execution_mode must be auto, full, chunk, or cpu")
        if self.nksr_voxel_size_m is not None and self.nksr_voxel_size_m <= 0:
            raise ValueError("nksr_voxel_size_m must be positive")
        if self.detail_level is not None and not 0 <= self.detail_level <= 1:
            raise ValueError("detail_level must be in [0,1]")
        if self.nksr_voxel_size_m is not None and self.detail_level is not None:
            raise ValueError("nksr_voxel_size_m and detail_level are mutually exclusive")
        if self.execution_mode == "chunk" and self.nksr_voxel_size_m is None:
            raise ValueError("chunk mode requires nksr_voxel_size_m for documented metric scaling")
        if self.chunk_size <= 0 or self.mise_iter < 0 or self.timeout_seconds < 1:
            raise ValueError("invalid NKSR execution limits")
        if self.full_cuda_max_points < 1 or self.low_vram_threshold_bytes < 1:
            raise ValueError("invalid NKSR automatic mode policy")
        if self.normal_knn < 1 or not 0 < self.normal_drop_threshold_degrees <= 180:
            raise ValueError("invalid NKSR sensor-normal preprocessing policy")
        if self.mesh_min_component_triangles < 1:
            raise ValueError("mesh_min_component_triangles must be positive")
        self.input.validate()

    def runner_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("input")
        payload.pop("timeout_seconds")
        payload.pop("retain_failed_input")
        payload.pop("mesh_min_component_triangles")
        return payload


def _configured_python(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    configured = os.environ.get("IPHONE3D_NKSR_PYTHON", "").strip()
    if configured:
        return Path(configured)
    if importlib.util.find_spec("nksr") is not None and importlib.util.find_spec("torch") is not None:
        return Path(sys.executable)
    return None


def _configured_runner(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    configured = os.environ.get("IPHONE3D_NKSR_RUNNER", "").strip()
    return Path(configured) if configured else Path(__file__).with_name("nksr_runner.py")


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _invoke(
    command: list[str],
    result_path: Path,
    timeout_seconds: int,
) -> tuple[int, dict[str, object], str]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        output, _ = process.communicate()
        raise NKSRExecutionError(f"NKSR subprocess timed out after {timeout_seconds}s") from exc
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NKSRExecutionError(
            f"NKSR subprocess returned {process.returncode} without a valid structured result"
        ) from exc
    return process.returncode, result, output[-4096:]


def probe_nksr_backend(
    python_executable: Path | None = None,
    runner_path: Path | None = None,
    timeout_seconds: int = 20,
) -> dict[str, object]:
    python = _configured_python(python_executable)
    if python is None:
        return {
            "available": False,
            "reason": "IPHONE3D_NKSR_PYTHON is not configured and NKSR is not installed in the current environment",
            "official_version": OFFICIAL_NKSR_VERSION,
            "official_commit": OFFICIAL_NKSR_COMMIT,
        }
    if not python.is_file():
        return {"available": False, "reason": "configured NKSR Python executable does not exist", "official_version": OFFICIAL_NKSR_VERSION, "official_commit": OFFICIAL_NKSR_COMMIT}
    runner = _configured_runner(runner_path)
    if not runner.is_file():
        return {"available": False, "reason": "configured NKSR runner does not exist", "official_version": OFFICIAL_NKSR_VERSION, "official_commit": OFFICIAL_NKSR_COMMIT}
    with tempfile.TemporaryDirectory(prefix="iphone3d-nksr-probe-") as temporary:
        result_path = Path(temporary) / "result.json"
        try:
            returncode, result, _ = _invoke(
                [str(python), str(runner), "--probe", "--result", str(result_path)],
                result_path,
                timeout_seconds,
            )
        except NKSRExecutionError as exc:
            return {"available": False, "reason": str(exc), "official_version": OFFICIAL_NKSR_VERSION, "official_commit": OFFICIAL_NKSR_COMMIT}
    if returncode != 0 or not result.get("available"):
        return {"available": False, "reason": str(result.get("error") or "NKSR probe failed")[:512], "official_version": OFFICIAL_NKSR_VERSION, "official_commit": OFFICIAL_NKSR_COMMIT}
    result.update({"official_version": OFFICIAL_NKSR_VERSION, "official_commit": OFFICIAL_NKSR_COMMIT})
    return result


def public_nksr_capability(capability: dict[str, object]) -> dict[str, object]:
    allowed = {
        "available", "reason", "nksr_version", "torch_version", "cuda_available",
        "cuda_runtime_version", "gpu_name", "gpu_vram_bytes", "supported_modes",
        "official_version", "official_commit",
    }
    return {key: value for key, value in capability.items() if key in allowed}


def _distance_report(points: np.ndarray, mesh_path: Path, maximum_samples: int = 50_000) -> dict[str, object] | None:
    if not mesh_path.is_file():
        return None
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        return None
    sample = points
    if len(sample) > maximum_samples:
        sample = sample[np.linspace(0, len(sample) - 1, maximum_samples, dtype=np.int64)]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(sample.astype(np.float32))).numpy()
    return {
        "sample_count": len(sample),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95)),
    }


def build_object_nksr(
    session_dir: Path,
    artifact_dir: Path,
    config: NKSRConfig = NKSRConfig(),
    progress=None,
    python_executable: Path | None = None,
    runner_path: Path | None = None,
) -> dict[str, object]:
    """Prepare observations in the main environment and run NKSR beyond a process boundary."""

    config.validate()
    started = time.perf_counter()
    capability = probe_nksr_backend(python_executable, runner_path)
    if not capability.get("available"):
        raise NKSRUnavailableError(f"NKSR unavailable: {capability.get('reason', 'capability probe failed')}")
    python = _configured_python(python_executable)
    assert python is not None
    output_dir = Path(artifact_dir) / "object" / "reconstruction" / "nksr"
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(5, "PREPARING NKSR INPUT")
    prepared = prepare_nksr_input(Path(session_dir), Path(artifact_dir), config.input, progress)
    input_summary_path = output_dir / "input_summary.json"
    job_temp = Path(tempfile.mkdtemp(prefix=".job-", dir=output_dir))
    input_path = job_temp / "input.npz"
    config_path = job_temp / "config.json"
    result_path = job_temp / "result.json"
    raw_path = output_dir / "object_nksr_raw.ply"
    clean_path = output_dir / "object_nksr_clean.ply"
    temporary_raw_path = job_temp / "object_nksr_raw.ply"
    temporary_clean_path = job_temp / "object_nksr_clean.ply"
    runner = _configured_runner(runner_path)
    success = False
    try:
        write_nksr_input(input_path, prepared)
        config_path.write_text(json.dumps(config.runner_payload(), indent=2), encoding="utf-8")
        if progress:
            progress(60, "RUNNING NKSR")
        returncode, runner_result, _ = _invoke(
            [str(python), str(runner), "--input", str(input_path), "--output", str(temporary_raw_path), "--config", str(config_path), "--result", str(result_path)],
            result_path,
            config.timeout_seconds,
        )
        if returncode != 0 or runner_result.get("status") != "success":
            reason = runner_result.get("error") or "NKSR subprocess failed"
            raise NKSRExecutionError(str(reason)[:2048])
        if progress:
            progress(85, "VALIDATING NKSR MESH")
        raw_mesh = o3d.io.read_triangle_mesh(str(temporary_raw_path))
        validate_mesh(raw_mesh)
        clean_mesh, components = conservative_clean_mesh_components(
            raw_mesh, config.mesh_min_component_triangles
        )
        write_mesh(temporary_clean_path, clean_mesh)
        tsdf_path = Path(artifact_dir) / "object" / "reconstruction" / "tsdf" / "object_tsdf_clean.ply"
        consistency = {
            "observed_to_nksr": _distance_report(prepared.xyz, temporary_clean_path),
            "observed_to_tsdf": _distance_report(prepared.xyz, tsdf_path),
            "note": "Distances measure consistency with observed points, not ground-truth accuracy.",
        }
        metrics = mesh_metrics(clean_mesh)
        report = {
            "schema_version": 1,
            "backend": "nksr",
            "coordinate_frame": "object",
            "device": runner_result.get("device"),
            "execution_mode": runner_result.get("mode"),
            "input_points": len(prepared.xyz),
            "pass_transforms_sha256": prepared.pass_transforms_sha256,
            "config": {**asdict(config), "input": asdict(config.input)},
            "backend_info": capability,
            "checkpoint": runner_result.get("checkpoint"),
            "attempts": runner_result.get("attempts", []),
            "input_scale_to_nksr": runner_result.get("input_scale_to_nksr", 1.0),
            "color_status": runner_result.get("color_status", "unavailable"),
            "mesh": {
                "raw_vertices": runner_result.get("vertices"),
                "raw_triangles": runner_result.get("triangles"),
                "clean_vertices": metrics.vertices,
                "clean_triangles": metrics.triangles,
                "connected_components": components,
            },
            "observed_point_consistency": consistency,
            "processing_seconds": time.perf_counter() - started,
            "warnings": [],
            "outputs": {
                "raw_mesh": "object/reconstruction/nksr/object_nksr_raw.ply",
                "clean_mesh": "object/reconstruction/nksr/object_nksr_clean.ply",
                "input_summary": "object/reconstruction/nksr/input_summary.json",
            },
        }
        os.replace(temporary_raw_path, raw_path)
        os.replace(temporary_clean_path, clean_path)
        temporary_summary = job_temp / "input_summary.json"
        temporary_summary.write_text(json.dumps(prepared.summary, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_summary, input_summary_path)
        temporary_report = job_temp / "reconstruction.json"
        temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_report, output_dir / "reconstruction.json")
        success = True
        if progress:
            progress(100, "DONE", report)
        return report
    finally:
        if success or not config.retain_failed_input:
            shutil.rmtree(job_temp, ignore_errors=True)
