"""Isolated entry point executed inside an optional NKSR environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _probe() -> dict[str, object]:
    import nksr
    import torch

    cuda = bool(torch.cuda.is_available())
    result: dict[str, object] = {
        "available": True,
        "nksr_version": getattr(nksr, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime_version": torch.version.cuda,
        "supported_modes": ["auto", "full", "chunk", "cpu"],
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_vram_bytes": int(properties.total_memory),
            }
        )
    return result


def _write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("NKSR returned invalid mesh arrays")
    if not len(vertices) or not len(faces) or not np.all(np.isfinite(vertices)):
        raise ValueError("NKSR returned an empty or non-finite mesh")
    vertex_dtype: list[tuple] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    normalized_colors = None
    if colors is not None:
        normalized_colors = np.asarray(colors)
        if normalized_colors.shape == vertices.shape and np.all(np.isfinite(normalized_colors)):
            if np.issubdtype(normalized_colors.dtype, np.floating):
                normalized_colors = np.clip(np.rint(normalized_colors * 255.0), 0, 255)
            normalized_colors = normalized_colors.astype(np.uint8)
            vertex_dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        else:
            normalized_colors = None
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices.T
    if normalized_colors is not None:
        vertex_data["red"], vertex_data["green"], vertex_data["blue"] = normalized_colors.T
    face_data = np.empty(len(faces), dtype=[("count", "u1"), ("vertices", "<i4", (3,))])
    face_data["count"] = 3
    face_data["vertices"] = faces
    color_header = "property uchar red\nproperty uchar green\nproperty uchar blue\n" if normalized_colors is not None else ""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{color_header}"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(header)
        vertex_data.tofile(handle)
        face_data.tofile(handle)
    os.replace(temporary, path)


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: memory" in message


def _select_attempts(config: dict[str, object], probe: dict[str, object], point_count: int) -> list[tuple[str, str]]:
    mode = str(config["execution_mode"])
    device = str(config["device"])
    cuda = bool(probe.get("cuda_available"))
    if mode == "cpu" or device == "cpu" or (device == "auto" and not cuda):
        return [("cpu", "cpu")]
    if device == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but unavailable in the NKSR environment")
    if mode in {"full", "chunk"}:
        return [(mode, "cuda")]
    vram = int(probe.get("gpu_vram_bytes") or 0)
    low_vram = int(config["low_vram_threshold_bytes"])
    full_limit = int(config["full_cuda_max_points"])
    first = "chunk" if vram <= low_vram or point_count > full_limit else "full"
    attempts = [(first, "cuda")]
    if first == "full":
        attempts.append(("chunk", "cuda"))
    if bool(config.get("cpu_fallback")):
        attempts.append(("cpu", "cpu"))
    return attempts


def _run_attempt(
    xyz: np.ndarray,
    sensor: np.ndarray,
    color: np.ndarray | None,
    config: dict[str, object],
    mode: str,
    device_name: str,
):
    import nksr
    import torch

    scale = 1.0
    xyz_work = xyz
    sensor_work = sensor
    kwargs: dict[str, object] = {}
    if mode == "chunk":
        voxel_size = float(config["nksr_voxel_size_m"])
        scale = 0.1 / voxel_size
        xyz_work = xyz * scale
        sensor_work = sensor * scale
        kwargs.update(
            chunk_size=float(config["chunk_size"]),
            preprocess_fn=nksr.get_estimate_normal_preprocess_fn(
                int(config["normal_knn"]), float(config["normal_drop_threshold_degrees"])
            ),
        )
    elif config.get("nksr_voxel_size_m") is not None:
        kwargs["voxel_size"] = float(config["nksr_voxel_size_m"])
        kwargs["detail_level"] = None
    else:
        kwargs["detail_level"] = float(config["detail_level"])
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    reconstructor = nksr.Reconstructor(device)
    if mode == "chunk":
        reconstructor.chunk_tmp_device = torch.device("cpu")
    input_xyz = torch.from_numpy(xyz_work).float().to(device)
    input_sensor = torch.from_numpy(sensor_work).float().to(device)
    field = reconstructor.reconstruct(input_xyz, sensor=input_sensor, **kwargs)
    if field is None:
        raise RuntimeError("NKSR produced no reconstruction field")
    color_status = "unavailable"
    if color is not None and bool(config.get("enable_color")):
        try:
            input_color = torch.from_numpy(color.astype(np.float32) / 255.0).float().to(device)
            field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
            color_status = "vertex_color"
        except Exception as exc:
            color_status = f"unavailable: {str(exc)[:160]}"
    if mode == "chunk" and device_name == "cuda":
        field.to_("cpu")
        reconstructor.network.to("cpu")
    mesh = field.extract_dual_mesh(mise_iter=int(config["mise_iter"]))
    vertices = np.asarray(mesh.v.detach().cpu() if hasattr(mesh.v, "detach") else mesh.v, dtype=np.float32) / scale
    faces = np.asarray(mesh.f.detach().cpu() if hasattr(mesh.f, "detach") else mesh.f, dtype=np.int32)
    mesh_color = getattr(mesh, "c", None)
    if mesh_color is not None:
        mesh_color = np.asarray(mesh_color.detach().cpu() if hasattr(mesh_color, "detach") else mesh_color)
    return vertices, faces, mesh_color, scale, color_status


def _run(input_path: Path, output_path: Path, config: dict[str, object]) -> dict[str, object]:
    import torch

    probe = _probe()
    with np.load(input_path) as payload:
        xyz = np.asarray(payload["xyz"], dtype=np.float32)
        sensor = np.asarray(payload["sensor"], dtype=np.float32)
        color = np.asarray(payload["color"], dtype=np.uint8) if "color" in payload else None
    if xyz.shape != sensor.shape or xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError("prepared NKSR xyz/sensor arrays are not aligned Nx3")
    attempts: list[dict[str, object]] = []
    plan = _select_attempts(config, probe, len(xyz))
    last_error: BaseException | None = None
    for index, (mode, device) in enumerate(plan):
        attempt_started = time.perf_counter()
        try:
            vertices, faces, colors, scale, color_status = _run_attempt(
                xyz, sensor, color, config, mode, device
            )
            _write_binary_ply(output_path, vertices, faces, colors)
            attempts.append({"mode": mode, "device": device, "status": "success", "seconds": time.perf_counter() - attempt_started})
            return {
                "status": "success",
                "mode": mode,
                "device": device,
                "vertices": len(vertices),
                "triangles": len(faces),
                "input_scale_to_nksr": scale,
                "color_status": color_status,
                "attempts": attempts,
                "backend": probe,
                "checkpoint": "ks (kitchen-sink default)",
            }
        except BaseException as exc:
            last_error = exc
            oom = _is_oom(exc)
            attempts.append({"mode": mode, "device": device, "status": "oom" if oom else "failed", "error": str(exc)[:512], "seconds": time.perf_counter() - attempt_started})
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            next_is_fallback = index + 1 < len(plan)
            if not oom or not next_is_fallback:
                break
    raise RuntimeError(json.dumps({"message": str(last_error)[:512], "attempts": attempts}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.probe:
            payload = _probe()
        else:
            if args.input is None or args.output is None or args.config is None:
                raise ValueError("--input, --output, and --config are required")
            config = json.loads(args.config.read_text(encoding="utf-8"))
            payload = _run(args.input, args.output, config)
        _write_json(args.result, payload)
        return 0
    except BaseException as exc:
        _write_json(
            args.result,
            {
                "available": False if args.probe else None,
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "error": (str(exc).strip() or exc.__class__.__name__)[:2048],
                "traceback": traceback.format_exc(limit=5)[-4096:],
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
