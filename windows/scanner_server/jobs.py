"""Crash-isolated CPU/GPU-aware reconstruction jobs."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION = ROOT / "reconstruction"


@lru_cache(maxsize=1)
def nksr_diagnostics() -> dict[str, object]:
    sys.path.insert(0, str(RECONSTRUCTION))
    from nksr_backend import probe_nksr_backend, public_nksr_capability

    return public_nksr_capability(probe_nksr_backend())


def device_diagnostics() -> dict[str, object]:
    result: dict[str, object] = {"torch_cuda_available": False, "open3d_cuda_available": False, "nvidia_gpu": None}
    try:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        result["torch_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            result["nvidia_gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["torch_available"] = False
    try:
        import open3d as o3d

        result["open3d_cuda_available"] = bool(getattr(o3d, "core", None) and o3d.core.cuda.is_available())
    except (ImportError, AttributeError):
        pass
    result["nksr"] = nksr_diagnostics()
    return result


def _write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _worker(session_dir: str, artifact_dir: str, kind: str, device: str) -> None:
    sys.path.insert(0, str(RECONSTRUCTION))
    job_path = Path(artifact_dir) / "job.json"
    started = time.time()
    try:
        _write(job_path, {"state": "running", "kind": kind, "progress": 5, "message": "Reading frames", "device": device, "started_at": started})
        if kind == "pointcloud":
            from frame_io import load_frame
            from fuse_session import _cloud, fuse_loaded_frames, session_frame_dirs, write_cloud
            from geometry import sample_rgb
            from point_backend import depth_to_world_points_backend

            frame_dirs = session_frame_dirs(Path(session_dir))
            frames = []
            for index, directory in enumerate(frame_dirs):
                frames.append(load_frame(directory))
                _write(job_path, {"state": "running", "kind": kind, "progress": int(5 + 45 * (index + 1) / len(frame_dirs)), "message": "Reading frames", "device": device, "started_at": started})
            if device == "cuda":
                import numpy as np

                points, colors = [], []
                for frame in frames:
                    frame_points, pixels = depth_to_world_points_backend(frame, "cuda")
                    points.append(frame_points); colors.append(sample_rgb(frame, pixels))
                raw = _cloud(np.concatenate(points), np.concatenate(colors))
                result_cloud = raw.voxel_down_sample(0.005)
            else:
                result_cloud = fuse_loaded_frames(frames).voxel_cloud
            _write(job_path, {"state": "running", "kind": kind, "progress": 80, "message": "Writing point cloud", "device": device, "started_at": started})
            count = write_cloud(Path(artifact_dir) / "pointcloud.ply", result_cloud)
            _write(job_path, {"state": "done", "kind": kind, "progress": 100, "message": "Point cloud complete", "device": device, "point_count": count, "finished_at": time.time()})
        elif kind == "object_pointcloud":
            from object_cloud import build_object_cloud

            def progress(value: int, message: str, metrics: dict[str, object] | None = None) -> None:
                payload: dict[str, object] = {"state": "running", "kind": kind, "progress": value, "message": message, "device": "cpu", "started_at": started}
                if metrics is not None:
                    payload["metrics"] = metrics
                _write(job_path, payload)

            metrics = build_object_cloud(Path(session_dir), Path(artifact_dir), progress=progress)
            _write(job_path, {"state": "done", "kind": kind, "progress": 100, "message": "DONE", "device": "cpu", "metrics": metrics, "finished_at": time.time()})
        elif kind == "registered_object_pointcloud":
            from multipass_object import build_registered_object_cloud
            def progress(value, message, metrics=None): _write(job_path, {"state":"running","kind":kind,"progress":value,"message":message,"device":"cpu","started_at":started, **({"metrics":metrics} if metrics else {})})
            metrics = build_registered_object_cloud(Path(session_dir), Path(artifact_dir), progress=progress)
            _write(job_path, {"state":"done","kind":kind,"progress":100,"message":"DONE","device":"cpu","metrics":metrics,"finished_at":time.time()})
        elif kind == "object_tsdf":
            from object_tsdf import build_object_tsdf

            def progress(value, message, metrics=None):
                _write(
                    job_path,
                    {
                        "state": "running",
                        "kind": kind,
                        "progress": value,
                        "message": message,
                        "device": "cpu",
                        "started_at": started,
                        **({"metrics": metrics} if metrics else {}),
                    },
                )

            metrics = build_object_tsdf(
                Path(session_dir), Path(artifact_dir), progress=progress
            )
            _write(job_path, {"state":"done","kind":kind,"progress":100,"message":"DONE","device":"cpu","metrics":metrics,"finished_at":time.time()})
        elif kind == "object_nksr":
            from nksr_backend import build_object_nksr

            def progress(value, message, metrics=None):
                _write(job_path, {"state":"running","kind":kind,"progress":value,"message":message,"device":device,"started_at":started, **({"metrics":metrics} if metrics else {})})

            metrics = build_object_nksr(Path(session_dir), Path(artifact_dir), progress=progress)
            _write(job_path, {"state":"done","kind":kind,"progress":100,"message":"DONE","device":metrics.get("device",device),"metrics":metrics,"finished_at":time.time()})
        elif kind in {"object_poisson", "object_bpa"}:
            from object_surface import build_object_bpa, build_object_poisson

            def progress(value, message, metrics=None):
                _write(job_path, {"state":"running","kind":kind,"progress":value,"message":message,"device":"cpu","started_at":started, **({"metrics":metrics} if metrics else {})})

            builder = build_object_poisson if kind == "object_poisson" else build_object_bpa
            metrics = builder(Path(session_dir), Path(artifact_dir), progress=progress)
            _write(job_path, {"state":"done","kind":kind,"progress":100,"message":"DONE","device":"cpu","metrics":metrics,"finished_at":time.time()})
        elif kind == "mesh":
            from reconstruct_tsdf import main as tsdf_main

            _write(job_path, {"state": "running", "kind": kind, "progress": 20, "message": "CPU TSDF reconstruction", "device": "cpu", "started_at": started})
            rc = tsdf_main([session_dir, "--output-prefix", str(Path(artifact_dir) / "mesh")])
            if rc:
                raise RuntimeError(f"TSDF command returned {rc}")
            _write(job_path, {"state": "done", "kind": kind, "progress": 100, "message": "Mesh complete", "device": "cpu", "finished_at": time.time()})
        else:
            raise ValueError("unsupported job kind")
    except BaseException as exc:
        _write(job_path, {"state": "failed", "kind": kind, "progress": 100, "message": (str(exc).strip() or exc.__class__.__name__)[:512], "device": device, "finished_at": time.time()})


class JobManager:
    def __init__(self, artifacts_root: Path, sessions_root: Path):
        self.artifacts_root = Path(artifacts_root)
        self.sessions_root = Path(sessions_root)
        self.processes: dict[str, mp.Process] = {}

    def start(self, session_id: str, kind: str, device: str = "auto") -> dict[str, object]:
        if kind not in {"pointcloud", "mesh", "object_pointcloud", "registered_object_pointcloud", "object_tsdf", "object_nksr", "object_poisson", "object_bpa"}:
            raise ValueError("unsupported job kind")
        session_dir = self.sessions_root / f"session_{session_id}"
        if not session_dir.is_dir():
            raise ValueError("verified session does not exist")
        if kind in {"object_pointcloud", "registered_object_pointcloud", "object_tsdf", "object_nksr", "object_poisson", "object_bpa"}:
            try:
                metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("verified session has invalid session.json") from exc
            if metadata.get("scan_mode") != "object":
                raise ValueError("Build Object Point Cloud requires an Object Scan session")
            if kind == "registered_object_pointcloud" and len(metadata.get("passes") or []) < 2:
                raise ValueError("Registered Object Cloud requires at least two completed passes")
            if kind == "object_nksr" and not nksr_diagnostics().get("available"):
                raise ValueError(f"NKSR unavailable: {nksr_diagnostics().get('reason', 'capability probe failed')}")
        artifact_dir = self.artifacts_root / f"session_{session_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        active = self.processes.get(session_id)
        if active and active.is_alive():
            raise ValueError("a job is already running for this session")
        selected = "cuda" if device == "auto" and device_diagnostics().get("torch_cuda_available") else ("cpu" if device == "auto" else device)
        process = mp.get_context("spawn").Process(target=_worker, args=(str(session_dir), str(artifact_dir), kind, selected), daemon=True)
        process.start()
        self.processes[session_id] = process
        return self.status(session_id)

    def status(self, session_id: str) -> dict[str, object]:
        path = self.artifacts_root / f"session_{session_id}" / "job.json"
        if not path.is_file():
            return {"state": "ready", "progress": 0, "message": "No job started"}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"state": "failed", "progress": 100, "message": "Invalid job status"}
