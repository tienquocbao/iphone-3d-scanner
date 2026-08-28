"""Isolated realtime processor for live scan sessions.

This module is intentionally independent from the HTTP receiver.  The child
process receives only immutable frame indexes and reads the staged files
itself, so Open3D failures cannot take down the transfer server.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Any


def _error_text(exc: BaseException) -> str:
    return (str(exc).strip() or exc.__class__.__name__)[:512]


def _process_frame(staging: Path, session_id: str, frame_index: int, state: dict[str, Any]) -> dict[str, Any]:
    # These imports must remain in the worker process.
    from frame_io import load_frame
    from geometry import depth_to_world_points
    from tsdf import TSDFPolicy, prepare_tsdf_frame, write_point_cloud
    import open3d as o3d

    frame = load_frame(staging / "frames" / f"{frame_index:06d}")
    points, _, _ = depth_to_world_points(frame, min_confidence=1)
    if state["volume"] is None:
        policy = TSDFPolicy()
        state["volume"] = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=policy.voxel_length,
            sdf_trunc=policy.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
    prepared = prepare_tsdf_frame(frame, TSDFPolicy())
    state["volume"].integrate(prepared.rgbd, prepared.intrinsic, prepared.extrinsic)
    state["processed"] += 1
    state["raw_points"] += len(points)
    if state["processed"] % 10 == 0:
        preview_path = staging.parent / ".live" / session_id / "preview_pointcloud.ply"
        write_point_cloud(preview_path, state["volume"].extract_point_cloud())
    elapsed = max(time.monotonic() - state["started"], 1e-6)
    return {
        "ok": True,
        "frame_index": frame_index,
        "processed_frames": state["processed"],
        "raw_points": state["raw_points"],
        "processing_fps": state["processed"] / elapsed,
    }


def processor_main(session_id: str, staging: str, commands: Any, statuses: Any, mode: str) -> None:
    """Child entry point. ``mode`` is test-only and never enabled by CLI."""
    if mode == "crash":
        # Wait for a real frame command before simulating a native processor exit.
        commands.get()
        os._exit(70)
    if mode == "hang":
        commands.get()
        while True:
            time.sleep(60)

    state: dict[str, Any] = {"volume": None, "processed": 0, "raw_points": 0, "started": time.monotonic()}
    root = Path(staging)
    while True:
        command = commands.get()
        if command == "stop":
            return
        if not isinstance(command, tuple) or command[0] != "frame":
            continue
        frame_index = int(command[1])
        try:
            statuses.put(_process_frame(root, session_id, frame_index, state))
        except Exception as exc:
            statuses.put({"ok": False, "frame_index": frame_index, "error": _error_text(exc)})


class LiveProcessor:
    """Parent-side controller for one bounded worker process."""

    def __init__(self, session_id: str, staging: Path, mode: str = "normal") -> None:
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=1)
        self.statuses = context.Queue(maxsize=1)
        self.process = context.Process(
            target=processor_main,
            args=(session_id, str(staging), self.commands, self.statuses, mode),
            name=f"live-processor-{session_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.process.start()

    def submit(self, frame_index: int) -> None:
        self.commands.put(("frame", frame_index))

    def poll(self, timeout: float = 0.1) -> dict[str, Any] | None:
        try:
            return self.statuses.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def stop(self, timeout: float = 2.0) -> str:
        if not self.process.is_alive():
            self.process.join(timeout=0)
            return "already_exited"
        try:
            self.commands.put("stop", timeout=0.2)
        except queue.Full:
            pass
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        if self.process.is_alive() and hasattr(self.process, "kill"):
            self.process.kill()
            self.process.join(timeout=1.0)
        return "terminated" if self.process.exitcode not in (0, None) else "stopped"
