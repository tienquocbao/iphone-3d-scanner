"""CPU reference and optional PyTorch CUDA backprojection backends."""

from __future__ import annotations

import numpy as np

from frame_io import FrameData
from geometry import depth_to_world_points, scale_intrinsics


def depth_to_world_points_backend(frame: FrameData, device: str = "cpu", min_confidence: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return Phase-2-compatible world points and depth-pixel indexes.

    CUDA uses PyTorch only for depth filtering/backprojection/world transforms;
    color sampling and Open3D export remain format-compatible CPU operations.
    """
    if device == "cpu":
        points, pixels, _ = depth_to_world_points(frame, min_confidence=min_confidence)
        return points, pixels
    if device != "cuda":
        raise ValueError("device must be cpu or cuda")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CUDA backend") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")
    intrinsics = scale_intrinsics(frame)
    depth = torch.as_tensor(frame.depth, device="cuda", dtype=torch.float64)
    confidence = torch.as_tensor(frame.confidence, device="cuda")
    valid = torch.isfinite(depth) & (depth > 0) & (confidence >= min_confidence)
    v, u = torch.where(valid)
    if not len(u):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    z = depth[v, u]
    x = (u.to(torch.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (v.to(torch.float64) - intrinsics.cy) * z / intrinsics.fy
    camera = torch.stack((x, -y, -z), dim=1)
    rotation = torch.as_tensor(frame.world_from_camera[:3, :3], device="cuda", dtype=torch.float64)
    translation = torch.as_tensor(frame.world_from_camera[:3, 3], device="cuda", dtype=torch.float64)
    points = (camera @ rotation.T + translation).cpu().numpy()
    pixels = torch.stack((u, v), dim=1).cpu().numpy().astype(np.int32)
    return points, pixels
