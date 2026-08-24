"""Load and strictly validate one Phase 1B/1C iPhone frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


class FrameValidationError(ValueError):
    """Raised when a frame does not satisfy the persisted sensor contract."""


@dataclass(frozen=True)
class FrameData:
    frame_dir: Path
    metadata: dict[str, Any]
    rgb: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise FrameValidationError(f"Missing {context}.{key}")
    return mapping[key]


def _safe_file(frame_dir: Path, relative_name: str, field: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name:
        raise FrameValidationError(f"{field} must be a non-empty relative filename")
    path = (frame_dir / relative_name).resolve()
    root = frame_dir.resolve()
    if path.parent != root or path.name != relative_name:
        raise FrameValidationError(f"{field} must name a file directly inside the frame directory")
    if not path.is_file():
        raise FrameValidationError(f"Missing file: {path.name}")
    return path


def _matrix(value: Any, shape: tuple[int, int], field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise FrameValidationError(f"{field} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FrameValidationError(f"{field} contains non-finite values")
    return array


def validate_frame(frame_dir: Path) -> dict[str, Any]:
    """Validate one frame and return parsed metadata plus diagnostics."""

    frame_dir = Path(frame_dir)
    metadata_path = frame_dir / "frame.json"
    if not metadata_path.is_file():
        raise FrameValidationError(f"Missing file: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameValidationError(f"Cannot read frame.json: {exc}") from exc

    if _required(metadata, "schema_version", "frame") != 1:
        raise FrameValidationError("Unsupported schema_version; expected 1")
    rgb_meta = _required(metadata, "rgb", "frame")
    depth_meta = _required(metadata, "depth", "frame")
    confidence_meta = _required(metadata, "confidence", "frame")
    camera_meta = _required(metadata, "camera", "frame")

    rgb_width = int(_required(rgb_meta, "width", "rgb"))
    rgb_height = int(_required(rgb_meta, "height", "rgb"))
    depth_width = int(_required(depth_meta, "width", "depth"))
    depth_height = int(_required(depth_meta, "height", "depth"))
    if min(rgb_width, rgb_height, depth_width, depth_height) <= 0:
        raise FrameValidationError("RGB and depth dimensions must be positive")

    rgb_path = _safe_file(frame_dir, _required(rgb_meta, "file", "rgb"), "rgb.file")
    depth_path = _safe_file(frame_dir, _required(depth_meta, "file", "depth"), "depth.file")
    confidence_path = _safe_file(
        frame_dir,
        _required(confidence_meta, "file", "confidence"),
        "confidence.file",
    )

    rgb_image = o3d.io.read_image(str(rgb_path))
    rgb = np.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise FrameValidationError(f"RGB JPEG must decode to HxWx3/4, got {rgb.shape}")
    actual_rgb_height, actual_rgb_width = rgb.shape[:2]
    if (actual_rgb_width, actual_rgb_height) != (rgb_width, rgb_height):
        raise FrameValidationError(
            "JPEG dimensions do not match metadata: "
            f"actual={actual_rgb_width}x{actual_rgb_height}, "
            f"metadata={rgb_width}x{rgb_height}"
        )

    if _required(depth_meta, "dtype", "depth") != "float32":
        raise FrameValidationError("depth.dtype must be float32")
    if _required(depth_meta, "endianness", "depth") != "little":
        raise FrameValidationError("depth.endianness must be little")
    if _required(depth_meta, "unit", "depth") != "meters":
        raise FrameValidationError("depth.unit must be meters")
    depth_bytes = depth_path.stat().st_size
    expected_depth_bytes = depth_width * depth_height * 4
    if depth_bytes != expected_depth_bytes:
        raise FrameValidationError(
            f"depth.f32 size {depth_bytes} != expected {expected_depth_bytes}"
        )
    depth = np.fromfile(depth_path, dtype="<f4").reshape((depth_height, depth_width))

    if _required(confidence_meta, "dtype", "confidence") != "uint8":
        raise FrameValidationError("confidence.dtype must be uint8")
    confidence_bytes = confidence_path.stat().st_size
    expected_confidence_bytes = depth_width * depth_height
    if confidence_bytes != expected_confidence_bytes:
        raise FrameValidationError(
            f"confidence.u8 size {confidence_bytes} != expected {expected_confidence_bytes}"
        )
    confidence = np.fromfile(confidence_path, dtype=np.uint8).reshape((depth_height, depth_width))
    invalid_confidence = np.setdiff1d(confidence, np.array([0, 1, 2], dtype=np.uint8))
    if invalid_confidence.size:
        raise FrameValidationError(
            f"confidence contains unsupported values: {invalid_confidence.tolist()}"
        )

    camera_width = int(_required(camera_meta, "image_width", "camera"))
    camera_height = int(_required(camera_meta, "image_height", "camera"))
    if min(camera_width, camera_height) <= 0:
        raise FrameValidationError("camera image dimensions must be positive")
    if (camera_width, camera_height) != (rgb_width, rgb_height):
        raise FrameValidationError(
            "camera image resolution does not match RGB metadata: "
            f"camera={camera_width}x{camera_height}, rgb={rgb_width}x{rgb_height}"
        )
    intrinsics = _matrix(_required(camera_meta, "intrinsics_rows", "camera"), (3, 3), "intrinsics_rows")
    transform = _matrix(_required(camera_meta, "transform_rows", "camera"), (4, 4), "transform_rows")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise FrameValidationError("intrinsics fx and fy must be positive")
    if _required(camera_meta, "transform_semantics", "camera") != "world_from_camera":
        raise FrameValidationError("camera transform must be world_from_camera")
    if _required(camera_meta, "coordinate_system", "camera") != "ARKit":
        raise FrameValidationError("camera coordinate_system must be ARKit")
    if _required(camera_meta, "units", "camera") != "meters":
        raise FrameValidationError("camera units must be meters")
    if _required(camera_meta, "forward_axis", "camera") != "-Z":
        raise FrameValidationError("camera forward_axis must be -Z")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-4):
        raise FrameValidationError("transform final row must be approximately [0, 0, 0, 1]")

    rgb_aspect = rgb_width / rgb_height
    depth_aspect = depth_width / depth_height
    if not np.isclose(rgb_aspect, depth_aspect, rtol=0.0, atol=1e-3):
        raise FrameValidationError(
            f"RGB/depth aspect ratios differ: {rgb_aspect:.6f} vs {depth_aspect:.6f}"
        )

    return {
        "metadata": metadata,
        "rgb": rgb,
        "depth": depth,
        "confidence": confidence,
        "intrinsics": intrinsics,
        "world_from_camera": transform,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "confidence_path": confidence_path,
        "camera_image_width": camera_width,
        "camera_image_height": camera_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "confidence_counts": {
            "low": int(np.count_nonzero(confidence == 0)),
            "medium": int(np.count_nonzero(confidence == 1)),
            "high": int(np.count_nonzero(confidence == 2)),
        },
    }


def load_frame(frame_dir: Path) -> FrameData:
    """Load and validate one frame through the authoritative parser."""

    parsed = validate_frame(Path(frame_dir))
    return FrameData(
        frame_dir=Path(frame_dir),
        metadata=parsed["metadata"],
        rgb=parsed["rgb"],
        depth=parsed["depth"],
        confidence=parsed["confidence"],
        intrinsics=parsed["intrinsics"],
        world_from_camera=parsed["world_from_camera"],
    )
