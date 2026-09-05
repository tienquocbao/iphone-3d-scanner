"""Gate A single-pass green-screen object point cloud reconstruction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from foreground import ForegroundConfig, calibrate_green_background, foreground_mask, project_rgb_mask_to_depth
from frame_io import FrameData, load_frame
from fuse_session import SessionValidationError, _cloud, session_frame_dirs, write_cloud
from geometry import depth_to_world_points, sample_rgb


@dataclass(frozen=True)
class ObjectCloudConfig:
    min_confidence: int = 1
    min_depth: float | None = None
    max_depth: float | None = None
    voxel_size: float = 0.005
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    diagnostic_stride: int = 10
    foreground: ForegroundConfig = ForegroundConfig()

    def validate(self) -> None:
        if self.min_confidence not in (0, 1, 2):
            raise ValueError("min_confidence must be 0, 1, or 2")
        if self.voxel_size <= 0 or self.outlier_neighbors < 1 or self.outlier_std_ratio <= 0 or self.diagnostic_stride < 1:
            raise ValueError("invalid object cloud filtering configuration")
        self.foreground.validate()


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _masked_points(frame: FrameData, mask: np.ndarray, config: ObjectCloudConfig) -> tuple[np.ndarray, np.ndarray, int, int]:
    points, pixels, stats = depth_to_world_points(frame, config.min_confidence, config.min_depth, config.max_depth)
    before_mask = int(stats["valid_samples"])
    if not len(points):
        return points, np.empty((0, 3), dtype=np.float64), before_mask, 0
    keep = project_rgb_mask_to_depth(mask, frame.depth.shape)[pixels[:, 1], pixels[:, 0]]
    kept_pixels = pixels[keep]
    return points[keep], sample_rgb(frame, kept_pixels), before_mask, int(np.count_nonzero(keep))


def build_object_cloud(session_dir: Path, artifact_dir: Path, config: ObjectCloudConfig = ObjectCloudConfig(), progress=None, selected_frame_dirs=None, output_dir: Path | None = None, mask_dir: Path | None = None, raw_name: str = "object_raw.ply", clean_name: str = "object_clean.ply") -> dict[str, object]:
    """Build object-only raw/clean clouds without changing immutable raw sessions."""

    config.validate()
    started = time.perf_counter()
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if metadata.get("scan_mode") != "object":
        raise SessionValidationError("Build Object Point Cloud requires scan_mode=object")
    frame_dirs = tuple(selected_frame_dirs) if selected_frame_dirs is not None else session_frame_dirs(session_dir)
    if progress:
        progress(10, "CALIBRATING BACKGROUND")
    calibration_started = time.perf_counter()
    model = calibrate_green_background((load_frame(directory).rgb for directory in frame_dirs), config.foreground)
    calibration_seconds = time.perf_counter() - calibration_started
    object_dir = Path(output_dir) if output_dir is not None else artifact_dir / "object"
    masks_dir = Path(mask_dir) if mask_dir is not None else artifact_dir / "object" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    rejected: list[dict[str, str]] = []
    diagnostic_masks: list[str] = []
    foreground_ratios: list[float] = []
    before_mask_total = foreground_total = 0
    mask_seconds = backprojection_seconds = 0.0
    for index, frame_dir in enumerate(frame_dirs):
        frame = load_frame(frame_dir)
        mask_started = time.perf_counter()
        mask = foreground_mask(frame.rgb, model, config.foreground)
        mask_seconds += time.perf_counter() - mask_started
        foreground_ratios.append(float(mask.mean()))
        if index % config.diagnostic_stride == 0 or index == len(frame_dirs) - 1:
            mask_name = f"{frame_dir.name}.png"
            cv2.imwrite(str(masks_dir / mask_name), (mask.astype(np.uint8) * 255))
            diagnostic_masks.append(mask_name)
        if progress:
            progress(15 + int(35 * (index + 1) / len(frame_dirs)), "MASKING")
        point_started = time.perf_counter()
        points, colors, before_mask, foreground_count = _masked_points(frame, mask, config)
        backprojection_seconds += time.perf_counter() - point_started
        before_mask_total += before_mask
        foreground_total += foreground_count
        if not len(points):
            rejected.append({"frame": frame_dir.name, "reason": "no foreground points after mask and confidence filtering"})
            continue
        all_points.append(points)
        all_colors.append(colors)
    if not all_points:
        raise SessionValidationError("Object foreground extraction produced zero points")
    if progress:
        progress(60, "BACKPROJECTING")
    raw = _cloud(np.concatenate(all_points), np.concatenate(all_colors))
    raw_path = object_dir / raw_name
    write_cloud(raw_path, raw)
    if progress:
        progress(75, "FILTERING")
    filtering_started = time.perf_counter()
    downsampled = raw.voxel_down_sample(config.voxel_size)
    if not len(downsampled.points):
        raise SessionValidationError("Object voxel downsampling produced zero points")
    neighbors = min(config.outlier_neighbors, len(downsampled.points) - 1)
    clean = downsampled
    if neighbors >= 1:
        filtered, _ = downsampled.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=config.outlier_std_ratio)
        if len(filtered.points):
            clean = filtered
    filtering_seconds = time.perf_counter() - filtering_started
    if progress:
        progress(90, "EXPORTING")
    clean_path = object_dir / clean_name
    write_cloud(clean_path, clean)
    warnings: list[str] = []
    ratio_array = np.asarray(foreground_ratios)
    if np.median(ratio_array) < 0.005 or np.std(ratio_array) > 0.35:
        warnings.append("Foreground segmentation may be removing object regions. Object may contain colors similar to the calibrated background.")
    processing = {
        "scan_mode": "object",
        "input_frames": len(frame_dirs),
        "processed_frames": len(all_points),
        "rejected_frames": rejected,
        "background_model": model.to_dict(),
        "foreground_ratio_median": float(np.median(ratio_array)),
        "foreground_ratio_stddev": float(np.std(ratio_array)),
        "points_before_mask": before_mask_total,
        "foreground_points": foreground_total,
        "points_after_downsample": len(downsampled.points),
        "clean_points": len(clean.points),
        "warnings": warnings,
        "timing_seconds": {"calibration": calibration_seconds, "masking": mask_seconds, "backprojection": backprojection_seconds, "filtering_export": filtering_seconds, "total": time.perf_counter() - started},
        "config": {**asdict(config), "foreground": asdict(config.foreground)},
        "outputs": {"raw": str(raw_path.relative_to(artifact_dir)).replace("\\", "/"), "clean": str(clean_path.relative_to(artifact_dir)).replace("\\", "/"), "masks": str(masks_dir.relative_to(artifact_dir)).replace("\\", "/"), "diagnostic_masks": diagnostic_masks},
    }
    _write_json(object_dir / "processing.json", processing)
    if progress:
        progress(100, "DONE", processing)
    return processing
