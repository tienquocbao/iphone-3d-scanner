from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frame_io import FrameValidationError, load_frame


def format_depth_stats(depth: np.ndarray, confidence: np.ndarray, min_confidence: int) -> str:
    valid = np.isfinite(depth) & (depth > 0) & (confidence >= min_confidence)
    if not np.any(valid):
        return "  valid: 0"
    values = depth[valid]
    return (
        f"  valid: {values.size}\n"
        f"  min: {values.min():.4f} m\n"
        f"  median: {np.median(values):.4f} m\n"
        f"  max: {values.max():.4f} m"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one iPhone RGB-D frame")
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()

    try:
        frame = load_frame(args.frame_dir)
    except (FrameValidationError, OSError, ValueError) as exc:
        print(f"Validation: FAIL\nReason: {exc}")
        return 1

    metadata = frame.metadata
    camera = metadata["camera"]
    counts = {
        "low": int(np.count_nonzero(frame.confidence == 0)),
        "medium": int(np.count_nonzero(frame.confidence == 1)),
        "high": int(np.count_nonzero(frame.confidence == 2)),
    }
    total_confidence = frame.confidence.size
    print(f"Frame: {args.frame_dir.name}")
    print(f"Schema: {metadata['schema_version']}")
    print("\nRGB")
    print(f"  size: {frame.rgb.shape[1]} x {frame.rgb.shape[0]}")
    print("\nDepth")
    print(f"  size: {frame.depth.shape[1]} x {frame.depth.shape[0]}")
    print(f"  samples: {frame.depth.size}")
    print(format_depth_stats(frame.depth, frame.confidence, args.min_confidence))
    print("\nConfidence")
    for label in ("low", "medium", "high"):
        count = counts[label]
        print(f"  {label}: {count} ({100.0 * count / total_confidence:.2f}%)")
    print("\nRGB intrinsics")
    print(f"  fx: {frame.intrinsics[0, 0]:.6f}")
    print(f"  fy: {frame.intrinsics[1, 1]:.6f}")
    print(f"  cx: {frame.intrinsics[0, 2]:.6f}")
    print(f"  cy: {frame.intrinsics[1, 2]:.6f}")
    print("\nPose")
    print(f"  translation: {frame.world_from_camera[:3, 3].tolist()}")
    print(f"  semantics: {camera['transform_semantics']}")
    print("\nValidation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
