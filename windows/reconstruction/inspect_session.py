from __future__ import annotations

import argparse
import json
from pathlib import Path

from frame_io import FrameValidationError, validate_frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an iPhone scan session")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    session_dir = args.session_dir
    try:
        session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        frame_dirs = sorted((session_dir / "frames").glob("[0-9][0-9][0-9][0-9][0-9][0-9]"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Session: FAIL\nReason: {exc}")
        return 1

    valid = 0
    expected_names = [f"{index:06d}" for index in range(len(frame_dirs))]
    sequential = [path.name for path in frame_dirs] == expected_names
    for frame_dir in frame_dirs:
        try:
            validate_frame(frame_dir)
            valid += 1
        except (FrameValidationError, OSError, ValueError):
            pass

    total_size = sum(path.stat().st_size for path in session_dir.rglob("*") if path.is_file())
    print("Session")
    print(f"  status: {session.get('status', 'unknown')}")
    print(f"  frames metadata: {session.get('frame_count', 'unknown')}")
    print(f"  frame directories: {len(frame_dirs)}")
    print(f"  sequential indexes: {'yes' if sequential else 'no'}")
    print(f"  duration: {session.get('duration_seconds', 'unknown')} seconds")
    print(f"  size: {total_size} bytes")
    print("\nFrames")
    print(f"  valid: {valid}")
    print(f"  invalid: {len(frame_dirs) - valid}")
    return 0 if valid == len(frame_dirs) and sequential and session.get("frame_count") == len(frame_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
