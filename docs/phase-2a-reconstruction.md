# Phase 2A Windows reconstruction

Phase 2A reconstructs one synchronized iPhone RGB-D frame into a world-space Open3D point cloud. It does not fuse multiple frames or create a mesh.

## Environment

```powershell
conda env update -f environment.yml
conda activate iphone3d
```

The pipeline uses NumPy, Open3D, and the Python standard library. OpenCV and Pillow are not required.

## Copy data from iPhone

Copy a completed session from:

```text
Files -> On My iPhone -> ScannerApp -> Scans
```

into `samples/`. Generated sessions and PLY outputs are ignored by Git.

## Validate a frame

```powershell
python windows/reconstruction/validate_frame.py `
    samples/session_<id>/frames/000000
```

The validator checks the Phase 1B schema, JPEG dimensions, binary sizes, confidence values, matrices, coordinate semantics, and depth statistics.

## Reconstruct a frame

```powershell
python windows/reconstruction/reconstruct_frame.py `
    samples/session_<id>/frames/000000 `
    --min-confidence 1 `
    --output samples/session_<id>/frames/000000/pointcloud_world.ply
```

Optional filters are `--min-depth`, `--max-depth`, and `--min-confidence`. The command reads the PLY back with Open3D after writing it. Add `--show` when a desktop Open3D viewer is available.

## Inspect a session

```powershell
python windows/reconstruction/inspect_session.py `
    samples/session_<id>
```

Depth pixels use CV optical coordinates first, then convert explicitly to ARKit camera coordinates (`x, -y, -z`) before applying the stored `world_from_camera` matrix. RGB pixels are sampled from the native-orientation JPEG using depth-pixel-center mapping.
