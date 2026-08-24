# Phase 2B session fusion

Phase 2B fuses the synchronized RGB-D keyframes from one completed iPhone
session into world-space Open3D point clouds. It deliberately does not perform
ICP, TSDF integration, meshing, or networking.

## Environment

```powershell
conda activate iphone3d
```

## Fuse a session

Copy a completed session from Files on the iPhone into `samples/`, then run:

```powershell
python windows/reconstruction/fuse_session.py samples/session_x
```

The command validates the session and every selected frame before fusing. The
default policy keeps confidence values 1 and 2, uses all frames, and applies a
0.005 m voxel grid. Optional `--min-depth`, `--max-depth`, `--every-n`, and
`--max-frames` controls are available for bounded experiments. Add
`--remove-outliers` for an additional statistical-cleaning output.

Outputs are written beside `session.json`:

```text
session_pointcloud_raw.ply
session_pointcloud_voxel.ply
session_pointcloud_clean.ply       # only with --remove-outliers
camera_trajectory.csv
```

The raw and voxel clouds can be opened with Open3D:

```powershell
python windows/reconstruction/fuse_session.py samples/session_x --show-raw
python windows/reconstruction/fuse_session.py samples/session_x --show
```

The geometry pipeline remains centralized in `frame_io.py` and `geometry.py`:
depth pixels are back-projected in optical coordinates, converted explicitly
to ARKit coordinates (`[x, -y, -z]`), and transformed once by the stored
`world_from_camera` matrix. RGB colors use the existing native-orientation,
pixel-center mapping.

