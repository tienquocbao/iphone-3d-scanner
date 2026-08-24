# Phase 2C TSDF mesh reconstruction

Phase 2C integrates the original synchronized RGB-D frames directly into an
Open3D `ScalableTSDFVolume` using the stored ARKit poses. It does not use the
Phase 2B point cloud as input, and it does not perform ICP, pose optimization,
smoothing, hole filling, or watertight conversion.

## Environment and baseline command

```powershell
conda activate iphone3d
python windows/reconstruction/reconstruct_tsdf.py `
  samples/session_<id> `
  --min-confidence 1 `
  --voxel-length 0.005 `
  --sdf-trunc 0.025
```

The default uses every frame in deterministic order. `--min-depth`,
`--max-depth`, `--every-n`, and `--max-frames` are available for controlled
experiments. `--show` displays the cleaned mesh; `--show-raw-mesh` and
`--show-tsdf-cloud` expose intermediate results.

## Coordinate and image conventions

The persisted transform is `world_from_arkit_camera`. Open3D's RGB-D camera
uses optical coordinates (+X right, +Y down, +Z forward), while ARKit uses
(+X right, +Y up, -Z forward). The authoritative conversion is:

```text
world_from_cv = world_from_arkit_camera @ diag(1, -1, -1, 1)
open3d_extrinsic = inverse(world_from_cv)
```

The extrinsic is therefore world-to-optical-camera, as required by the legacy
Open3D TSDF integration API. Depth remains Float32 meters with `depth_scale=1`.
RGB is explicitly sampled into the depth resolution using the validated Phase
2A depth-pixel-to-RGB mapping; no blind image resize is used.

## Outputs

At the session root:

```text
session_tsdf_pointcloud.ply
session_mesh_raw.ply
session_mesh_clean.ply
```

Cleanup only removes duplicated vertices/triangles, degenerate triangles, and
unreferenced vertices. Generated files and captured sessions are ignored by
Git.

