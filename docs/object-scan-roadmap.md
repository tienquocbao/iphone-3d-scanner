# Object Scan roadmap

Object Scan keeps the iPhone data contract identical to Scene Scan: synchronized
RGB, depth, confidence, intrinsics, and `world_from_camera` are captured locally
and transferred through V2 only after local STOP/finalization.

## Gate A — single-pass foreground point cloud

One stationary, non-green object is scanned against a mostly green background.
Windows calibrates a green HSV model from frame borders, writes selected mask
diagnostics, invalidates background depth before backprojection, and exports
`object_raw.ply` plus conservatively filtered `object_clean.ply`. This gate does
not register flipped passes or create meshes. Green object regions can be
mistaken for background and are reported as a warning.

## Gate B — multi-pass registered point cloud

Object passes have explicit, contiguous global frame ranges. Each pass is
foreground-processed independently in its own ARKit pass-world frame. Pass 0
is initially canonical. Later clean clouds are globally registered with
FPFH/RANSAC, refined with point-to-plane ICP, and rejected unless configured
fitness/RMSE thresholds pass. Accepted `object_from_pass` transforms are stored
separately from immutable per-frame `world_from_camera` poses.

## Later gates

- Gate C: object TSDF/NKSR and reconstruction comparison.
- Gate D: hand robustness, mesh cleanup, texture, and production exports.
