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

## Later gates

- Gate B: explicit passes, object repositioning, FPFH/RANSAC and ICP/GICP.
- Gate C: object TSDF/NKSR and reconstruction comparison.
- Gate D: hand robustness, mesh cleanup, texture, and production exports.
