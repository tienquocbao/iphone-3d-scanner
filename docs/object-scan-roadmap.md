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

## Gate C1 — deterministic object-relative TSDF mesh

Gate C1 reuses the original masked RGB-D observations rather than meshing the
registered point cloud. For every frame it composes
`object_from_camera = object_from_pass @ pass_world_from_camera`, applies the
existing ARKit-to-CV convention, and integrates into a CPU Open3D scalable TSDF
volume. The registered cloud supplies only a conservatively expanded safety
bound. Raw and conservatively cleaned colored PLY meshes are retained, and the
reconstruction records the SHA-256 of the Gate B transform artifact so stale
meshes are visible after registration changes.

## Gate C2 — optional NKSR surface reconstruction

Gate C2 prepares aligned object-frame `xyz`, per-point sensor origins, and RGB
from the same masked source observations. Joint voxel aggregation preserves
sensor provenance. A separately configured NKSR Python subprocess reconstructs
the mesh, so missing packages, CUDA failures, and timeouts cannot compromise
FastAPI or TSDF. NKSR is optional and its consistency metrics are not
ground-truth accuracy measurements. See [object-nksr.md](object-nksr.md).

## Gate C3 — Windows-native point-based surfaces

Poisson and Ball Pivoting Algorithm (BPA) reconstruct from the same canonical,
masked object observations without NKSR, CUDA, WSL, or another runtime. Their
normal preparation preserves aligned points, observing sensor origins, and
colors through joint voxel aggregation; normals are made sensor-facing before
either backend runs. Poisson is the smooth point-based candidate and is
conservatively density-trimmed and safety-cropped to the registered object
bounds. BPA derives ball radii from measured nearest-neighbor spacing and is a
geometry-preserving, possibly incomplete comparison backend. Both record the
Gate B transform SHA so changed registration marks their meshes stale. Backend
comparison reports captured-point consistency only, not ground-truth accuracy.

## Later gates

- Gate D: hand robustness, mesh cleanup, texture, and production exports.
