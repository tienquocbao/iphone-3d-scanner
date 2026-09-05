"""Test-only subprocess adapter; it does not implement or simulate NKSR geometry."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import open3d as o3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result = Path(args.result)
    if args.probe:
        result.write_text(json.dumps({"available":True,"nksr_version":"fake-adapter","torch_version":"fake","cuda_available":True,"gpu_name":"Fake GPU","gpu_vram_bytes":4*1024**3,"supported_modes":["auto","full","chunk","cpu"]}), encoding="utf-8")
        return 0
    mode = os.environ.get("IPHONE3D_FAKE_NKSR_MODE", "success")
    if mode == "timeout":
        time.sleep(30)
    if mode == "failure":
        result.write_text(json.dumps({"status":"failed","error":"deliberate adapter failure"}), encoding="utf-8")
        return 2
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=10)
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    result.write_text(json.dumps({"status":"success","mode":"chunk","device":"cuda","vertices":len(mesh.vertices),"triangles":len(mesh.triangles),"input_scale_to_nksr":20.0,"color_status":"unavailable","checkpoint":"fake adapter fixture","attempts":[{"mode":"full","device":"cuda","status":"oom"},{"mode":"chunk","device":"cuda","status":"success"}]}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
