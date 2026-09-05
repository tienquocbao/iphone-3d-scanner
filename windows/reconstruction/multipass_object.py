"""Gate B orchestration: per-pass foreground clouds, validated registration, canonical export."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import open3d as o3d
from fuse_session import session_frame_dirs, write_cloud
from object_cloud import build_object_cloud
from registration import ObjectRegistrationConfig, register_pass

def build_registered_object_cloud(session_dir: Path, artifact_dir: Path, progress=None, registration_config=ObjectRegistrationConfig()):
    session_dir, artifact_dir = Path(session_dir), Path(artifact_dir)
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    passes = metadata.get("passes") or []
    if metadata.get("scan_mode") != "object" or len(passes) < 2: raise ValueError("Registered Object Cloud requires an Object Scan with at least two passes")
    dirs = session_frame_dirs(session_dir); started=time.perf_counter(); object_dir=artifact_dir/"object"; pass_dir=object_dir/"passes"; mask_root=object_dir/"masks"; pass_dir.mkdir(parents=True, exist_ok=True)
    reports=[]; clouds=[]
    for item in passes:
        pid,start,end=item["id"],item["start_frame"],item["end_frame"]
        if pid != len(reports) or start > end or start != (0 if not reports else passes[len(reports)-1]["end_frame"]+1) or end >= len(dirs): raise ValueError("invalid contiguous pass metadata")
        if progress: progress(int(5+40*(pid+1)/len(passes)), f"PROCESSING PASS {pid+1}")
        report=build_object_cloud(session_dir, artifact_dir, selected_frame_dirs=dirs[start:end+1], output_dir=pass_dir, mask_dir=mask_root/f"pass_{pid:03d}", raw_name=f"pass_{pid:03d}_raw.ply", clean_name=f"pass_{pid:03d}_clean.ply")
        reports.append({"id":pid,"frames":end-start+1,"background_model":report["background_model"],"foreground_points":report["foreground_points"],"clean_points":report["clean_points"]})
        clouds.append(o3d.io.read_point_cloud(str(pass_dir/f"pass_{pid:03d}_clean.ply")))
    canonical=o3d.geometry.PointCloud(clouds[0]); raw_clouds=[o3d.io.read_point_cloud(str(pass_dir/"pass_000_raw.ply"))]; transforms=[{"id":0,"object_from_pass":np.eye(4).tolist(),"registration_status":"reference"}]; registrations=[]
    for pid in range(1,len(clouds)):
        if progress: progress(50+int(25*pid/(len(clouds)-1)), f"GLOBAL REGISTRATION {pid}→CANONICAL")
        result=register_pass(clouds[pid],canonical,registration_config); result.update({"source_pass":pid,"target":"canonical"}); registrations.append(result)
        if not result["accepted"]: raise ValueError(f"Pass {pid} registration unreliable. Scan more overlapping side geometry and retry.")
        transform=np.asarray(result["object_from_pass"]); aligned=o3d.geometry.PointCloud(clouds[pid]); aligned.transform(transform); canonical+=aligned
        raw=o3d.io.read_point_cloud(str(pass_dir/f"pass_{pid:03d}_raw.ply")); raw.transform(transform); raw_clouds.append(raw)
        transforms.append({"id":pid,"object_from_pass":result["object_from_pass"],"registration_status":"accepted","fitness":result["icp_fitness"],"rmse":result["icp_rmse"]})
    if progress: progress(80,"COMBINING")
    raw=raw_clouds[0]
    for cloud in raw_clouds[1:]: raw += cloud
    write_cloud(object_dir/"object_registered_raw.ply",raw)
    if progress: progress(90,"FILTERING")
    clean=raw.voxel_down_sample(registration_config.registration_voxel_size)
    if len(clean.points)>20: clean,_=clean.remove_statistical_outlier(nb_neighbors=min(20,len(clean.points)-1),std_ratio=2.0)
    write_cloud(object_dir/"object_registered_clean.ply",clean)
    registration_dir=object_dir/"registration"; registration_dir.mkdir(parents=True,exist_ok=True)
    transforms_payload={"canonical_pass":0,"transform_semantics":"object_from_pass; row-major matrix; p_object = object_from_pass @ p_pass","passes":transforms}; (registration_dir/"pass_transforms.json").write_text(json.dumps(transforms_payload,indent=2),encoding="utf-8")
    result={"scan_mode":"object","pass_count":len(passes),"passes":reports,"registrations":registrations,"registered_points":len(raw.points),"clean_points":len(clean.points),"processing_seconds":time.perf_counter()-started,"outputs":{"registered_raw":"object/object_registered_raw.ply","registered_clean":"object/object_registered_clean.ply","pass_transforms":"object/registration/pass_transforms.json"}}
    (registration_dir/"registration.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); (object_dir/"processing.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if progress: progress(100,"DONE",result)
    return result
