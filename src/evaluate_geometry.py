import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import trimesh


CANDIDATE_RE = re.compile(r"candidate_(\d+)$")


def candidate_id(path):
    match = CANDIDATE_RE.match(Path(path).stem)
    if not match:
        raise ValueError(f"Cannot extract candidate id from: {path}")
    return int(match.group(1))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def mesh_stats(path):
    mesh = load_mesh(path)
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]

    return {
        "file": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_x": float(size[0]),
        "bbox_y": float(size[1]),
        "bbox_z": float(size[2]),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "watertight": bool(mesh.is_watertight),
    }


def rel_err(a, b):
    if b == 0 or b is None or a is None:
        return None
    return abs(a - b) / abs(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gt_path = Path(args.gt)
    gt = mesh_stats(gt_path)
    gt_sha256 = sha256(gt_path)
    rows = []

    for stl in sorted(Path(args.pred_dir).glob("candidate*.stl")):
        pred = mesh_stats(stl)

        rows.append({
            "candidate": candidate_id(stl),
            "candidate_stl": str(stl),
            "candidate_sha256": sha256(stl),
            "gt_stl": str(gt_path),
            "gt_sha256": gt_sha256,
            "vertices": pred["vertices"],
            "faces": pred["faces"],
            "watertight": pred["watertight"],
            "bbox_x": pred["bbox_x"],
            "bbox_y": pred["bbox_y"],
            "bbox_z": pred["bbox_z"],
            "gt_bbox_x": gt["bbox_x"],
            "gt_bbox_y": gt["bbox_y"],
            "gt_bbox_z": gt["bbox_z"],
            "bbox_x_error": rel_err(pred["bbox_x"], gt["bbox_x"]),
            "bbox_y_error": rel_err(pred["bbox_y"], gt["bbox_y"]),
            "bbox_z_error": rel_err(pred["bbox_z"], gt["bbox_z"]),
            "volume": pred["volume"],
            "gt_volume": gt["volume"],
            "volume_error": rel_err(pred["volume"], gt["volume"]),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("candidate")
    print(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
