import argparse
from pathlib import Path

import pandas as pd
import torch
import trimesh


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {path}")
    return mesh


def sample_points(path, n_points, seed):
    mesh = load_mesh(path)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    except TypeError:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def nearest_squared_distance(a, b, chunk_size):
    values = []
    for start in range(0, a.shape[0], chunk_size):
        chunk = a[start:start + chunk_size]
        dist = torch.cdist(chunk, b, p=2)
        values.append(dist.min(dim=1).values.square())
    return torch.cat(values)


def chamfer_distance(gt_points, pred_points, device, chunk_size):
    gt = torch.as_tensor(gt_points, dtype=torch.float32, device=device)
    pred = torch.as_tensor(pred_points, dtype=torch.float32, device=device)

    pred_to_gt = nearest_squared_distance(pred, gt, chunk_size).mean()
    gt_to_pred = nearest_squared_distance(gt, pred, chunk_size).mean()
    return (pred_to_gt + gt_to_pred).item()


def bbox_diag(points):
    pts = torch.as_tensor(points, dtype=torch.float32)
    size = pts.max(dim=0).values - pts.min(dim=0).values
    return float(torch.linalg.norm(size).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    uses_cuda = args.device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA device is visible to PyTorch.")

    gt_points = sample_points(args.gt, args.points, args.seed)
    diag = bbox_diag(gt_points)

    rows = []
    for stl in sorted(Path(args.pred_dir).glob("candidate*.stl")):
        pred_points = sample_points(stl, args.points, args.seed)
        chamfer = chamfer_distance(gt_points, pred_points, args.device, args.chunk_size)
        rows.append({
            "candidate_stl": str(stl),
            "points": args.points,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(0) if uses_cuda else "cpu",
            "torch_hip_version": getattr(torch.version, "hip", None),
            "chamfer_l2_squared": chamfer,
            "normalized_chamfer_l2_squared": chamfer / (diag * diag) if diag > 0 else None,
        })

    df = pd.DataFrame(rows)
    print(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
