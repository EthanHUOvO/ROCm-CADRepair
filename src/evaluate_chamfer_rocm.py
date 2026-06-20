import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    if mesh.faces is None or len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {path}")
    return mesh


def sample_points(path, n_points, seed):
    mesh = load_mesh(path)
    np.random.seed(seed)
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


def chamfer_metrics(gt_points, pred_points, device, chunk_size, diag):
    gt = torch.as_tensor(gt_points, dtype=torch.float32, device=device)
    pred = torch.as_tensor(pred_points, dtype=torch.float32, device=device)

    pred_to_gt_sq = nearest_squared_distance(pred, gt, chunk_size)
    gt_to_pred_sq = nearest_squared_distance(gt, pred, chunk_size)
    pred_to_gt = pred_to_gt_sq.mean()
    gt_to_pred = gt_to_pred_sq.mean()

    pred_dist = pred_to_gt_sq.sqrt()
    gt_dist = gt_to_pred_sq.sqrt()
    metrics = {
        "chamfer_pred_to_gt_squared": pred_to_gt.item(),
        "chamfer_gt_to_pred_squared": gt_to_pred.item(),
        "chamfer_l2_squared": (pred_to_gt + gt_to_pred).item(),
        "hausdorff_95_normalized": (
            max(torch.quantile(pred_dist, 0.95), torch.quantile(gt_dist, 0.95)).item() / diag
            if diag > 0
            else None
        ),
    }
    for ratio in (0.01, 0.02, 0.05):
        threshold = ratio * diag
        precision = (pred_dist <= threshold).float().mean()
        recall = (gt_dist <= threshold).float().mean()
        denom = precision + recall
        fscore = (2 * precision * recall / denom).item() if denom.item() > 0 else 0.0
        metrics[f"fscore_{int(ratio * 100):02d}"] = fscore
    return metrics


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
    parser.add_argument(
        "--alignment",
        choices=["none", "centroid"],
        default="centroid",
        help="Centroid alignment removes arbitrary translation but preserves scale and orientation.",
    )
    args = parser.parse_args()

    uses_cuda = args.device.startswith("cuda")
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA device is visible to PyTorch.")

    gt_path = Path(args.gt)
    gt_points = sample_points(gt_path, args.points, args.seed)
    if args.alignment == "centroid":
        gt_points = gt_points - gt_points.mean(axis=0, keepdims=True)
    diag = bbox_diag(gt_points)
    gt_sha256 = sha256(gt_path)

    rows = []
    for stl in sorted(Path(args.pred_dir).glob("candidate*.stl")):
        pred_points = sample_points(stl, args.points, args.seed)
        if args.alignment == "centroid":
            pred_points = pred_points - pred_points.mean(axis=0, keepdims=True)
        metrics = chamfer_metrics(gt_points, pred_points, args.device, args.chunk_size, diag)
        chamfer = metrics["chamfer_l2_squared"]
        rows.append({
            "candidate": candidate_id(stl),
            "candidate_stl": str(stl),
            "candidate_sha256": sha256(stl),
            "gt_stl": str(gt_path),
            "gt_sha256": gt_sha256,
            "points": args.points,
            "alignment": args.alignment,
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(0) if uses_cuda else "cpu",
            "torch_hip_version": getattr(torch.version, "hip", None),
            **metrics,
            "normalized_chamfer_l2_squared": chamfer / (diag * diag) if diag > 0 else None,
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
