import argparse
import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from render_views_from_stl import render_views


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


def views_sha256(view_paths):
    digest = hashlib.sha256()
    for path in view_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def silhouette(path, image_size, threshold):
    image = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    return np.asarray(image) < threshold


def dilate(mask, iterations=1):
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[y:y + result.shape[0], x:x + result.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
        result = np.logical_or.reduce(neighbors)
    return result


def mask_metrics(target, prediction):
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    target_area = target.sum()
    pred_area = prediction.sum()
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / (target_area + pred_area) if target_area + pred_area else 1.0
    return float(iou), float(dice)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-views", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=248)
    parser.add_argument("--edge-threshold", type=int, default=115)
    parser.add_argument("--max-faces", type=int, default=12000)
    args = parser.parse_args()

    target_paths = [Path(args.target_views) / f"view_{index}.png" for index in range(8)]
    missing = [str(path) for path in target_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing target views: {missing}")
    target_masks = [silhouette(path, args.image_size, args.threshold) for path in target_paths]
    target_edges = [
        dilate(silhouette(path, args.image_size, args.edge_threshold)) for path in target_paths
    ]
    target_hash = views_sha256(target_paths)

    rows = []
    for stl in sorted(Path(args.pred_dir).glob("candidate_*.stl")):
        with tempfile.TemporaryDirectory() as temp_dir:
            render_views(
                stl,
                temp_dir,
                image_size=args.image_size,
                dpi=100,
                max_faces=args.max_faces,
            )
            ious = []
            dices = []
            edge_ious = []
            for index, (target_mask, target_edge) in enumerate(zip(target_masks, target_edges)):
                pred_path = Path(temp_dir) / f"view_{index}.png"
                pred_mask = silhouette(
                    pred_path,
                    args.image_size,
                    args.threshold,
                )
                iou, dice = mask_metrics(target_mask, pred_mask)
                pred_edge = dilate(silhouette(pred_path, args.image_size, args.edge_threshold))
                edge_iou, _ = mask_metrics(target_edge, pred_edge)
                ious.append(iou)
                dices.append(dice)
                edge_ious.append(edge_iou)

        row = {
            "candidate": candidate_id(stl),
            "candidate_stl": str(stl),
            "candidate_sha256": sha256(stl),
            "target_views_sha256": target_hash,
            "mean_view_iou": float(np.mean(ious)),
            "min_view_iou": float(np.min(ious)),
            "mean_view_dice": float(np.mean(dices)),
            "mean_edge_iou": float(np.mean(edge_ious)),
            "min_edge_iou": float(np.min(edge_ious)),
            "mean_visual_score": float(0.60 * np.mean(ious) + 0.40 * np.mean(edge_ious)),
        }
        for index, value in enumerate(ious):
            row[f"view_{index}_iou"] = value
            row[f"view_{index}_edge_iou"] = edge_ious[index]
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("candidate")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(
        frame[
            [
                "candidate",
                "mean_view_iou",
                "mean_edge_iou",
                "mean_visual_score",
                "min_view_iou",
            ]
        ]
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
