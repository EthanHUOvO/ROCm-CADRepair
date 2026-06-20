import argparse
from pathlib import Path

import pandas as pd


def classify(row):
    bbox = row.get("best_bbox_mean_error")
    volume = row.get("best_volume_error")
    chamfer = row.get("best_normalized_chamfer_l2_squared")
    fscore = row.get("best_fscore_02")
    view_iou = row.get("best_mean_visual_score")
    edge_iou = row.get("best_mean_edge_iou")
    if pd.isna(bbox) or pd.isna(chamfer):
        return "insufficient_geometry_evidence"
    if bbox < 0.15 and (pd.isna(volume) or volume < 0.25) and chamfer < 0.01 and (
        pd.isna(fscore) or fscore >= 0.75
    ) and (
        pd.isna(view_iou) or view_iou >= 0.75
    ) and (
        pd.isna(edge_iou) or edge_iou >= 0.55
    ):
        return "high_quality"
    if (
        bbox < 0.35
        and (pd.isna(volume) or volume < 0.75)
        and chamfer < 0.04
        and (pd.isna(view_iou) or view_iou >= 0.45)
        and (pd.isna(edge_iou) or edge_iou >= 0.30)
    ):
        return "medium_quality"
    return "low_quality"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/results/overall_8parts_summary.csv")
    parser.add_argument("--out", default="docs/results/geometry_quality_level.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["geometry_quality_level"] = frame.apply(classify, axis=1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(frame[["sample", "best_candidate", "geometry_quality_level"]])
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
