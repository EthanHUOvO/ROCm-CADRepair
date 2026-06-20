import argparse
import json
import re
from pathlib import Path

import pandas as pd


CANDIDATE_RE = re.compile(r"candidate_(\d+)$")


def candidate_id(path):
    match = CANDIDATE_RE.match(Path(str(path)).stem)
    return int(match.group(1)) if match else None


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def candidate_row(frame, candidate):
    if frame.empty or candidate is None:
        return None
    frame = frame.copy()
    if "candidate" not in frame.columns and "candidate_stl" in frame.columns:
        frame["candidate"] = frame["candidate_stl"].map(candidate_id)
    match = frame[frame["candidate"] == candidate]
    return match.iloc[0] if not match.empty else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="./outputs")
    parser.add_argument("--out", default="./docs/results/overall_8parts_summary.csv")
    args = parser.parse_args()

    rows = []
    for part_dir in sorted(Path(args.outputs_root).glob("part_*")):
        pipeline = read_csv(part_dir / "pipeline_summary.csv")
        geometry = read_csv(part_dir / "geometry_eval.csv")
        chamfer = read_csv(part_dir / "chamfer_eval.csv")
        visual = read_csv(part_dir / "multiview_eval.csv")
        best = read_json(part_dir / "best_candidate.json")
        candidate = best.get("candidate")
        if candidate is None:
            candidate = candidate_id(best.get("candidate_stl", ""))

        row = {
            "sample": part_dir.name,
            "candidates": len(pipeline) if not pipeline.empty else None,
            "pipeline_success": None,
            "pipeline_success_rate": None,
            "raw_selected": None,
            "safe_selected": None,
            "failed": None,
            "best_candidate": candidate,
            "best_bbox_mean_error": None,
            "best_volume_error": None,
            "best_normalized_chamfer_l2_squared": None,
            "best_fscore_02": None,
            "best_mean_view_iou": None,
            "best_mean_edge_iou": None,
            "best_mean_visual_score": None,
            "metric_integrity": best.get("metric_integrity", "not_verified"),
        }

        if not pipeline.empty:
            if "final_success" in pipeline.columns:
                success = int(pipeline["final_success"].sum())
                row["pipeline_success"] = success
                row["pipeline_success_rate"] = success / len(pipeline)
            if "selected_stage" in pipeline.columns:
                row["raw_selected"] = int((pipeline["selected_stage"] == "raw").sum())
                row["safe_selected"] = int((pipeline["selected_stage"] == "safe").sum())
                row["failed"] = int((pipeline["selected_stage"] == "failed").sum())

        geom_row = candidate_row(geometry, candidate)
        if geom_row is not None:
            if "bbox_mean_error" in geom_row:
                row["best_bbox_mean_error"] = geom_row.get("bbox_mean_error")
            elif all(col in geom_row for col in ["bbox_x_error", "bbox_y_error", "bbox_z_error"]):
                row["best_bbox_mean_error"] = geom_row[
                    ["bbox_x_error", "bbox_y_error", "bbox_z_error"]
                ].mean()
            row["best_volume_error"] = geom_row.get("volume_error")

        chamfer_row = candidate_row(chamfer, candidate)
        if chamfer_row is not None:
            row["best_normalized_chamfer_l2_squared"] = chamfer_row.get(
                "normalized_chamfer_l2_squared"
            )
            row["best_fscore_02"] = chamfer_row.get("fscore_02")

        visual_row = candidate_row(visual, candidate)
        if visual_row is not None:
            row["best_mean_view_iou"] = visual_row.get("mean_view_iou")
            row["best_mean_edge_iou"] = visual_row.get("mean_edge_iou")
            row["best_mean_visual_score"] = visual_row.get("mean_visual_score")

        rows.append(row)

    output = pd.DataFrame(rows)
    print(output)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
