import argparse
from pathlib import Path

import pandas as pd


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_part(label_name, label_value, part_dir):
    pipeline = read_csv(part_dir / "pipeline_summary.csv")
    geometry = read_csv(part_dir / "geometry_eval.csv")
    inference_rows = []

    for path in sorted(part_dir.glob("infer_candidate_*.json")):
        try:
            inference_rows.append(pd.read_json(path, typ="series").to_dict())
        except ValueError:
            pass
    inference = pd.DataFrame(inference_rows)

    row = {
        label_name: label_value,
        "part": part_dir.name,
        "candidates": len(pipeline),
        "pipeline_success": None,
        "pipeline_success_rate": None,
        "raw_success": None,
        "safe_success": None,
        "failed": None,
        "best_bbox_mean_error": None,
        "best_volume_error": None,
        "best_normalized_chamfer_l2_squared": None,
        "avg_inference_time_sec": None,
        "avg_generated_tokens_per_sec": None,
        "max_peak_vram_gb": None,
    }

    if not pipeline.empty:
        row["pipeline_success"] = int(pipeline["final_success"].sum())
        row["pipeline_success_rate"] = row["pipeline_success"] / len(pipeline)
        row["raw_success"] = int((pipeline["selected_stage"] == "raw").sum())
        row["safe_success"] = int((pipeline["selected_stage"] == "safe").sum())
        row["failed"] = int((pipeline["selected_stage"] == "failed").sum())

    if not geometry.empty and all(
        col in geometry.columns for col in ["bbox_x_error", "bbox_y_error", "bbox_z_error"]
    ):
        geometry["bbox_mean_error"] = geometry[
            ["bbox_x_error", "bbox_y_error", "bbox_z_error"]
        ].mean(axis=1)
        if "watertight" in geometry.columns:
            geometry["watertight_score"] = geometry["watertight"].apply(
                lambda x: 0 if bool(x) else 1
            )
        else:
            geometry["watertight_score"] = 1

        sort_cols = ["watertight_score", "bbox_mean_error"]
        if "volume_error" in geometry.columns:
            sort_cols.append("volume_error")
        best = geometry.sort_values(sort_cols).iloc[0]
        row["best_bbox_mean_error"] = best.get("bbox_mean_error")
        row["best_volume_error"] = best.get("volume_error")

    chamfer = read_csv(part_dir / "chamfer_eval.csv")
    if not chamfer.empty and "normalized_chamfer_l2_squared" in chamfer.columns:
        row["best_normalized_chamfer_l2_squared"] = chamfer[
            "normalized_chamfer_l2_squared"
        ].min()

    if not inference.empty:
        if "inference_time_sec" in inference.columns:
            row["avg_inference_time_sec"] = inference["inference_time_sec"].mean()
        if "generated_tokens_per_sec" in inference.columns:
            row["avg_generated_tokens_per_sec"] = inference[
                "generated_tokens_per_sec"
            ].mean()
        if "peak_vram_gb" in inference.columns:
            row["max_peak_vram_gb"] = inference["peak_vram_gb"].max()

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--label-name", default="setting")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for setting_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = setting_dir.name
        if args.label_prefix and label.startswith(args.label_prefix):
            label = label[len(args.label_prefix):]
        for part_dir in sorted(p for p in setting_dir.glob("part_*") if p.is_dir()):
            rows.append(summarize_part(args.label_name, label, part_dir))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    if not df.empty:
        grouped = df.groupby(args.label_name).agg(
            parts=("part", "count"),
            total_candidates=("candidates", "sum"),
            total_success=("pipeline_success", "sum"),
            avg_success_rate=("pipeline_success_rate", "mean"),
            avg_bbox_error=("best_bbox_mean_error", "mean"),
            avg_volume_error=("best_volume_error", "mean"),
            avg_latency_sec=("avg_inference_time_sec", "mean"),
            max_vram_gb=("max_peak_vram_gb", "max"),
        )
        grouped["overall_success_rate"] = grouped["total_success"] / grouped["total_candidates"]
        print(grouped)

    print("Saved:", out)


if __name__ == "__main__":
    main()
