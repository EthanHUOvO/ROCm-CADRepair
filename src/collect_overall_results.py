from pathlib import Path
import pandas as pd

rows = []

for part_dir in sorted(Path("./outputs").glob("part_*")):
    p = part_dir.name
    pipeline = part_dir / "pipeline_summary.csv"
    geom = part_dir / "geometry_eval.csv"
    chamfer = part_dir / "chamfer_eval.csv"

    row = {
        "sample": p,
        "candidates": None,
        "pipeline_success": None,
        "pipeline_success_rate": None,
        "raw_selected": None,
        "safe_selected": None,
        "failed": None,
        "best_bbox_mean_error": None,
        "best_volume_error": None,
        "best_normalized_chamfer_l2_squared": None,
    }

    if pipeline.exists():
        df = pd.read_csv(pipeline)

        row["candidates"] = len(df)

        if "final_success" in df.columns and len(df) > 0:
            row["pipeline_success"] = int(df["final_success"].sum())
            row["pipeline_success_rate"] = row["pipeline_success"] / len(df)

        if "selected_stage" in df.columns:
            row["raw_selected"] = int((df["selected_stage"] == "raw").sum())
            row["safe_selected"] = int((df["selected_stage"] == "safe").sum())
            row["failed"] = int((df["selected_stage"] == "failed").sum())

    if geom.exists():
        g = pd.read_csv(geom)

        if len(g) > 0:
            if all(c in g.columns for c in ["bbox_x_error", "bbox_y_error", "bbox_z_error"]):
                g["bbox_mean_error"] = g[["bbox_x_error", "bbox_y_error", "bbox_z_error"]].mean(axis=1)
            else:
                g["bbox_mean_error"] = None

            if "watertight" in g.columns:
                g["watertight_score"] = g["watertight"].apply(lambda x: 0 if bool(x) else 1)
            else:
                g["watertight_score"] = 1

            sort_cols = ["watertight_score", "bbox_mean_error"]
            if "volume_error" in g.columns:
                sort_cols.append("volume_error")

            best = g.sort_values(sort_cols).iloc[0]

            row["best_bbox_mean_error"] = best.get("bbox_mean_error")
            row["best_volume_error"] = best.get("volume_error", None)

    if chamfer.exists():
        c = pd.read_csv(chamfer)
        if len(c) > 0 and "normalized_chamfer_l2_squared" in c.columns:
            row["best_normalized_chamfer_l2_squared"] = c[
                "normalized_chamfer_l2_squared"
            ].min()

    rows.append(row)

out = pd.DataFrame(rows)
print(out)

Path("./docs/results").mkdir(parents=True, exist_ok=True)
out.to_csv("./docs/results/overall_8parts_summary.csv", index=False)

print("Saved: ./docs/results/overall_8parts_summary.csv")
