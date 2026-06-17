import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value, digits=3):
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def find_best_candidate(part_dir):
    best = read_json(part_dir / "best_candidate.json")
    if best:
        stl = best.get("candidate_stl", "")
        stem = Path(stl).stem
        if stem.startswith("candidate_"):
            best["candidate"] = int(stem.split("_")[-1])
        return best

    geometry = read_csv(part_dir / "geometry_eval.csv")
    if geometry.empty:
        return {}

    geometry["bbox_mean_error"] = geometry[
        ["bbox_x_error", "bbox_y_error", "bbox_z_error"]
    ].mean(axis=1)
    geometry["watertight_score"] = geometry["watertight"].apply(
        lambda x: 0 if bool(x) else 1
    )
    sort_cols = ["watertight_score", "bbox_mean_error"]
    if "volume_error" in geometry.columns:
        sort_cols.append("volume_error")
    row = geometry.sort_values(sort_cols).iloc[0].to_dict()
    stem = Path(row.get("candidate_stl", "")).stem
    if stem.startswith("candidate_"):
        row["candidate"] = int(stem.split("_")[-1])
    return row


def build_part_report(part, outputs_root, outdir):
    part_dir = Path(outputs_root) / part
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline = read_csv(part_dir / "pipeline_summary.csv")
    geometry = read_csv(part_dir / "geometry_eval.csv")
    chamfer = read_csv(part_dir / "chamfer_eval.csv")
    errors = read_csv(part_dir / "error_analysis.csv")
    best = find_best_candidate(part_dir)

    infer_rows = []
    for p in sorted(part_dir.glob("infer_candidate_*.json")):
        d = read_json(p)
        if d:
            infer_rows.append(d)
    inference = pd.DataFrame(infer_rows)

    total = len(pipeline)
    success = int(pipeline["final_success"].sum()) if not pipeline.empty else 0
    raw = int((pipeline["selected_stage"] == "raw").sum()) if not pipeline.empty else 0
    safe = int((pipeline["selected_stage"] == "safe").sum()) if not pipeline.empty else 0
    failed = int((pipeline["selected_stage"] == "failed").sum()) if not pipeline.empty else 0

    best_candidate = best.get("candidate", "n/a")
    best_stl = best.get("candidate_stl", best.get("file", "n/a"))
    best_step = str(part_dir / f"candidate_{best_candidate}.step") if best_candidate != "n/a" else "n/a"

    bbox_mean = best.get("bbox_mean_error")
    if bbox_mean is None and all(k in best for k in ["bbox_x_error", "bbox_y_error", "bbox_z_error"]):
        bbox_mean = pd.Series([best["bbox_x_error"], best["bbox_y_error"], best["bbox_z_error"]]).mean()

    chamfer_value = None
    if not chamfer.empty and "normalized_chamfer_l2_squared" in chamfer.columns:
        chamfer_value = chamfer["normalized_chamfer_l2_squared"].min()

    avg_latency = inference["inference_time_sec"].mean() if "inference_time_sec" in inference else None
    avg_tokens = (
        inference["generated_tokens_per_sec"].mean()
        if "generated_tokens_per_sec" in inference
        else None
    )
    peak_vram = inference["peak_vram_gb"].max() if "peak_vram_gb" in inference else None
    gpu_name = inference["gpu_name"].dropna().iloc[0] if "gpu_name" in inference and not inference["gpu_name"].dropna().empty else "n/a"
    hip_version = inference["torch_hip_version"].dropna().iloc[0] if "torch_hip_version" in inference and not inference["torch_hip_version"].dropna().empty else "n/a"

    lines = [
        f"# Local CAD Assistant Report: {part}",
        "",
        "## Task Summary",
        "",
        f"- Input views: `examples/{part}/views/view_0.png` ... `view_7.png`",
        f"- Generated candidates: {total}",
        f"- Successful CAD exports: {success}/{total}" if total else "- Successful CAD exports: n/a",
        f"- Raw successes: {raw}",
        f"- Safe repair successes: {safe}",
        f"- Failed candidates: {failed}",
        f"- Best candidate: `candidate_{best_candidate}`",
        f"- Best STEP: `{best_step}`",
        f"- Best STL: `{best_stl}`",
        "",
        "## Geometry Quality",
        "",
        f"- Watertight: {fmt(best.get('watertight'))}",
        f"- Mean bbox relative error: {fmt(bbox_mean)}",
        f"- Volume relative error: {fmt(best.get('volume_error'))}",
        f"- Normalized Chamfer distance: {fmt(chamfer_value)}",
        "",
        "## ROCm Inference Evidence",
        "",
        f"- GPU: {gpu_name}",
        f"- HIP version: {hip_version}",
        f"- Average inference latency: {fmt(avg_latency, 2)} sec",
        f"- Average generated throughput: {fmt(avg_tokens, 2)} tokens/sec",
        f"- Peak allocated VRAM: {fmt(peak_vram, 2)} GB",
        "",
        "## Candidate Status",
        "",
        "| candidate | status | selected stage | raw error | safe error |",
        "|---:|---|---|---|---|",
    ]

    if not pipeline.empty:
        for _, row in pipeline.sort_values("candidate").iterrows():
            status = "success" if bool(row.get("final_success")) else "failed"
            lines.append(
                "| {candidate} | {status} | {stage} | {raw_error} | {safe_error} |".format(
                    candidate=int(row.get("candidate")),
                    status=status,
                    stage=row.get("selected_stage", "n/a"),
                    raw_error=fmt(row.get("raw_error_type")),
                    safe_error=fmt(row.get("safe_error_type")),
                )
            )

    lines.extend(["", "## Engineering Notes", ""])
    if total and success == 0:
        lines.append("- No candidate was exportable. The part should be re-run with a larger token budget or more candidates.")
    elif total and failed > 0:
        lines.append("- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.")
    else:
        lines.append("- All generated candidates passed export validation.")

    if bbox_mean is not None and not pd.isna(bbox_mean) and bbox_mean > 0.5:
        lines.append("- The best candidate has a large bbox error; manual dimension correction is recommended.")
    if best.get("volume_error") is not None and not pd.isna(best.get("volume_error")) and best.get("volume_error") > 1.0:
        lines.append("- The best candidate has a large volume error; check missing holes, bosses, and support features.")
    if safe > 0:
        lines.append("- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.")

    if not errors.empty:
        lines.extend(["", "## Error Distribution", "", "| error type | count |", "|---|---:|"])
        summary = errors["error_type"].value_counts().reset_index()
        summary.columns = ["error_type", "count"]
        for _, row in summary.iterrows():
            lines.append(f"| {row['error_type']} | {int(row['count'])} |")

    report_path = outdir / f"{part}_assistant_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", report_path)
    return {
        "part": part,
        "candidates": total,
        "success": success,
        "raw": raw,
        "safe": safe,
        "failed": failed,
        "best_candidate": best_candidate,
        "bbox_mean_error": bbox_mean,
        "volume_error": best.get("volume_error"),
        "normalized_chamfer": chamfer_value,
        "avg_latency_sec": avg_latency,
        "avg_tokens_per_sec": avg_tokens,
        "peak_vram_gb": peak_vram,
        "report": str(report_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=None, help="Example: part_003. If omitted, reports all part_* outputs.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="assistant_reports")
    parser.add_argument("--summary-csv", default="assistant_reports/assistant_summary.csv")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if args.part:
        parts = [args.part]
    else:
        parts = sorted(p.name for p in outputs_root.glob("part_*") if p.is_dir())

    rows = [build_part_report(part, outputs_root, args.outdir) for part in parts]
    if rows:
        summary = pd.DataFrame(rows)
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print("Saved:", summary_path)


if __name__ == "__main__":
    main()
