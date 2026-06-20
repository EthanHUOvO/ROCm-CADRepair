import argparse
from pathlib import Path

import pandas as pd


def read_csv_if_exists(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_pct(value):
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_float(value, digits=3):
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="docs/results")
    parser.add_argument("--out-csv", default="docs/results/competition_report.csv")
    parser.add_argument("--out-md", default="docs/results/competition_report.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    overall = read_csv_if_exists(results_dir / "overall_8parts_summary.csv")
    inference = read_csv_if_exists(results_dir / "inference_by_part.csv")

    if overall.empty:
        raise RuntimeError(
            "Missing overall_8parts_summary.csv. Run src/collect_overall_results.py first."
        )

    report = overall.copy()
    if not inference.empty:
        report = report.merge(
            inference,
            left_on="sample",
            right_on="sample",
            how="left",
            suffixes=("", "_infer"),
        )

    quality_components = []
    if "best_bbox_mean_error" in report.columns:
        quality_components.append(
            ((1 - report["best_bbox_mean_error"]).clip(0, 1), 0.20)
        )
    if "best_volume_error" in report.columns:
        quality_components.append(
            ((1 - report["best_volume_error"]).clip(0, 1), 0.15)
        )
    if "best_normalized_chamfer_l2_squared" in report.columns:
        chamfer_score = 1 - (report["best_normalized_chamfer_l2_squared"] / 0.05)
        quality_components.append((chamfer_score.clip(0, 1), 0.25))
    if "best_mean_visual_score" in report.columns:
        quality_components.append((report["best_mean_visual_score"].clip(0, 1), 0.20))
    if "best_mean_edge_iou" in report.columns:
        quality_components.append((report["best_mean_edge_iou"].clip(0, 1), 0.20))

    if quality_components:
        numerator = sum(series.fillna(0) * weight for series, weight in quality_components)
        denominator = sum(series.notna().astype(float) * weight for series, weight in quality_components)
        report["geometry_score"] = numerator / denominator.mask(denominator == 0)
    else:
        report["geometry_score"] = pd.NA

    if "pipeline_success_rate" in report.columns:
        report["cad_repair_score"] = report["pipeline_success_rate"]
    else:
        report["cad_repair_score"] = pd.NA

    if all(c in report.columns for c in ["cad_repair_score", "geometry_score"]):
        report["overall_demo_score"] = (
            0.4 * report["cad_repair_score"] + 0.6 * report["geometry_score"]
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    total_candidates = int(report["candidates"].sum()) if "candidates" in report else 0
    total_success = int(report["pipeline_success"].sum()) if "pipeline_success" in report else 0
    overall_success_rate = total_success / total_candidates if total_candidates else float("nan")

    avg_latency = (
        report["avg_inference_time_sec"].mean()
        if "avg_inference_time_sec" in report.columns
        else float("nan")
    )
    avg_tokens = (
        report["avg_generated_tokens_per_sec"].mean()
        if "avg_generated_tokens_per_sec" in report.columns
        else float("nan")
    )
    avg_vram = (
        report["avg_peak_vram_gb"].mean()
        if "avg_peak_vram_gb" in report.columns
        else float("nan")
    )

    lines = [
        "# ROCm CAD Repair Competition Report",
        "",
        "## Overall",
        "",
        f"- Samples: {len(report)}",
        f"- Candidates: {total_candidates}",
        f"- Pipeline success: {total_success}/{total_candidates} ({fmt_pct(overall_success_rate)})",
        f"- Average inference latency: {fmt_float(avg_latency, 2)} sec",
        f"- Average generated throughput: {fmt_float(avg_tokens, 2)} tokens/sec",
        f"- Average peak VRAM: {fmt_float(avg_vram, 2)} GB",
        "",
        "## Per-Part Summary",
        "",
        "| sample | best | success | bbox | volume | chamfer | view IoU | edge IoU | geometry score | latency(s) | tokens/s | VRAM(GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in report.iterrows():
        lines.append(
            "| {sample} | {best} | {success} | {bbox} | {vol} | {chamfer} | {view} | {edge} | {geometry} | {lat} | {tok} | {vram} |".format(
                sample=row.get("sample", "n/a"),
                best=int(row.get("best_candidate")) if not pd.isna(row.get("best_candidate", pd.NA)) else "n/a",
                success=fmt_pct(row.get("pipeline_success_rate", float("nan"))),
                bbox=fmt_float(row.get("best_bbox_mean_error", float("nan"))),
                vol=fmt_float(row.get("best_volume_error", float("nan"))),
                chamfer=fmt_float(row.get("best_normalized_chamfer_l2_squared", float("nan"))),
                view=fmt_float(row.get("best_mean_view_iou", float("nan"))),
                edge=fmt_float(row.get("best_mean_edge_iou", float("nan"))),
                geometry=fmt_float(row.get("geometry_score", float("nan"))),
                lat=fmt_float(row.get("avg_inference_time_sec", float("nan")), 2),
                tok=fmt_float(row.get("avg_generated_tokens_per_sec", float("nan")), 2),
                vram=fmt_float(row.get("avg_peak_vram_gb", float("nan")), 2),
            )
        )

    lines.extend(
        [
            "",
            "## How To Use In The Paper",
            "",
            "- Use latency, tokens/sec and VRAM to support the ROCm performance and resource-analysis section.",
            "- Use raw vs safe repair counts to show the engineering contribution beyond plain model generation.",
            "- Use candidate-consistent bbox, volume, Chamfer, view IoU and edge IoU to discuss fidelity and failure cases.",
            "- Add rocprof and rocm-smi screenshots/logs for kernel-level evidence on the AMD test platform.",
            "",
        ]
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", out_md)


if __name__ == "__main__":
    main()
