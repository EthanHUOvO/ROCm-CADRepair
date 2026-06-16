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

    if "best_bbox_mean_error" in report.columns:
        report["geometry_score"] = (1 - report["best_bbox_mean_error"]).clip(lower=0)
    else:
        report["geometry_score"] = pd.NA

    if "pipeline_success_rate" in report.columns:
        report["cad_repair_score"] = report["pipeline_success_rate"]
    else:
        report["cad_repair_score"] = pd.NA

    if all(c in report.columns for c in ["cad_repair_score", "geometry_score"]):
        report["overall_demo_score"] = (
            0.6 * report["cad_repair_score"] + 0.4 * report["geometry_score"]
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
        "| sample | success rate | raw | safe repair | failed | bbox error | volume error | chamfer | avg latency(s) | tokens/s | peak VRAM(GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in report.iterrows():
        lines.append(
            "| {sample} | {success} | {raw} | {safe} | {failed} | {bbox} | {vol} | {chamfer} | {lat} | {tok} | {vram} |".format(
                sample=row.get("sample", "n/a"),
                success=fmt_pct(row.get("pipeline_success_rate", float("nan"))),
                raw=int(row.get("raw_selected", 0)) if not pd.isna(row.get("raw_selected", pd.NA)) else "n/a",
                safe=int(row.get("safe_selected", 0)) if not pd.isna(row.get("safe_selected", pd.NA)) else "n/a",
                failed=int(row.get("failed", 0)) if not pd.isna(row.get("failed", pd.NA)) else "n/a",
                bbox=fmt_float(row.get("best_bbox_mean_error", float("nan"))),
                vol=fmt_float(row.get("best_volume_error", float("nan"))),
                chamfer=fmt_float(row.get("best_normalized_chamfer_l2_squared", float("nan"))),
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
            "- Use bbox, volume error and Chamfer distance to discuss geometric fidelity and failure cases.",
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
