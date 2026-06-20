import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--results-dir", default="docs/results")
    args = parser.parse_args()

    rows = []
    for part_dir in sorted(Path(args.outputs_root).glob("part_*")):
        for path in sorted(part_dir.glob("infer_candidate_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["sample"] = part_dir.name
            rows.append(data)

    if not rows:
        raise RuntimeError(f"No inference logs found under {args.outputs_root}")

    frame = pd.DataFrame(rows)
    overall = {
        "num_inferences": len(frame),
        "avg_inference_time_sec": frame["inference_time_sec"].mean(),
        "min_inference_time_sec": frame["inference_time_sec"].min(),
        "max_inference_time_sec": frame["inference_time_sec"].max(),
        "avg_peak_vram_gb": frame["peak_vram_gb"].mean(),
        "max_peak_vram_gb": frame["peak_vram_gb"].max(),
    }
    if "generated_tokens_per_sec" in frame.columns:
        overall["avg_generated_tokens_per_sec"] = frame["generated_tokens_per_sec"].mean()
        overall["max_generated_tokens_per_sec"] = frame["generated_tokens_per_sec"].max()
    if "peak_reserved_vram_gb" in frame.columns:
        overall["avg_peak_reserved_vram_gb"] = frame["peak_reserved_vram_gb"].mean()
        overall["max_peak_reserved_vram_gb"] = frame["peak_reserved_vram_gb"].max()

    by_part = frame.groupby("sample").agg(
        num_inferences=("candidate", "count"),
        avg_inference_time_sec=("inference_time_sec", "mean"),
        avg_peak_vram_gb=("peak_vram_gb", "mean"),
    ).reset_index()
    if "generated_tokens_per_sec" in frame.columns:
        token_stats = frame.groupby("sample").agg(
            avg_generated_tokens_per_sec=("generated_tokens_per_sec", "mean")
        ).reset_index()
        by_part = by_part.merge(token_stats, on="sample", how="left")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / "all_inference_logs.csv", index=False)
    pd.DataFrame([overall]).to_csv(results_dir / "inference_overall_stats.csv", index=False)
    by_part.to_csv(results_dir / "inference_by_part.csv", index=False)
    print("Saved inference stats to:", results_dir)


if __name__ == "__main__":
    main()
