import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/results/overall_8parts_summary.csv")
    parser.add_argument("--results-dir", default="docs/results")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["raw_success_rate"] = frame["raw_selected"] / frame["candidates"]
    frame["safe_gain"] = frame["safe_selected"] / frame["candidates"]
    frame["success_gain"] = frame["pipeline_success_rate"] - frame["raw_success_rate"]

    total_candidates = frame["candidates"].sum()
    total_raw = frame["raw_selected"].sum()
    total_safe = frame["safe_selected"].sum()
    total_success = frame["pipeline_success"].sum()
    total_failed = frame["failed"].sum()
    summary = {
        "total_candidates": total_candidates,
        "total_raw_success": total_raw,
        "total_safe_success": total_safe,
        "total_pipeline_success": total_success,
        "total_failed": total_failed,
        "raw_success_rate": total_raw / total_candidates,
        "pipeline_success_rate": total_success / total_candidates,
        "absolute_gain": total_success / total_candidates - total_raw / total_candidates,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / "success_gain_analysis.csv", index=False)
    pd.DataFrame([summary]).to_csv(results_dir / "overall_success_summary.csv", index=False)
    print(pd.DataFrame([summary]))


if __name__ == "__main__":
    main()
