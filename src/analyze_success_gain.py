import pandas as pd
from pathlib import Path

df = pd.read_csv("docs/results/overall_8parts_summary.csv")

df["raw_success_rate"] = df["raw_selected"] / df["candidates"]
df["safe_gain"] = df["safe_selected"] / df["candidates"]
df["success_gain"] = df["pipeline_success_rate"] - df["raw_success_rate"]

print(df[[
    "sample",
    "candidates",
    "raw_selected",
    "raw_success_rate",
    "safe_selected",
    "pipeline_success",
    "pipeline_success_rate",
    "success_gain"
]])

total_candidates = df["candidates"].sum()
total_raw = df["raw_selected"].sum()
total_safe = df["safe_selected"].sum()
total_success = df["pipeline_success"].sum()
total_failed = df["failed"].sum()

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

print("\nOverall:")
for k, v in summary.items():
    print(k, v)

Path("docs/results").mkdir(parents=True, exist_ok=True)
df.to_csv("docs/results/success_gain_analysis.csv", index=False)
pd.DataFrame([summary]).to_csv("docs/results/overall_success_summary.csv", index=False)
print("\nSaved: docs/results/success_gain_analysis.csv")
print("Saved: docs/results/overall_success_summary.csv")
