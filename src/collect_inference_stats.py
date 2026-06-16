import json
from pathlib import Path
import pandas as pd

rows = []

for part_dir in sorted(Path("outputs").glob("part_*")):
    for f in sorted(part_dir.glob("infer_candidate_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["sample"] = part_dir.name
        rows.append(d)

if not rows:
    print("No inference logs found.")
    raise SystemExit

df = pd.DataFrame(rows)

cols = ["sample", "candidate", "inference_time_sec", "peak_vram_gb", "max_new_tokens", "temperature"]
print(df[cols if all(c in df.columns for c in cols) else df.columns])

overall = {
    "num_inferences": len(df),
    "avg_inference_time_sec": df["inference_time_sec"].mean(),
    "min_inference_time_sec": df["inference_time_sec"].min(),
    "max_inference_time_sec": df["inference_time_sec"].max(),
    "avg_peak_vram_gb": df["peak_vram_gb"].mean(),
    "max_peak_vram_gb": df["peak_vram_gb"].max(),
}

by_part = df.groupby("sample").agg(
    num_inferences=("candidate", "count"),
    avg_inference_time_sec=("inference_time_sec", "mean"),
    avg_peak_vram_gb=("peak_vram_gb", "mean"),
).reset_index()

print("\nOverall inference stats:")
for k, v in overall.items():
    print(k, v)

print("\nBy part:")
print(by_part)

Path("docs/results").mkdir(parents=True, exist_ok=True)
df.to_csv("docs/results/all_inference_logs.csv", index=False)
pd.DataFrame([overall]).to_csv("docs/results/inference_overall_stats.csv", index=False)
by_part.to_csv("docs/results/inference_by_part.csv", index=False)

print("\nSaved inference stats.")
