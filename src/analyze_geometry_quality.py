import pandas as pd
from pathlib import Path

df = pd.read_csv("docs/results/overall_8parts_summary.csv")

def classify(row):
    bbox = row["best_bbox_mean_error"]
    vol = row["best_volume_error"]

    if pd.isna(bbox):
        return "no_successful_geometry"

    if bbox < 0.25 and (pd.isna(vol) or vol < 0.5):
        return "high_quality"

    if bbox < 0.5:
        return "medium_quality"

    return "low_quality"

df["geometry_quality_level"] = df.apply(classify, axis=1)

print(df[[
    "sample",
    "pipeline_success_rate",
    "best_bbox_mean_error",
    "best_volume_error",
    "geometry_quality_level"
]])

print("\nQuality distribution:")
print(df["geometry_quality_level"].value_counts())

Path("docs/results").mkdir(parents=True, exist_ok=True)
df.to_csv("docs/results/geometry_quality_level.csv", index=False)
print("\nSaved: docs/results/geometry_quality_level.csv")
