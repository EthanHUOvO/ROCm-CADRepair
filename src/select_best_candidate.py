import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--geometry_csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = pd.read_csv(args.geometry_csv)

if len(df) == 0:
    raise RuntimeError("No candidates found in geometry csv.")

for col in ["bbox_x_error", "bbox_y_error", "bbox_z_error", "volume_error"]:
    if col not in df.columns:
        df[col] = None

df["bbox_mean_error"] = df[["bbox_x_error", "bbox_y_error", "bbox_z_error"]].mean(axis=1)
df["watertight_score"] = df["watertight"].apply(lambda x: 0 if bool(x) else 1)

df_sorted = df.sort_values(
    by=["watertight_score", "bbox_mean_error", "volume_error"],
    ascending=[True, True, True],
)

best = df_sorted.iloc[0]
print("Best candidate:")
print(best)

Path(args.out).write_text(best.to_json(indent=2, force_ascii=False), encoding="utf-8")
print("Saved:", args.out)
