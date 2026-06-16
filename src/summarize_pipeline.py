import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True)
parser.add_argument("--csv", required=True)
args = parser.parse_args()

outdir = Path(args.outdir)
rows = []

for p in sorted(outdir.glob("pipeline_candidate_*.json")):
    rows.append(json.loads(p.read_text(encoding="utf-8")))

df = pd.DataFrame(rows)

if len(df) > 0:
    df = df.sort_values("candidate")

print(df)

if len(df) > 0:
    print("\nPipeline success:", int(df["final_success"].sum()), "/", len(df))
    print("\nSelected stages:")
    print(df["selected_stage"].value_counts())

df.to_csv(args.csv, index=False)
print("Saved:", args.csv)
