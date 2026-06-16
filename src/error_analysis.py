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

for p in sorted(outdir.glob("verify_candidate*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "log": p.name,
        "candidate": d.get("candidate"),
        "safe_mode": d.get("safe_mode", False),
        "success": d.get("success"),
        "error_type": d.get("error_type"),
    })

df = pd.DataFrame(rows)
print(df)

if len(df) > 0:
    summary = (
        df[df["success"] == False]
        .groupby(["safe_mode", "error_type"])
        .size()
        .reset_index(name="count")
    )
else:
    summary = pd.DataFrame(columns=["safe_mode", "error_type", "count"])

print("\nError summary:")
print(summary)

df.to_csv(args.csv, index=False)
summary.to_csv(args.csv.replace(".csv", "_summary.csv"), index=False)
