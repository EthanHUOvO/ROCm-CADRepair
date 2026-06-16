from pathlib import Path
import pandas as pd

rows = []

for p in sorted(Path("outputs").glob("part_*")):
    f = p / "error_analysis.csv"
    if not f.exists():
        continue

    df = pd.read_csv(f)
    df["sample"] = p.name
    rows.append(df)

if not rows:
    print("No error_analysis.csv found.")
    raise SystemExit

all_df = pd.concat(rows, ignore_index=True)

failed = all_df[all_df["success"] == False].copy()

summary = (
    failed.groupby(["safe_mode", "error_type"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

overall = (
    failed.groupby(["error_type"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("Error summary by mode:")
print(summary)

print("\nOverall error summary:")
print(overall)

Path("docs/results").mkdir(parents=True, exist_ok=True)
all_df.to_csv("docs/results/all_error_analysis.csv", index=False)
summary.to_csv("docs/results/error_summary_by_mode.csv", index=False)
overall.to_csv("docs/results/error_summary_overall.csv", index=False)

print("\nSaved:")
print("docs/results/all_error_analysis.csv")
print("docs/results/error_summary_by_mode.csv")
print("docs/results/error_summary_overall.csv")
