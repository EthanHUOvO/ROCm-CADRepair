import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args()

    errors = []
    checked = 0
    for part_dir in sorted(Path(args.outputs_root).glob("part_*")):
        geometry = read_csv(part_dir / "geometry_eval.csv")
        chamfer = read_csv(part_dir / "chamfer_eval.csv")
        visual = read_csv(part_dir / "multiview_eval.csv")
        best_path = part_dir / "best_candidate.json"
        if geometry.empty or chamfer.empty or not best_path.exists():
            errors.append(f"{part_dir.name}: missing geometry, Chamfer, or best-candidate output")
            continue

        required = {"candidate", "candidate_sha256", "gt_sha256"}
        for name, frame in (("geometry", geometry), ("chamfer", chamfer)):
            missing = required - set(frame.columns)
            if missing:
                errors.append(f"{part_dir.name}: {name} missing columns {sorted(missing)}")

        if not required.issubset(geometry.columns) or not required.issubset(chamfer.columns):
            continue

        merged = geometry[list(required)].merge(
            chamfer[list(required)],
            on="candidate",
            suffixes=("_geometry", "_chamfer"),
            how="outer",
            indicator=True,
        )
        unmatched = merged[merged["_merge"] != "both"]
        if not unmatched.empty:
            errors.append(
                f"{part_dir.name}: candidate sets differ: {unmatched['candidate'].tolist()}"
            )
        hash_mismatch = merged[
            (merged["_merge"] == "both")
            & (
                (merged["candidate_sha256_geometry"] != merged["candidate_sha256_chamfer"])
                | (merged["gt_sha256_geometry"] != merged["gt_sha256_chamfer"])
            )
        ]
        if not hash_mismatch.empty:
            errors.append(
                f"{part_dir.name}: hash mismatch: {hash_mismatch['candidate'].tolist()}"
            )

        if not visual.empty:
            visual_required = {"candidate", "candidate_sha256", "target_views_sha256"}
            missing = visual_required - set(visual.columns)
            if missing:
                errors.append(f"{part_dir.name}: visual metrics missing columns {sorted(missing)}")
            else:
                visual_merged = geometry[["candidate", "candidate_sha256"]].merge(
                    visual[["candidate", "candidate_sha256"]],
                    on="candidate",
                    suffixes=("_geometry", "_visual"),
                    how="outer",
                    indicator=True,
                )
                visual_bad = visual_merged[
                    (visual_merged["_merge"] != "both")
                    | (
                        visual_merged["candidate_sha256_geometry"]
                        != visual_merged["candidate_sha256_visual"]
                    )
                ]
                if not visual_bad.empty:
                    errors.append(
                        f"{part_dir.name}: visual metric mismatch: "
                        f"{visual_bad['candidate'].tolist()}"
                    )

        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_candidate = int(best["candidate"])
        best_geom = geometry[geometry["candidate"] == best_candidate]
        if best_geom.empty:
            errors.append(f"{part_dir.name}: best candidate {best_candidate} is absent from geometry")
        elif best.get("candidate_sha256") != best_geom.iloc[0]["candidate_sha256"]:
            errors.append(f"{part_dir.name}: best-candidate hash does not match geometry row")
        if not visual.empty:
            best_visual = visual[visual["candidate"] == best_candidate]
            if best_visual.empty or best.get("candidate_sha256") != best_visual.iloc[0].get(
                "candidate_sha256"
            ):
                errors.append(f"{part_dir.name}: best-candidate hash does not match visual row")
        if best.get("metric_integrity") != "same_candidate_verified":
            errors.append(f"{part_dir.name}: best candidate is not marked integrity-verified")
        checked += 1

    if errors:
        print("Run integrity audit failed:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    print(f"Run integrity audit passed for {checked} parts.")


if __name__ == "__main__":
    main()
