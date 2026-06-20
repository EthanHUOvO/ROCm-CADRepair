import argparse
import json
import re
from pathlib import Path

import pandas as pd


CANDIDATE_RE = re.compile(r"candidate_(\d+)$")


def candidate_id(path):
    match = CANDIDATE_RE.match(Path(str(path)).stem)
    if not match:
        raise ValueError(f"Cannot extract candidate id from: {path}")
    return int(match.group(1))


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def add_candidate_column(frame):
    frame = frame.copy()
    if "candidate" not in frame.columns:
        frame["candidate"] = frame["candidate_stl"].map(candidate_id)
    frame["candidate"] = frame["candidate"].astype(int)
    return frame


def load_metrics(geometry_csv, chamfer_csv=None, visual_csv=None):
    geometry = add_candidate_column(pd.read_csv(geometry_csv))
    geometry["bbox_mean_error"] = geometry[
        ["bbox_x_error", "bbox_y_error", "bbox_z_error"]
    ].mean(axis=1)
    geometry["watertight"] = geometry["watertight"].map(as_bool)

    merged = geometry
    has_chamfer = False
    if chamfer_csv and Path(chamfer_csv).exists():
        chamfer = add_candidate_column(pd.read_csv(chamfer_csv))
        integrity_columns = {"candidate_sha256", "gt_sha256"}
        if not integrity_columns.issubset(geometry.columns) or not integrity_columns.issubset(
            chamfer.columns
        ):
            raise RuntimeError(
                "Missing metric hashes. Re-run evaluate_geometry.py and "
                "evaluate_chamfer_rocm.py; do not combine legacy CSV files."
            )
        keep = [
            "candidate",
            "normalized_chamfer_l2_squared",
            "hausdorff_95_normalized",
            "fscore_01",
            "fscore_02",
            "fscore_05",
            "candidate_sha256",
            "gt_sha256",
        ]
        chamfer = chamfer[[col for col in keep if col in chamfer.columns]].rename(
            columns={
                "candidate_sha256": "chamfer_candidate_sha256",
                "gt_sha256": "chamfer_gt_sha256",
            }
        )
        merged = geometry.merge(chamfer, on="candidate", how="left", validate="one_to_one")

        mismatch = merged[
            merged["candidate_sha256"].notna()
            & merged["chamfer_candidate_sha256"].notna()
            & (merged["candidate_sha256"] != merged["chamfer_candidate_sha256"])
        ]
        if not mismatch.empty:
            ids = mismatch["candidate"].tolist()
            raise RuntimeError(f"Geometry/Chamfer STL hash mismatch for candidates: {ids}")
        mismatch = merged[
            merged["gt_sha256"].notna()
            & merged["chamfer_gt_sha256"].notna()
            & (merged["gt_sha256"] != merged["chamfer_gt_sha256"])
        ]
        if not mismatch.empty:
            raise RuntimeError("Geometry and Chamfer metrics use different ground-truth files.")
        has_chamfer = merged["normalized_chamfer_l2_squared"].notna().any()
    has_visual = False
    if visual_csv and Path(visual_csv).exists():
        visual = add_candidate_column(pd.read_csv(visual_csv))
        visual_keep = [
            "candidate",
            "mean_view_iou",
            "min_view_iou",
            "mean_view_dice",
            "mean_edge_iou",
            "min_edge_iou",
            "mean_visual_score",
            "candidate_sha256",
            "target_views_sha256",
        ]
        visual = visual[[col for col in visual_keep if col in visual.columns]].rename(
            columns={"candidate_sha256": "visual_candidate_sha256"}
        )
        merged = merged.merge(visual, on="candidate", how="left", validate="one_to_one")
        if "visual_candidate_sha256" not in merged:
            raise RuntimeError("Visual metrics do not contain candidate hashes.")
        mismatch = merged[
            merged["candidate_sha256"].notna()
            & merged["visual_candidate_sha256"].notna()
            & (merged["candidate_sha256"] != merged["visual_candidate_sha256"])
        ]
        if not mismatch.empty:
            raise RuntimeError(
                f"Geometry/visual STL hash mismatch for candidates: {mismatch['candidate'].tolist()}"
            )
        has_visual = "mean_visual_score" in merged and merged["mean_visual_score"].notna().any()
    return merged, has_chamfer, has_visual


def rank_candidates(frame, has_chamfer, has_visual):
    ranked = frame.copy()
    if ranked["watertight"].any():
        ranked = ranked[ranked["watertight"]].copy()

    metric_weights = {"bbox_mean_error": 0.30, "volume_error": 0.20}
    if has_chamfer:
        metric_weights = {
            "bbox_mean_error": 0.25,
            "volume_error": 0.15,
            "normalized_chamfer_l2_squared": 0.40,
            "fscore_02_error": 0.10,
        }
        if "fscore_02" in ranked.columns:
            fscore_02 = pd.to_numeric(ranked["fscore_02"], errors="coerce").fillna(0.0)
        else:
            fscore_02 = pd.Series(0.0, index=ranked.index)
        ranked["fscore_02_error"] = 1.0 - fscore_02
        if has_visual:
            metric_weights["normalized_chamfer_l2_squared"] = 0.30
            metric_weights["visual_silhouette_error"] = 0.20
            ranked["visual_silhouette_error"] = 1.0 - pd.to_numeric(
                ranked["mean_visual_score"], errors="coerce"
            ).fillna(0.0)
    else:
        metric_weights = {"bbox_mean_error": 0.60, "volume_error": 0.40}
        if has_visual:
            metric_weights = {
                "bbox_mean_error": 0.40,
                "volume_error": 0.25,
                "visual_silhouette_error": 0.35,
            }
            ranked["visual_silhouette_error"] = 1.0 - pd.to_numeric(
                ranked["mean_visual_score"], errors="coerce"
            ).fillna(0.0)

    ranked["selection_score"] = 0.0
    for metric, weight in metric_weights.items():
        if metric in ranked.columns:
            values = pd.to_numeric(ranked[metric], errors="coerce")
        else:
            values = pd.Series(float("nan"), index=ranked.index)
        if values.notna().any():
            ranks = values.rank(method="min", pct=True, ascending=True, na_option="bottom")
            ranks = ranks.fillna(1.0)
        else:
            ranks = pd.Series(1.0, index=ranked.index)
        ranked[f"rank_{metric}"] = ranks
        ranked["selection_score"] += weight * ranks

    ranked["selection_policy"] = (
        "watertight_then_ranked_chamfer_visual_bbox_volume_fscore"
        if has_chamfer and has_visual
        else "watertight_then_ranked_chamfer_bbox_volume_fscore"
        if has_chamfer
        else "watertight_then_ranked_visual_bbox_volume"
        if has_visual
        else "watertight_then_ranked_bbox_volume"
    )
    return ranked.sort_values(
        ["selection_score", "normalized_chamfer_l2_squared", "bbox_mean_error", "candidate"]
        if has_chamfer
        else ["selection_score", "bbox_mean_error", "volume_error", "candidate"],
        na_position="last",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry_csv", required=True)
    parser.add_argument("--chamfer_csv", default=None)
    parser.add_argument("--visual_csv", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranking-out", default=None)
    args = parser.parse_args()

    metrics, has_chamfer, has_visual = load_metrics(
        args.geometry_csv, args.chamfer_csv, args.visual_csv
    )
    if metrics.empty:
        raise RuntimeError("No candidates found in geometry csv.")

    ranking = rank_candidates(metrics, has_chamfer, has_visual)
    if ranking.empty:
        raise RuntimeError("No valid candidate remains after quality filtering.")

    best = ranking.iloc[0]
    result = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in best.to_dict().items()
    }
    result["candidate"] = int(result["candidate"])
    result["metric_integrity"] = "same_candidate_verified"

    ranking_out = Path(args.ranking_out or Path(args.out).with_name("candidate_ranking.csv"))
    ranking_out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_out, index=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Candidate ranking:")
    print(ranking[["candidate", "selection_score", "bbox_mean_error", "volume_error"] + (
        ["normalized_chamfer_l2_squared", "fscore_02"] if has_chamfer else []
    )])
    print("Best candidate:", result["candidate"])
    print("Saved:", args.out)
    print("Saved:", ranking_out)


if __name__ == "__main__":
    main()
