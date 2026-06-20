import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class CandidateMetricIntegrityTest(unittest.TestCase):
    def test_selection_and_audit_keep_metrics_on_one_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            part_dir = Path(temp_dir) / "outputs" / "part_001"
            part_dir.mkdir(parents=True)

            geometry = pd.DataFrame(
                [
                    {
                        "candidate": 0,
                        "candidate_stl": "candidate_0.stl",
                        "candidate_sha256": "hash0",
                        "gt_sha256": "gt-hash",
                        "watertight": True,
                        "bbox_x_error": 0.01,
                        "bbox_y_error": 0.01,
                        "bbox_z_error": 0.01,
                        "volume_error": 0.90,
                    },
                    {
                        "candidate": 1,
                        "candidate_stl": "candidate_1.stl",
                        "candidate_sha256": "hash1",
                        "gt_sha256": "gt-hash",
                        "watertight": True,
                        "bbox_x_error": 0.20,
                        "bbox_y_error": 0.20,
                        "bbox_z_error": 0.20,
                        "volume_error": 0.20,
                    },
                    {
                        "candidate": 2,
                        "candidate_stl": "candidate_2.stl",
                        "candidate_sha256": "hash2",
                        "gt_sha256": "gt-hash",
                        "watertight": True,
                        "bbox_x_error": 0.10,
                        "bbox_y_error": 0.10,
                        "bbox_z_error": 0.10,
                        "volume_error": 0.10,
                    },
                ]
            )
            chamfer = pd.DataFrame(
                [
                    {
                        "candidate": 0,
                        "candidate_stl": "candidate_0.stl",
                        "candidate_sha256": "hash0",
                        "gt_sha256": "gt-hash",
                        "normalized_chamfer_l2_squared": 0.05,
                        "fscore_02": 0.50,
                    },
                    {
                        "candidate": 1,
                        "candidate_stl": "candidate_1.stl",
                        "candidate_sha256": "hash1",
                        "gt_sha256": "gt-hash",
                        "normalized_chamfer_l2_squared": 0.001,
                        "fscore_02": 0.95,
                    },
                    {
                        "candidate": 2,
                        "candidate_stl": "candidate_2.stl",
                        "candidate_sha256": "hash2",
                        "gt_sha256": "gt-hash",
                        "normalized_chamfer_l2_squared": 0.20,
                        "fscore_02": 0.10,
                    },
                ]
            )
            visual = pd.DataFrame(
                [
                    {
                        "candidate": 0,
                        "candidate_stl": "candidate_0.stl",
                        "candidate_sha256": "hash0",
                        "target_views_sha256": "views-hash",
                        "mean_view_iou": 0.40,
                        "mean_visual_score": 0.35,
                    },
                    {
                        "candidate": 1,
                        "candidate_stl": "candidate_1.stl",
                        "candidate_sha256": "hash1",
                        "target_views_sha256": "views-hash",
                        "mean_view_iou": 0.90,
                        "mean_visual_score": 0.88,
                    },
                    {
                        "candidate": 2,
                        "candidate_stl": "candidate_2.stl",
                        "candidate_sha256": "hash2",
                        "target_views_sha256": "views-hash",
                        "mean_view_iou": 0.20,
                        "mean_visual_score": 0.18,
                    },
                ]
            )
            geometry.to_csv(part_dir / "geometry_eval.csv", index=False)
            chamfer.to_csv(part_dir / "chamfer_eval.csv", index=False)
            visual.to_csv(part_dir / "multiview_eval.csv", index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "select_best_candidate.py"),
                    "--geometry_csv",
                    str(part_dir / "geometry_eval.csv"),
                    "--chamfer_csv",
                    str(part_dir / "chamfer_eval.csv"),
                    "--visual_csv",
                    str(part_dir / "multiview_eval.csv"),
                    "--out",
                    str(part_dir / "best_candidate.json"),
                ],
                check=True,
            )

            best = json.loads((part_dir / "best_candidate.json").read_text(encoding="utf-8"))
            self.assertEqual(best["candidate"], 1)
            self.assertEqual(best["candidate_sha256"], "hash1")
            self.assertAlmostEqual(best["bbox_mean_error"], 0.20)
            self.assertAlmostEqual(best["normalized_chamfer_l2_squared"], 0.001)
            self.assertAlmostEqual(best["mean_view_iou"], 0.90)
            self.assertEqual(best["metric_integrity"], "same_candidate_verified")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "audit_run_integrity.py"),
                    "--outputs-root",
                    str(Path(temp_dir) / "outputs"),
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
