# Corrected CAD Reconstruction Experiment

The corrected workflow keeps every metric tied to one candidate STL and stores each run in an isolated directory.

## Run the full benchmark

```bash
cd /app/ROCm-CADRepair

RUN_ID=final_v2 \
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=/opt/python/bin/python \
GPU=0 \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
ENABLE_CHAMFER=1 \
CHAMFER_POINTS=8192 \
ENABLE_VISUAL_EVAL=1 \
VISUAL_IMAGE_SIZE=256 \
bash scripts/run_8parts_benchmark.sh
```

Do not reuse a `RUN_ID`. The script refuses to write into a run directory that already contains candidate STL files.

## Outputs

The command above writes to `runs/final_v2/`:

- `run_manifest.json`: run parameters and Git commit;
- `outputs/part_*/geometry_eval.csv`: candidate-specific geometry metrics and hashes;
- `outputs/part_*/chamfer_eval.csv`: candidate-specific Chamfer, F-score and hashes;
- `outputs/part_*/multiview_eval.csv`: eight-view silhouette IoU for every candidate;
- `outputs/part_*/candidate_ranking.csv`: the complete ranking table;
- `outputs/part_*/best_candidate.json`: metrics for one integrity-verified candidate;
- `outputs/part_*/best_candidate_multiview_comparison.png`: direct eight-view target/selected audit;
- `docs/results/overall_8parts_summary.csv`: corrected eight-part summary;
- `assistant_reports/`: reports that use the same candidate for every metric.

## Verify result integrity

```bash
/opt/python/bin/python src/audit_run_integrity.py \
  --outputs-root runs/final_v2/outputs
```

Only use a run in the paper when this command prints `Run integrity audit passed`.

## Interpretation

`pipeline_success_rate` measures executable/exportable CAD code. It is not reconstruction accuracy. Reconstruction quality must be reported with the selected candidate's BBox error, volume error, normalized Chamfer distance, surface F-score, eight-view silhouette IoU and rendered target-versus-candidate comparison.
