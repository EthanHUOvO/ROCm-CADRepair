#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
MAX_TOKENS=${MAX_TOKENS:-1024}
PY=${PY:-/opt/python/bin/python}
ENABLE_CHAMFER=${ENABLE_CHAMFER:-1}
CHAMFER_POINTS=${CHAMFER_POINTS:-4096}
ENABLE_VISUAL_EVAL=${ENABLE_VISUAL_EVAL:-1}
VISUAL_IMAGE_SIZE=${VISUAL_IMAGE_SIZE:-256}
GPU=${GPU:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-./runs/${RUN_ID}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}/outputs}
REPORT_DIR=${REPORT_DIR:-${RUN_ROOT}/assistant_reports}
RESULTS_DIR=${RESULTS_DIR:-${RUN_ROOT}/docs/results}

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}" "${RESULTS_DIR}"

if find "${OUTPUT_ROOT}" -name 'candidate_*.stl' -print -quit | grep -q .; then
  echo "Refusing to mix results: ${OUTPUT_ROOT} already contains candidate STL files."
  echo "Choose a new RUN_ID or RUN_ROOT."
  exit 2
fi

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
"${PY}" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "run_id": "${RUN_ID}",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": "${GIT_COMMIT}",
    "model": "${MODEL}",
    "num_candidates": int("${NUM_CANDIDATES}"),
    "max_tokens": int("${MAX_TOKENS}"),
    "chamfer_enabled": "${ENABLE_CHAMFER}" == "1",
    "chamfer_points": int("${CHAMFER_POINTS}"),
    "visual_evaluation_enabled": "${ENABLE_VISUAL_EVAL}" == "1",
    "visual_image_size": int("${VISUAL_IMAGE_SIZE}"),
    "gpu": int("${GPU}"),
    "output_root": "${OUTPUT_ROOT}",
}
Path("${RUN_ROOT}").mkdir(parents=True, exist_ok=True)
Path("${RUN_ROOT}/run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY

for p in part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008; do
  echo "=============================="
  echo "Running corrected benchmark for ${p}"
  echo "RUN_ID=${RUN_ID}"
  echo "=============================="

  VIEWS=./examples/${p}/views
  OUT=${OUTPUT_ROOT}/${p}
  GT_STL=./examples/${p}/${p}_gt.stl
  mkdir -p "${OUT}"

  if [ ! -d "${VIEWS}" ] || [ ! -f "${GT_STL}" ]; then
    echo "Missing views or ground truth for ${p}."
    exit 1
  fi

  for i in $(seq 0 $((NUM_CANDIDATES - 1))); do
    TEMP=$("${PY}" - <<PY
print(round(0.2 + 0.1 * int("${i}"), 2))
PY
)
    echo "Generate ${p}, candidate ${i}, temperature=${TEMP}"

    "${PY}" src/infer_one.py \
      --model "${MODEL}" \
      --views "${VIEWS}" \
      --outdir "${OUT}" \
      --candidate "${i}" \
      --max-new-tokens "${MAX_TOKENS}" \
      --do-sample \
      --temperature "${TEMP}"

    "${PY}" src/verify_pipeline.py \
      --code "${OUT}/candidate_${i}.py" \
      --outdir "${OUT}" \
      --candidate "${i}" || true
  done

  "${PY}" src/summarize_pipeline.py \
    --outdir "${OUT}" \
    --csv "${OUT}/pipeline_summary.csv"

  "${PY}" src/error_analysis.py \
    --outdir "${OUT}" \
    --csv "${OUT}/error_analysis.csv" || true

  "${PY}" src/evaluate_geometry.py \
    --gt "${GT_STL}" \
    --pred_dir "${OUT}" \
    --out "${OUT}/geometry_eval.csv"

  CHAMFER_ARGS=()
  if [ "${ENABLE_CHAMFER}" = "1" ]; then
    "${PY}" src/evaluate_chamfer_rocm.py \
      --gt "${GT_STL}" \
      --pred-dir "${OUT}" \
      --out "${OUT}/chamfer_eval.csv" \
      --points "${CHAMFER_POINTS}" \
      --alignment centroid
    CHAMFER_ARGS=(--chamfer_csv "${OUT}/chamfer_eval.csv")
  fi

  VISUAL_ARGS=()
  if [ "${ENABLE_VISUAL_EVAL}" = "1" ]; then
    "${PY}" src/evaluate_multiview_similarity.py \
      --target-views "${VIEWS}" \
      --pred-dir "${OUT}" \
      --out "${OUT}/multiview_eval.csv" \
      --image-size "${VISUAL_IMAGE_SIZE}"
    VISUAL_ARGS=(--visual_csv "${OUT}/multiview_eval.csv")
  fi

  "${PY}" src/select_best_candidate.py \
    --geometry_csv "${OUT}/geometry_eval.csv" \
    "${CHAMFER_ARGS[@]}" \
    "${VISUAL_ARGS[@]}" \
    --ranking-out "${OUT}/candidate_ranking.csv" \
    --out "${OUT}/best_candidate.json"

  "${PY}" src/render_best_candidate_comparison.py \
    --part-dir "${OUT}" \
    --target-views "${VIEWS}" \
    --out "${OUT}/best_candidate_multiview_comparison.png" \
    --image-size "${VISUAL_IMAGE_SIZE}"
done

if [ "${ENABLE_CHAMFER}" = "1" ]; then
  "${PY}" src/audit_run_integrity.py --outputs-root "${OUTPUT_ROOT}"
fi

"${PY}" src/collect_overall_results.py \
  --outputs-root "${OUTPUT_ROOT}" \
  --out "${RESULTS_DIR}/overall_8parts_summary.csv"

"${PY}" src/collect_inference_stats.py \
  --outputs-root "${OUTPUT_ROOT}" \
  --results-dir "${RESULTS_DIR}"

"${PY}" src/analyze_success_gain.py \
  --input "${RESULTS_DIR}/overall_8parts_summary.csv" \
  --results-dir "${RESULTS_DIR}"

"${PY}" src/analyze_geometry_quality.py \
  --input "${RESULTS_DIR}/overall_8parts_summary.csv" \
  --out "${RESULTS_DIR}/geometry_quality_level.csv"

"${PY}" src/build_competition_report.py \
  --results-dir "${RESULTS_DIR}" \
  --out-csv "${RESULTS_DIR}/competition_report.csv" \
  --out-md "${RESULTS_DIR}/competition_report.md"

"${PY}" src/generate_assistant_report.py \
  --outputs-root "${OUTPUT_ROOT}" \
  --outdir "${REPORT_DIR}" \
  --summary-csv "${REPORT_DIR}/assistant_summary.csv"

echo "Corrected benchmark complete."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${RESULTS_DIR}/overall_8parts_summary.csv"
