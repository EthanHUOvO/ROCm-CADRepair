#!/bin/bash
set -euo pipefail

PART=${1:-${PART:-part_003}}
MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
MAX_TOKENS=${MAX_TOKENS:-1024}
BASE_TEMPERATURE=${BASE_TEMPERATURE:-0.2}
TEMP_STEP=${TEMP_STEP:-0.1}
GPU=${GPU:-0}
ENABLE_CHAMFER=${ENABLE_CHAMFER:-1}
CHAMFER_POINTS=${CHAMFER_POINTS:-4096}
VIEWS=${VIEWS:-./examples/${PART}/views}
OUT=${OUT:-./outputs/${PART}}
REPORT_DIR=${REPORT_DIR:-./assistant_reports}

mkdir -p "${OUT}" "${REPORT_DIR}"

if [ ! -d "${VIEWS}" ]; then
  echo "Missing views folder: ${VIEWS}"
  exit 1
fi

if [ ! -d "${MODEL}" ]; then
  echo "Missing model folder: ${MODEL}"
  exit 1
fi

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

echo "===== Local CAD Engineering Assistant ====="
echo "PART=${PART}"
echo "MODEL=${MODEL}"
echo "VIEWS=${VIEWS}"
echo "OUT=${OUT}"
echo "GPU=${GPU}"
echo "NUM_CANDIDATES=${NUM_CANDIDATES}"
echo "MAX_TOKENS=${MAX_TOKENS}"

for i in $(seq 0 $((NUM_CANDIDATES - 1))); do
  TEMP=$("${PY}" - <<PY
base = float("${BASE_TEMPERATURE}")
step = float("${TEMP_STEP}")
i = int("${i}")
print(round(base + step * i, 2))
PY
)

  echo "----- Candidate ${i}, temperature=${TEMP} -----"

  "${PY}" src/infer_one.py \
    --model "${MODEL}" \
    --views "${VIEWS}" \
    --outdir "${OUT}" \
    --candidate "${i}" \
    --max-new-tokens "${MAX_TOKENS}" \
    --do-sample \
    --temperature "${TEMP}" \
    --profile-label "assistant_${PART}_gpu${GPU}"

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

GT_STL="./examples/${PART}/${PART}_gt.stl"
if [ -f "${GT_STL}" ]; then
  "${PY}" src/evaluate_geometry.py \
    --gt "${GT_STL}" \
    --pred_dir "${OUT}" \
    --out "${OUT}/geometry_eval.csv" || true

  if [ -f "${OUT}/geometry_eval.csv" ]; then
    "${PY}" src/select_best_candidate.py \
      --geometry_csv "${OUT}/geometry_eval.csv" \
      --out "${OUT}/best_candidate.json" || true
  fi

  if [ "${ENABLE_CHAMFER}" = "1" ]; then
    "${PY}" src/evaluate_chamfer_rocm.py \
      --gt "${GT_STL}" \
      --pred-dir "${OUT}" \
      --out "${OUT}/chamfer_eval.csv" \
      --points "${CHAMFER_POINTS}" || true
  fi
fi

"${PY}" src/generate_assistant_report.py \
  --part "${PART}" \
  --outputs-root ./outputs \
  --outdir "${REPORT_DIR}" \
  --summary-csv "${REPORT_DIR}/assistant_summary.csv"

echo "Assistant report: ${REPORT_DIR}/${PART}_assistant_report.md"
