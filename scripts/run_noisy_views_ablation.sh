#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
PARTS=${PARTS:-"part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008"}
NUM_CANDIDATES=${NUM_CANDIDATES:-3}
MAX_TOKENS=${MAX_TOKENS:-1024}
GPU=${GPU:-0}
NOISE=${NOISE:-10}
ROTATE=${ROTATE:-3.0}
BRIGHTNESS=${BRIGHTNESS:-0.12}
CONTRAST=${CONTRAST:-0.12}
BLUR=${BLUR:-0.5}
NOISY_DIR=${NOISY_DIR:-./examples_noisy}
OUT_ROOT=${OUT_ROOT:-./outputs_ablation/noisy_views}
REPORT_ROOT=${REPORT_ROOT:-./assistant_reports/noisy_views}

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}" docs/results

"${PY}" src/create_noisy_views.py \
  --examples-dir examples \
  --outdir "${NOISY_DIR}" \
  --noise "${NOISE}" \
  --rotate "${ROTATE}" \
  --brightness "${BRIGHTNESS}" \
  --contrast "${CONTRAST}" \
  --blur "${BLUR}"

for mode in clean noisy; do
  echo "===== View robustness ablation: ${mode} ====="
  for part in ${PARTS}; do
    if [ "${mode}" = "clean" ]; then
      views="./examples/${part}/views"
    else
      views="${NOISY_DIR}/${part}/views"
    fi

    PART="${part}" \
    MODEL="${MODEL}" \
    PY="${PY}" \
    NUM_CANDIDATES="${NUM_CANDIDATES}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    GPU="${GPU}" \
    ENABLE_CHAMFER=0 \
    VIEWS="${views}" \
    OUT="${OUT_ROOT}/${mode}/${part}" \
    REPORT_DIR="${REPORT_ROOT}/${mode}" \
    bash scripts/run_local_cad_assistant.sh "${part}"
  done
done

"${PY}" src/summarize_ablation.py \
  --root "${OUT_ROOT}" \
  --label-name view_mode \
  --out docs/results/noisy_view_ablation.csv

echo "Saved: docs/results/noisy_view_ablation.csv"
