#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
PARTS=${PARTS:-"part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008"}
CANDIDATE_COUNTS=${CANDIDATE_COUNTS:-"1 3 5 10"}
MAX_TOKENS=${MAX_TOKENS:-1024}
GPU=${GPU:-0}
ENABLE_CHAMFER=${ENABLE_CHAMFER:-0}
OUT_ROOT=${OUT_ROOT:-./outputs_ablation/candidates}
REPORT_ROOT=${REPORT_ROOT:-./assistant_reports/candidate_ablation}

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}" docs/results

for k in ${CANDIDATE_COUNTS}; do
  echo "===== Candidate-count ablation: K=${k} ====="
  for part in ${PARTS}; do
    OUT="${OUT_ROOT}/K_${k}/${part}"
    REPORT_DIR="${REPORT_ROOT}/K_${k}"

    PART="${part}" \
    MODEL="${MODEL}" \
    PY="${PY}" \
    NUM_CANDIDATES="${k}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    GPU="${GPU}" \
    ENABLE_CHAMFER="${ENABLE_CHAMFER}" \
    OUT="${OUT}" \
    REPORT_DIR="${REPORT_DIR}" \
    bash scripts/run_local_cad_assistant.sh "${part}"
  done
done

"${PY}" src/summarize_ablation.py \
  --root "${OUT_ROOT}" \
  --label-prefix K_ \
  --label-name candidate_count \
  --out docs/results/candidate_count_ablation.csv

echo "Saved: docs/results/candidate_count_ablation.csv"
